import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# #os.environ["CUDA_VISIBLE_DEVICES"] = "0"#"0,1,2"

import time
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from common.utils import log, set_seed, generate_log_dir, AverageMeter, \
    compute_topk_accuracy, checkpoint, log_intermediate_iteration_stats, log_stats, test
from common.utils import hook_fn_random_walk,hook_fn_parameter,hook_fn_moutput
from tqdm import tqdm
from cmd_args_sst import args
from tensorboard_logger import log_value
import torch.nn.functional as F
import numpy as np
from torch.nn.utils import clip_grad_norm_
from data.sst_dataset import get_sst_train_and_val_loader,get_SST_model_and_loss_criterion
import json
from hyperopt import STATUS_OK
import csv

first = True
counter = 0
sig_max = args.sig_max
delta = 0#判断是否满足增速

MD_CLASSES = {
    'SST': (get_sst_train_and_val_loader, get_SST_model_and_loss_criterion)
}

def save_predict(save_dir, predict, epoch):
    '''
    save loss for each sample in one epoch
    '''
    save_path = save_dir + '/predict_each_sample_one_eps.txt'
    with open(save_path, 'a+') as outfile:
        loss_ep = {'epoch:{}'.format(epoch):predict.tolist()}
        outfile.write('{}{}'.format(loss_ep,'\n'))

def save_loss(save_dir, loss, epoch):
    '''
    save loss for each sample in one epoch
    '''
    save_path = save_dir + '/loss_each_sample_one_eps.txt'
    with open(save_path, 'a+') as outfile:
        loss_ep = {'epoch:{}'.format(epoch):loss.tolist()}
        outfile.write('{}{}'.format(loss_ep,'\n'))

def train_others(args, model, loader, optimizer, criterion, global_iter, epoch, logpath):
    '''
    Gaussian noise on the gradient of loss w.r.t parameters
    Gaussian noise on the gradient of loss w.r.t the model output
    train_for_one_epoch
    '''
    model.train()
    train_loss = AverageMeter('Loss', ':.4e')
    correct = AverageMeter('Acc@1', ':6.2f')#for classification
    fcorrect = AverageMeter('Acc@1', ':6.2f')
    tcorrect = AverageMeter('Acc@1', ':6.2f')
    t0 = time.time()
    
    loss_parameters = torch.zeros(len(loader.dataset))
    predictions = torch.zeros(len(loader.dataset),args.num_class)
    #loss_lst = TDigest()
    loss_lst = []
    for i, (data, target, target_gt, index) in enumerate(loader):
        global_iter += 1
        # similar to global variable
        args.index = index
        # if len(target.size()) == 1:
        #     target = torch.zeros(target.size(0), args.num_class).scatter_(1, target.view(-1, 1),
        #                                                                   1)  # convert label to one-hot
        data, target = {k:v.to(args.device) for k,v in data.items()}, target.to(args.device)

        output = model(**data)['logits']
        args.sm = F.softmax(output)
        
        # ADD
        predictions[index] = args.sm.detach().cpu()

        # SLN
        if args.mode == 'GN_on_label':
            onehot = F.one_hot(target.long(), args.num_class).float()
            onehot += args.sigma*torch.randn(onehot.size()).to(args.device) # 在标签上添加onehot噪声
            loss = -torch.sum(F.log_softmax(output, dim=1)*onehot, dim=1) #交叉熵
        else:
            if args.mode == 'GN_on_moutput':
            # TODO: NMO: noise on model output
                output.register_hook(hook_fn_moutput)
            elif args.mode == 'Random_walk':
                output.register_hook(hook_fn_random_walk)
            loss = criterion(output, target)
            if args.mode == 'Random_walk':
                # TODO: element1: from loss perspective
                # TODO: quantile
                loss_lst.append(loss.detach().cpu().numpy().tolist())
                if len(loss_lst) > args.avg_steps:
                    loss_lst.pop(0)
                #print('random_walk',len(loss_lst[-1]),args.drop_rate_schedule[args.cur_epoch - 1])
                losses = sum(loss_lst,[]) #合成一个一维向量
                k1 = torch.quantile(torch.tensor(losses).to(args.device),
                                    1 - args.drop_rate_schedule[args.cur_epoch - 1])
                #TODO: element2: from forgetting events perspective, see algorithm 1 in ICLR19 an empirical study of example...
                _, predicted = torch.max(output.data, 1)
                # Update statistics and loss
                acc = (predicted == target).to(torch.long)
                forget_or_not = torch.gt(args.prev_acc[index], acc)#greater than
                args.forgetting[index] = args.forgetting[index] + forget_or_not
                args.prev_acc[index] = acc

                #when to update, since forgetting times of any sample reaches to args.forget_times

                # 这里可以用这个替代
                # if (args.forgetting>args.forget_times).any():
                times_ge_or_not = torch.ge(args.forgetting[index], args.forget_times).detach()
                if times_ge_or_not.any(): #有遗忘次数较多的样本
                # if args.forget_times in args.forgetting: #这里用in不一定很好吧
                #     #greater or equal
                #     times_ge_or_not = torch.ge(args.forgetting[index], args.forget_times).detach()
                    args.sign_forgetting_events = ((1-args.ratio_l)*args.total) * torch.tensor([1 if t == True else -1 for t in times_ge_or_not]).to(args.device)
                    args.sign_loss = (args.ratio_l * args.total) * torch.sign(loss - k1).to(args.device)
                else:
                    args.sign_forgetting_events = torch.tensor([0]*len(loss)).to(args.device)
                    if args.ratio_l != 0:
                        args.sign_loss = torch.sign(loss - k1).to(args.device)
                    else:
                        args.sign_loss = torch.tensor([0] * len(loss)).to(args.device)

        # ADD
        loss_parameters[index] = loss.detach().cpu()

        loss = loss.mean()        
        #new_l = new_l.mean()
        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=3, norm_type=2)
        optimizer.step()

        args.sm = None

        # Measure accuracy and record loss
        # train_loss.update(loss.item(), data.size(0))
        # pred = output.argmax(dim=1, keepdim=True)
        # # noise & disturbance ground-truth index
        # gt = [1 if target[i] == target_gt[i] else 0 for i in range(len(target))]
        # fc = [1 if pred[i] == target[i] and gt[i] == 0 else 0 for i in range(len(target))]  # memorize noise label
        # tc = [1 if pred[i] == target[i] and gt[i] == 1 else 0 for i in range(len(target))]  # learn clean label
        # fcorrect.update(sum(fc) * (100.0 / len(fc)), len(fc))
        # tcorrect.update(sum(tc) * (100.0 / len(tc)), len(tc))
        # acc1 = compute_topk_accuracy(output, target, topk=(1,))
        # correct.update(acc1[0].item(), data.size(0))

        # Measure accuracy and record loss
        train_loss.update(loss.item(), target.size(0))
        pred = output.argmax(dim=1)
        # noise & disturbance ground-truth index
        target_gt = target_gt.to(args.device)
        gt = target==target_gt
        agree = pred==target
        fc = agree[~gt]
        tc = agree[gt]
        num = target.size(0)
        fcorrect.update(fc.sum().item() * (100.0 / num), num)
        tcorrect.update(tc.sum().item() * (100.0 / num), num)
        # if(len(fc)>0):
        #     fcorrect.update(fc.sum().item() * (100.0 / len(fc)), len(fc))
        # if(len(tc)>0):
        #     tcorrect.update(tc.sum().item() * (100.0 / len(tc)), len(tc))
        acc1 = compute_topk_accuracy(output, target, topk=(1,))
        correct.update(acc1[0].item(), target.size(0))

        # Log stats for data parameters and loss every few iterations
        if i % args.print_freq == 0:
            log_intermediate_iteration_stats(epoch, global_iter, train_loss, top1=correct)

    # Print and log stats for the epoch
    log_value('train/loss', train_loss.avg, step=epoch)
    log(logpath, 'Time for Train-Epoch-{}/{}:{:.1f}s Acc:{}, Loss:{}\n'.
            format(epoch, args.epochs, time.time() - t0, correct.avg, train_loss.avg))

    log_value('train/accuracy', correct.avg, step=epoch)
    log_value('train/true_correct_from_clean', tcorrect.avg, step=epoch)
    log_value('train/false_correct_from_noise_disturb', fcorrect.avg, step=epoch)

    save_loss(args.save_dir, loss_parameters, epoch)
    save_predict(args.save_dir, predictions, epoch)

    return global_iter, train_loss.avg, correct.avg, tcorrect.avg, fcorrect.avg

def validate(args, model, loader, criterion, epoch, logpath, mode='val'):
    '''
    Evaluates model on validation/test set and logs score on tensorboard.
    '''
    test_loss = AverageMeter('Loss', ':.4e')
    correct = AverageMeter('Acc@1', ':6.2f')#for classification
    # switch to evaluate mode
    model.eval()
    t0 = time.time()
    with torch.no_grad():
        for i, (data, target, target_gt, _) in enumerate(loader):
            data, target = {k:v.to(args.device) for k,v in data.items()}, target.to(args.device)
            output = model(**data)['logits']
            loss = criterion(output, target)
            loss = loss.mean()
            # measure accuracy and record loss
            test_loss.update(loss.item(), target.size(0))
            acc1 = compute_topk_accuracy(output, target, topk=(1,))
            correct.update(acc1[0].item(), target.size(0))
    log(logpath, 'Time for {}-Epoch-{}/{}:{:.1f}s Acc:{}, Loss:{}\n'.format('Test'if mode=='test'else 'Val',
                      epoch, args.epochs, time.time()-t0, correct.avg, test_loss.avg))
    log_value('{}/loss'.format(mode), test_loss.avg, step=epoch)
    # Logging results on tensorboard
    log_value('{}/accuracy'.format(mode), correct.avg, step=epoch)
    return test_loss.avg, correct.avg

# 这里没必要记录数据了
def save_data(args, net, train_loader, val_loader, test_loader):
    st = time.time()
    print(f'Save sentence labels and embbeding at {args.save_dir}')
    state = torch.load(os.path.join(args.save_dir,'net.pt'))
    net.load_state_dict(state)
    net.eval()
    loaders = {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader,
    }
    args.hidden_size = 768
    with torch.no_grad():
        for type, loader in loaders.items():
            data_length = len(loader.dataset)
            embeds = torch.zeros(data_length, args.hidden_size)
            targets = torch.zeros(data_length, dtype=int)
            target_gts = torch.zeros(data_length, dtype=int)

            for data, target, target_gt, index in loader:
                data = {k:v.to(args.device) for k,v in data.items()}
                embeds[index] = net.bert(**data)['last_hidden_state'][:,0,:].cpu() #get_sentence_output
                # embeds[index] = net.bert(**data)['pooler_output'].cpu() #get_pooler_output
                targets[index] = target
                target_gts[index] = target_gt
        
            # with open(f'{args.save_dir}/{type}_embed.txt', 'w') as f:
            #     for embed, target, target_gt in zip(embeds, targets, target_gts):
            #         f.write(f"{target}\t{target_gt}\t{','.join(map(str,embed.numpy().tolist()))}\n")
            torch.save({
                'targets': targets,
                'target_gt': target_gts,
                'embed': embeds
            },os.path.join(args.save_dir,f'{type}_embed.pt'))
    ed = time.time()
    print(f'Sentence labels saved, {ed-st} sec used')

def main(params):
    """Objective function for Hyperparameter Optimization"""
    # Keep track of evals

    if 'ab_sigma' in args.exp_name:
        params['sigma'] = args.sigma
    if 'ab_l' in args.exp_name:
        params['ratio_l'] = args.ratio_l
    # For STGN
    #TODO: automatic adjustment (sig_max, lr_sig)
    if 'STGN' in args.exp_name:
        args.times = params['times']
        args.sigma = params['sigma']
        args.sig_max = 2.0 * params['sigma']
        args.lr_sig = 0.1 * params['sigma']
        #others
        args.avg_steps = params['avg_steps']
        args.ratio_l = params['ratio_l']
        args.forget_times = params['forget_times']

    #args.noise_rate = params['noise_rate']

    if 'GCE' in args.exp_name:
        args.q = params['q']

    if ('SLN' in args.exp_name) or ('GNMP' in args.exp_name) or ('GNMO' in args.exp_name):
        args.sigma = params['sigma']    
    
    if not os.path.exists(args.exp_name):
        os.makedirs(args.exp_name)
    #args.logpath = os.path.join(args.exp_name, 'log.txt')
    args.logpath = args.exp_name + '/' + 'log.txt' # 后面用/分割，所以这里要指定/
    
    args.log_dir = os.path.join(os.getcwd(), args.exp_name)
    args.save_dir = os.path.join(args.log_dir, 'weights')

    generate_log_dir(args)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)

    #should be placed after generate_log_dir()
    log(args.logpath, 'Settings: {}\n'.format(args))

    args.device = torch.device('cuda:'+str(args.gpu_id) if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(args.gpu_id)
    set_seed(args)

    loaders, mdl_loss = MD_CLASSES[args.dataset]
    # Create model
    net, criterion, criterion_val = mdl_loss(args)

    train_loader, val_loader, test_loader, noisy_ind, clean_ind = loaders(args)

    train_length = len(train_loader.dataset)

    #update perturb variance, dynamic sigma for each sample
    #暂时未写入文件保存
    args.sigma_dyn = torch.tensor([args.sigma]*train_length,
                           dtype=torch.float32,
                           requires_grad=False,
                           device=args.device)

    args.prev_acc = torch.tensor(np.zeros(train_length),
                           dtype=torch.long,
                           requires_grad=False,
                           device=args.device)
    args.forgetting = torch.tensor(np.zeros(train_length),
                                 dtype=torch.long,
                                 requires_grad=False,
                                 device=args.device)
    parameters = list(filter(lambda x:x.requires_grad, net.parameters()))
    if args.mode == 'GN_on_parameters':
        # TODO: NMP: noise on model parameters
        #https://discuss.pytorch.org/t/difference-between-state-dict-and-parameters/37531/7
        for param in parameters:
                # TODO: leaf nodes
                param.register_hook(hook_fn_parameter)

    cudnn.benchmark = True
    optimizer = torch.optim.Adam(parameters, lr=args.lr)
    # optimizer = torch.optim.RMSprop(parameters, lr=args.lr, alpha=args.decay, weight_decay=args.weight_decay)
    #optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)

    # Training
    global_t0 = time.time()
    global_iter = 0
    global val_best, test_best
    val_best, test_best = 0, 0
    res_lst = []

    args.drop_rate_schedule = np.ones(args.epochs) * args.noise_rate
    args.drop_rate_schedule[:args.num_gradual] = np.linspace(0, args.noise_rate, args.num_gradual)

    # 在train之前测试一次
    for epoch in range(0, args.epochs + 1): # epoch 0就是train之前
        args.cur_epoch = epoch
        if epoch > 0:
            global_iter, train_loss, train_acc, tc_acc, fc_acc = train_others(args, net, train_loader, optimizer, criterion,
                                                                global_iter, epoch, args.logpath)

        val_loss, val_acc = validate(args, net, val_loader, criterion_val, epoch, args.logpath, mode='val')
        test_loss, test_acc = validate(args, net, test_loader, criterion_val, epoch, args.logpath, mode='test')
        # Save checkpoint.
        if val_acc > val_best:
            val_best = val_acc
            test_best = test_acc
            # TODO 调参的时候，没有保存模型，节省存储空间
            #checkpoint(val_acc, epoch, model, args.save_dir)
            torch.save(net.state_dict(), os.path.join(args.save_dir,'net.pt'))

        if epoch == 0:
            continue
        res_lst.append((train_acc, tc_acc, fc_acc, test_acc, test_best, train_loss, test_loss))

        if len(noisy_ind)>0:
            log_stats(data=torch.tensor([args.sigma_dyn[i] for i in noisy_ind]),
                    name='epoch_stats_sigma_dyn_noisy',
                    step=epoch)
        if len(clean_ind)>0:
            log_stats(data=torch.tensor([args.sigma_dyn[i] for i in clean_ind]),
                    name='epoch_stats_sigma_dyn_clean',
                    step=epoch)

    # reload the best model
    save_data(args, net, train_loader, val_loader, test_loader)

    run_time = time.time()-global_t0
    #save 3 types of acc
    # record best_acc/best_mae
    with open(os.path.join(args.log_dir, 'acc_loss_results.txt'), 'w', newline='') as logfile:
        logwriter = csv.writer(logfile, delimiter=',')
        logwriter.writerows(res_lst)
    
    stable_acc = sum([x[3] for x in res_lst[-5:]])/5 # 最后稳定时的test_ACC

    # Val_best Test_at_val_best Stable_test_acc
    with open(os.path.join(args.log_dir, 'best_results.txt'), 'w') as outfile:
        outfile.write(f'{val_best}\t{test_best}\t{stable_acc}')
    log(args.logpath, '\nBest Acc: {}\tVal Acc: {}\t Stable Acc:{}'.format(test_best,val_best,stable_acc))
    log(args.logpath, '\nTotal Time: {:.1f}s.\n'.format(run_time))

    # loss = - val_best
    loss = - test_best
    return {'loss': loss, 'best_acc': val_best, 'test_at_best': test_best, 'stable_acc': stable_acc,
            'params': params, 'train_time': run_time, 'status': STATUS_OK}
            
if __name__ == '__main__':
    print("load params from : ", args.params_path)
    # TODO 之前的方式是载入 ['best']，这里更换是因为程序跑的总是提前停机，产生不了 best params 文件，所以手动选择要载入的参数选择文件
    params = json.load(open(args.params_path, 'r', encoding="utf-8"))['best'] if 'base' not in args.exp_name else {}
    assert params is not None
    main(params=params)