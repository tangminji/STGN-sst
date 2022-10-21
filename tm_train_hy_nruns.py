import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import time
import torch
import torch.backends.cudnn as cudnn
from common.utils import log, set_seed, generate_log_dir, AverageMeter, \
    compute_topk_accuracy, log_intermediate_iteration_stats, log_stats
from common.utils import hook_fn_random_walk,hook_fn_parameter,hook_fn_moutput
from cmd_args_sst import args
from tensorboard_logger import log_value
import torch.nn.functional as F
import numpy as np
from torch.nn.utils import clip_grad_norm_
from data.sst_dataset import get_sst_train_and_val_loader,get_SST_model_and_loss_criterion
from data.glue_dataset import get_glue_train_and_val_loader
import json
from hyperopt import STATUS_OK
import csv

from sklearn.metrics import f1_score

train_f1, dev_f1 = [], []
macro_train_f1, macro_dev_f1 = [], []

MD_CLASSES = {
    'SST': (get_sst_train_and_val_loader, get_SST_model_and_loss_criterion),
    'MNLI': (get_glue_train_and_val_loader, get_SST_model_and_loss_criterion),
    'QQP': (get_glue_train_and_val_loader, get_SST_model_and_loss_criterion),
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

    pred_lst, target_lst = [],[]

    if args.know_clean:
        for i, (data, target, target_gt, index) in enumerate(loader):
            target, target_gt = target.to(args.device), target_gt.to(args.device)
            args.sigma_dyn[index] = (args.sig_max * (target!=target_gt).float()).detach()            

    for i, (data, target, target_gt, index) in enumerate(loader):
        global_iter += 1
        # similar to global variable
        args.index = index
        
        data = {k:v.to(args.device) for k,v in data.items()}
        output = model(**data)['logits']
        target = target.to(args.device)

        args.sm = F.softmax(output)
        
        # ADD
        predictions[index] = args.sm.detach().cpu()

        # SLN
        if args.mode == 'GN_on_label':
            onehot = F.one_hot(target.long(), args.num_class).float()
            onehot += args.sigma*torch.randn(onehot.size()).to(args.device)
            loss = -torch.sum(F.log_softmax(output, dim=1)*onehot, dim=1)
        else:
            if args.mode == 'GN_on_moutput':
            # TODO: NMO: noise on model output
                output.register_hook(hook_fn_moutput)
            elif args.mode == 'Random_walk':
                output.register_hook(hook_fn_random_walk)
            loss = criterion(output, target)
            if args.mode == 'Random_walk' and not args.know_clean:
                # TODO: element1: from loss perspective
                # TODO: quantile
                loss_lst.append(loss.detach().cpu().numpy().tolist())
                if len(loss_lst) > args.avg_steps:
                    loss_lst.pop(0)
                #print('random_walk',len(loss_lst[-1]),args.drop_rate_schedule[args.cur_epoch - 1])
                losses = sum(loss_lst,[])
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

                times_ge_or_not = torch.ge(args.forgetting[index], args.forget_times).detach()
                if times_ge_or_not.any():
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
        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=3, norm_type=2)
        optimizer.step()

        args.sm = None

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
        # fc + tc <= 1.0
        fcorrect.update(fc.sum().item() * (100.0 / num), num)
        tcorrect.update(tc.sum().item() * (100.0 / num), num)
        acc1 = compute_topk_accuracy(output, target, topk=(1,))
        correct.update(acc1[0].item(), target.size(0))

        # Log stats for data parameters and loss every few iterations
        if i % args.print_freq == 0:
            log_intermediate_iteration_stats(epoch, global_iter, train_loss, top1=correct)

        if args.dataset in ["QQP"]:
            pred_lst.append(pred.detach().cpu())
            target_lst.append(target.cpu())

    f1_txt = ""
    if args.dataset in ["QQP"]:
        pred_lst = torch.cat(pred_lst)
        target_lst = torch.cat(target_lst)
        f1 = f1_score(target_lst, pred_lst)
        macro_f1 = f1_score(target_lst, pred_lst, average="macro")
        log_value("train/f1", f1, step=epoch)
        log_value("train/macro_f1", macro_f1, step=epoch)
        f1_txt = ", F1:{}, Macro:{}".format(f1, macro_f1)
        train_f1.append(f1)
        macro_train_f1.append(macro_f1)
        del pred_lst, target_lst

    # Print and log stats for the epoch
    log_value('train/loss', train_loss.avg, step=epoch)
    log(logpath, 'Time for Train-Epoch-{}/{}:{:.1f}s Acc:{}, Loss:{}{}\n'.
            format(epoch, args.epochs, time.time() - t0, correct.avg, train_loss.avg, f1_txt))

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
    pred_lst, target_lst = [], []

    with torch.no_grad():
        for i, (data, target, target_gt, _) in enumerate(loader):
            data = {k:v.to(args.device) for k,v in data.items()}
            output = model(**data)['logits']
            target = target.to(args.device)
            loss = criterion(output, target)
            loss = loss.mean()
            # measure accuracy and record loss
            test_loss.update(loss.item(), target.size(0))
            acc1 = compute_topk_accuracy(output, target, topk=(1,))
            correct.update(acc1[0].item(), target.size(0))

            if args.dataset in ["QQP"]:
                pred = output.argmax(dim=1) # no_grad
                pred_lst.append(pred.cpu())
                target_lst.append(target.cpu())

    f1_txt = ""
    if args.dataset in ["QQP"]:
        pred_lst = torch.cat(pred_lst)
        target_lst = torch.cat(target_lst)
        f1 = f1_score(target_lst, pred_lst)
        macro_f1 = f1_score(target_lst, pred_lst, average="macro")
        log_value("train/f1", f1, step=epoch)
        log_value("train/macro_f1", macro_f1, step=epoch)
        f1_txt = ", F1:{}, Macro:{}".format(f1, macro_f1)
        dev_f1.append(f1)
        macro_dev_f1.append(macro_f1)
        del pred_lst, target_lst    
    
    log(logpath, 'Time for {}-Epoch-{}/{}:{:.1f}s Acc:{}, Loss:{}{}\n'.format('Test'if mode=='test'else 'Val',
                      epoch, args.epochs, time.time()-t0, correct.avg, test_loss.avg, f1_txt))
    log_value('{}/loss'.format(mode), test_loss.avg, step=epoch)
    # Logging results on tensorboard
    log_value('{}/accuracy'.format(mode), correct.avg, step=epoch)
    return test_loss.avg, correct.avg

def main(params):
    """Objective function for Hyperparameter Optimization"""
    # Keep track of evals

    # Ablation Study
    if 'ab_sigma' in args.exp_name:
        params['sigma'] = args.sigma
    if 'ab_l' in args.exp_name:
        params['ratio_l'] = args.ratio_l
    if 'sigma' in args.exp_name:
        params['sigma'] = args.sigma
    if 'times' in args.exp_name:
        params['times'] = args.times
    if 'forget_times' in args.exp_name:
        params['forget_times'] = args.forget_times
    if 'ab_q' in args.exp_name:
        params['q'] = args.q

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
    args.logpath = args.exp_name + '/' + 'log.txt'
    
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

    # For Bert
    optimizer = torch.optim.Adam(parameters, lr=args.lr)

    # Training
    global_t0 = time.time()
    global_iter = 0
    global val_best, test_best
    val_best, test_best = 0, 0
    res_lst = []

    args.drop_rate_schedule = np.ones(args.epochs) * args.noise_rate
    args.drop_rate_schedule[:args.num_gradual] = np.linspace(0, args.noise_rate, args.num_gradual)

    for epoch in range(0, args.epochs + 1):
        args.cur_epoch = epoch
        # Test only on epoch 0
        if epoch > 0:
            global_iter, train_loss, train_acc, tc_acc, fc_acc = train_others(args, net, train_loader, optimizer, criterion,
                                                                global_iter, epoch, args.logpath)

        val_loss, val_acc = validate(args, net, val_loader, criterion_val, epoch, args.logpath, mode='val')
        test_loss, test_acc = val_loss, val_acc
        if test_loader is not None:
            test_loss, test_acc = validate(args, net, test_loader, criterion_val, epoch, args.logpath, mode='test')
        # Save checkpoint.
        if val_acc > val_best:
            val_best = val_acc
            test_best = test_acc
            torch.save(net.state_dict(), os.path.join(args.save_dir,'net.pt'))

        if epoch == 0:
            continue
        # Record val_acc for MNLI
        res_lst.append((train_acc, tc_acc, fc_acc, test_acc, test_best, train_loss, test_loss, val_acc))

        if len(noisy_ind)>0:
            log_stats(data=torch.tensor([args.sigma_dyn[i] for i in noisy_ind]),
                    name='epoch_stats_sigma_dyn_noisy',
                    step=epoch)
        if len(clean_ind)>0:
            log_stats(data=torch.tensor([args.sigma_dyn[i] for i in clean_ind]),
                    name='epoch_stats_sigma_dyn_clean',
                    step=epoch)

    run_time = time.time()-global_t0
    #save 3 types of acc
    # record best_acc/best_mae
    with open(os.path.join(args.log_dir, 'acc_loss_results.txt'), 'w', newline='') as logfile:
        logwriter = csv.writer(logfile, delimiter=',')
        logwriter.writerows(res_lst)
    
    stable_acc = sum([x[3] for x in res_lst[-5:]])/5 # avg test acc for Last five epoch

    # Val_best Test_at_val_best Stable_test_acc
    with open(os.path.join(args.log_dir, 'best_results.txt'), 'w') as outfile:
        outfile.write(f'{val_best}\t{test_best}\t{stable_acc}')
    log(args.logpath, '\nBest Acc: {}\tVal Acc: {}\t Stable Acc:{}'.format(test_best,val_best,stable_acc))
    
    if args.dataset in ["MNLI"]:
        stable_match_acc = sum([x[7] for x in res_lst[-5:]])/5
        stable_mismatch_acc = stable_acc
        log(args.logpath, '\nStable acc: Match: {}\tMismatch: {}'.format(stable_match_acc, stable_mismatch_acc))

    if args.dataset in ["QQP"]:
        best_dev_f1, best_macro_dev_f1 = max(dev_f1), max(macro_dev_f1)
        stable_dev_f1, stable_macro_dev_f1 = sum(dev_f1[-5:])/5, sum(macro_dev_f1[-5:])/5
        log(args.logpath, '\nBest F1: {}\t Macro: {}'.format(best_dev_f1, best_macro_dev_f1))
        log(args.logpath, '\nStable F1: {}\t Macro: {}'.format(stable_dev_f1, stable_macro_dev_f1))

    log(args.logpath, '\nTotal Time: {:.1f}s.\n'.format(run_time))

    loss = - test_best
    return {'loss': loss, 'best_acc': val_best, 'test_at_best': test_best, 'stable_acc': stable_acc,
            'params': params, 'train_time': run_time, 'status': STATUS_OK}
            
if __name__ == '__main__':
    print("load params from : ", args.params_path)
    params = json.load(open(args.params_path, 'r', encoding="utf-8"))['best'] if 'base' not in args.exp_name else {}
    assert params is not None
    main(params=params)