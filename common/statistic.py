import os
import matplotlib.pyplot as plt
import numpy as np
from common.utils import AverageMeter_pnorm

def plot_lines(acc_lst):
    colors = ['#1f77b4',
              '#ff7f0e',
              '#2ca02c',
              '#d62728',
              '#9467bd',
              '#8c564b',
              '#e377c2',
              '#7f7f7f',
              '#bcbd22',
              '#17becf',
              '#1a55FF']
    linestyles = ['--', '-.', '-',':']#,':'
    method = [r'GN_label_adap $\sigma=0.01$',r'GN_label_adap $\sigma=0.005$',r'GN_label_adap $\sigma=0.001$', r'GN_label $\sigma=0.5$']
    #train_acc, tc_acc, fc_acc, test_acc, test_best, train_loss, test_loss
    x_axis = [i for i in range(len(acc_lst[0]))]
    train_acc_lst_, tc_acc_lst_, fc_acc_lst_, test_acc_lst_, test_best_lst_, train_loss_lst_, test_loss_lst_ = [], [], [], [], [], [], []
    for acc in acc_lst:
        train_acc_lst, tc_acc_lst, fc_acc_lst, test_acc_lst, test_best_lst, train_loss_lst, test_loss_lst = [], [], [], [], [], [], []
        for (train_acc, tc_acc, fc_acc, test_acc, test_best, train_loss, test_loss) in acc:
            train_acc_lst.append(train_acc)
            tc_acc_lst.append(tc_acc)
            fc_acc_lst.append(fc_acc)
            test_acc_lst.append(test_acc)
            test_best_lst.append(test_best)
            train_loss_lst.append(train_loss)
            test_loss_lst.append(test_loss)
        train_acc_lst_.append(train_acc_lst)
        tc_acc_lst_.append(tc_acc_lst)
        fc_acc_lst_.append(fc_acc_lst)
        test_acc_lst_.append(test_acc_lst)
        test_best_lst_.append(test_best_lst)
        train_loss_lst_.append(train_loss_lst)
        test_loss_lst_.append(test_loss_lst)

    #axs2 = plt.subplot(231)
    plt.figure(figsize=(5, 3))
    for index in range(len(train_acc_lst_)):
        plt.plot(x_axis, train_acc_lst_[index], color=colors[index], label='{}'.format(method[index]))
    plt.legend()
    plt.xlabel('Epoch')#set_xlabel
    plt.ylabel('Train accuracy')#set_ylabel
    plt.xticks(np.arange(0, 301, 50))#set_xticks
    plt.yticks(np.arange(15, 75, 10))#set_yticks
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(5, 3))
    #axs3 = plt.subplot(232)
    for index in range(len(tc_acc_lst_)):
        plt.plot(x_axis, tc_acc_lst_[index], color=colors[index], label='{}'.format(method[index]))
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Train acc from clean labels')
    plt.xticks(np.arange(0, 301, 50))
    plt.yticks(np.arange(10, 61, 10))
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(5, 3))
    #axs4 = plt.subplot(233)
    for index in range(len(fc_acc_lst_)):
        plt.plot(x_axis, fc_acc_lst_[index], color=colors[index], label='{}'.format(method[index]))
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Train acc from noisy labels')
    plt.xticks(np.arange(0, 301, 50))
    plt.yticks(np.arange(5, 20, 5))
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(5, 3))
    #axs5 = plt.subplot(234)
    for index in range(len(test_acc_lst_)):
        plt.plot(x_axis, test_acc_lst_[index], color=colors[index], label='{}'.format(method[index]))
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Test accuracy')
    plt.xticks(np.arange(0, 301, 50))
    plt.yticks(np.arange(25, 90, 10))
    plt.grid(True)
    plt.axes([0.22, 0.3, 0.3, 0.2])
    for index in range(len(test_acc_lst_)):
        plt.plot(x_axis[250:], test_acc_lst_[index][250:], linestyle=linestyles[index], color=colors[index])
        plt.xticks(np.arange(250, 301, 10))
        plt.grid(True)
    plt.show()

    plt.figure(figsize=(5, 3))
    #axs6 = plt.subplot(235)
    for index in range(len(train_loss_lst_)):
        plt.plot(x_axis, train_loss_lst_[index], color=colors[index], label='{}'.format(method[index]))
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Train loss')
    plt.xticks(np.arange(0, 301, 50))
    plt.yticks(np.arange(0, 3, 0.5))
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(5, 3))
    #axs7 = plt.subplot(236)
    for index in range(len(test_loss_lst_)):
        plt.plot(x_axis, test_loss_lst_[index], color=colors[index], label='{}'.format(method[index]))
    plt.legend()#(loc=3, bbox_to_anchor=(1.0, 0.2), borderaxespad=0.5)
    # axs7.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Test loss')
    plt.xticks(np.arange(0, 301, 50))
    plt.yticks(np.arange(0, 3, 0.5))
    plt.grid(True)
    plt.show()
    #plt.suptitle('CIFAR-10 with 40% dependent noise')
    # plt.legend()
    # plt.savefig('CIFAR10_3_types_acc.eps', dpi=600)


def three_types_acc(log_dir_lst):
    '''
    plot 3 baselines about noise on SGD, i.e., 'GN_on_label','GN_on_moutput','GN_on_parameters'
    '''
    lst = []
    with open(os.path.join(log_dir_lst[0], 'acc_loss_results.txt'), 'r') as fl_0,\
        open(os.path.join(log_dir_lst[1], 'acc_loss_results.txt'), 'r') as fl_1,\
        open(os.path.join(log_dir_lst[2], 'acc_loss_results.txt'), 'r') as fl_2,\
        open(os.path.join(log_dir_lst[3], 'acc_loss_results.txt'), 'r') as fl_3:
        #open(os.path.join(log_dir_lst[4], 'acc_loss_results.txt'), 'r') as fl_4:
        # open(os.path.join(log_dir_lst[5], 'acc_loss_results.txt'), 'r') as fl_5:
        lst_0 = eval(fl_0.readlines()[0])
        lst_1 = eval(fl_1.readlines()[0])
        lst_2 = eval(fl_2.readlines()[0])
        lst_3 = eval(fl_3.readlines()[0])
        #lst_4 = eval(fl_4.readlines()[0])
        # lst_5 = eval(fl_5.readlines()[0])
        lst.append(lst_0)
        lst.append(lst_1)
        lst.append(lst_2)
        lst.append(lst_3)
        #lst.append(lst_4)
        # lst.append(lst_5)
        assert len(lst_0)==len(lst_1)==len(lst_2)==len(lst_3)#==len(lst_4)#==len(lst_5)

        #plt.subplots(2, 3, figsize=(12, 5))
        plot_lines(lst)

def pnorm_avg():
    #power average
    x = [i for i in range(100)]
    y_ = AverageMeter_pnorm('Avg', ':.4e')
    y = np.linspace(3, 0, 100)
    y_1, y_2, y_3, y_4, y_5, y_6, y_7 = [], [], [], [], [], [], []
    total_lst = [y_1, y_2, y_3, y_4, y_5, y_6, y_7]
    p_lst = []#['p=1','p=2','p=3','p=4','p=5','p=6','p=7']
    for (index, y_i) in enumerate(total_lst):
        for i in y:
            y_.update(i,1./(index+1))
            y_i.append(y_.avg)
        p_lst.append('p={}'.format(1./(index+1)))
        y_.reset()

    colors = ['#1f77b4',
              '#ff7f0e',
              '#2ca02c',
              '#d62728',
              '#9467bd',
              '#8c564b',
              '#e377c2',
              '#7f7f7f',
              '#bcbd22',
              '#17becf',
              '#1a55FF']
    plt.figure(figsize=(5, 3))
    for (index, y_i) in enumerate(total_lst):
        plt.plot(x, y_i, color=colors[index],
             label='{}'.format(p_lst[index]))
    plt.legend()
    plt.grid(True)
    plt.show()



if __name__ == "__main__":
    base_pth = '/users6/ttwu/script/Robustness/SLN-master/CIFAR-10/'
    # log_dir = ['cifar10_dependent0.4_seed0_GN_on_parameters/sigma=0.005/',
    #            'cifar10_dependent0.4_seed0_GN_on_label/sigma=0.5/',
    #            'cifar10_dependent0.4_seed0_GN_on_moutput/sigma=0.005/',
    #            'cifar10_dependent0.4_seed0_no_GN/sigma=0.005/']
    log_dir = ['dependent0.4_seed0_GN_adaptive_label_iterate/sigma=0.5/pnorm=2.0/lr_sig=0.01/delay_eps=0.0/',
               'dependent0.4_seed0_GN_adaptive_label_iterate/sigma=0.5/pnorm=2.0/lr_sig=0.005/delay_eps=0.0/',
               'dependent0.4_seed0_GN_adaptive_label_iterate/sigma=0.5/pnorm=2.0/lr_sig=0.001/delay_eps=0.0/',
               'cifar10_dependent0.4_seed0_GN_on_label/sigma=0.5/'
               ]
    # 'cifar10_dependent0.4_seed0_no_GN/sigma=0.005/',
    # 'dependent0.4_seed0_GN_adaptive_new/sigma=0.01/norm=1.0',
    # 'dependent0.4_seed0_GN_adaptive_new/sigma=0.001/norm=1.0',
    # 'dependent0.4_seed0_GN_adaptive_new/sigma=0.005/norm=1.0',
    # 'dependent0.4_seed0_GN_adaptive_new/sigma=0.005/norm=0.5',
    log_dir_lst = [os.path.join(base_pth, dir) for dir in log_dir]
    #three_types_acc(log_dir_lst)
    pnorm_avg()