import argparse
from common.losses import *
import os
# Settings
parser = argparse.ArgumentParser(description='PyTorch SST')

parser.add_argument('--exp_name', default='SST', type=str)
parser.add_argument('--sub_script', default='sbatch_sst_sub.sh', type=str)
parser.add_argument('--out_tmp', default='sst_out_tmp.json', type=str)
parser.add_argument('--params_path', default='sst_params.json', type=str)
parser.add_argument('--log_dir', default='/users6/ttwu/script/Robustness/SLN-master/cifar10/', type=str)

parser.add_argument('--dataset', default='SST', type=str, help="Model type selected in the list: [MNIST, CIFAR10, CIFAR-100, CIFAR-10_5K, UTKFACE, SST]")
parser.add_argument('--loss', default='CE', type=str, help="loss type")
#parser.add_argument('--num_class', default=2, type=int)
# parser.add_argument('--batch_size', type=int, default=128, help='input batch size for training')
# parser.add_argument('--epochs', type=int, default=1, help='number of epochs to train')#300
#parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--lr_sig', type=float, default=0.005, help='learning rate for sigma iteration')
parser.add_argument('--noise_mode', type=str, default='sym', help='Noise mode in the list: [sym, asym, dependent]') #这里加的是对称噪声
parser.add_argument('--noise_rate', type=float, default=0.4, help='Noise rate')
#parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
parser.add_argument('--forget_times', type=int, default=1, help='thereshold to differentiate clean/noisy samples')
parser.add_argument('--num_gradual', type=int, default=0, help='epochs for warmup')
parser.add_argument('--ratio_l', type=float, default=0.5, help='element1 to total ratio')
parser.add_argument('--total', type=float, default=1.0, help='total amount of every elements')
parser.add_argument('--patience', type=int, default=3, help='patience for increasing sig_max for avoiding overfitting')
parser.add_argument('--times', type=float, default=3.0, help='increase perturb by times')
parser.add_argument('--avg_steps', type=int, default=10, help='step nums at most to calculate k1')
parser.add_argument('--adjustimes', type=int, default=10, help='Maximum number of adjustments')
parser.add_argument('--sigma', type=float, default=0.05, help='STD of Gaussian noise')#label0.5/para5e-3/moutput5e-3
parser.add_argument('--sig_max', type=float, default=0.1, help='max threshold of sigma')
parser.add_argument('--smoothing', type=float, default=0.1, help='used in mode Label_smoothing')
parser.add_argument('--delay_eps', type=float, default=50.0, help='p-norm of adaptive regularization')
parser.add_argument('--early_eps', type=float, default=200.0, help='p-norm of adaptive regularization')
parser.add_argument('--pnorm', type=float, default=2.0, help='p-norm of adaptive regularization')
parser.add_argument('--beta', type=float, default=0.9, help='beta for exponential moving average of the gradient')
parser.add_argument('--skip_clamp_param', default=False, const=True, action='store_const',
                    help='Do not clamp data parameters during optimization')
parser.add_argument('--gpu_id', type=int, default=0, help='index of gpu to use')
parser.add_argument('--seed', type=int, default=0, help='random seed (default: 0)')
parser.add_argument('-p', '--print-freq', default=10, type=int, metavar='N', help='print frequency')
parser.add_argument('--restart', default=True, const=True, action='store_const',
                    help='Erase log and saved checkpoints and restart training')#False

parser.add_argument('--mode', type=str, default='no_GN',
                    choices=['GN_on_label',
                             'GN_on_moutput',
                             'GN_on_parameters',
                             'no_GN',
                             'GN_noisy_samples',
                             'GN_gods_perpective2',
                             'GN_gods_perpective3',
                             'Random_walk'])# 这里默认改成no_GN, for Base & GCE

# 5分类 batch_size=32 epoch=128
# set exp_name with ['base','GCE','SLN','STGN'] to change the method

parser.add_argument('--batch_size', type=int, default=32, help='input batch size for training') #32 STGN batch_size小一点较好
# parser.add_argument('--epochs', type=int, default=30, help='number of epochs to train') #30 epoch
parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train')
parser.add_argument('--data_path', type=str, default='data/sst', help='the data and embedding path')
parser.add_argument('--hidden_size',type=int, default=168, help='hidden_size of LSTM') #更正，原文中是168
parser.add_argument('--dropout',type=float, default=0.5, help='dropout=0,0.5') #0/0.5 (未使用)
parser.add_argument('--lr', type=float, default=1e-5, help='learning rate') #这里默认1e-5
parser.add_argument('--decay', type=float, default=0.9, help='decay rate of RMSProp')
parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay')
parser.add_argument('--num_class', default=5, type=int, choices=[2,5], 
    help='2 for SST-binary, 5 for SST-fine') # 目前研究5分类，5分类加噪声效果显著一些
parser.add_argument('--q', default=0.7, type=float, help='q for GCE')

parser.add_argument('--model', type=str, default='bert-base-uncased', choices=['bert-base-uncased','bert-large-uncased'])
parser.add_argument('--cache_dir', type=str, default='/users10/mjtang/dataset/huggingface')

args = parser.parse_args()

args.data_path = {
    2: 'data/sst-2',
    5: 'data/sst'
}[args.num_class]

# 本地缓存模型路径
args.model_path = os.path.join(args.cache_dir,args.model)

# 根据exp_name修正参数， 这样可以减少要传入参数的数量
if 'STGN' in args.exp_name:
    args.mode = 'Random_walk'
if 'GCE' in args.exp_name:
    args.loss = 'GCE'
if 'SLN' in args.exp_name:
    args.mode = 'GN_on_label'
if 'GNMP' in args.exp_name:
    args.mode = 'GN_on_parameters'
if 'GNMO' in args.exp_name:
    args.mode = 'GN_on_moutput'

SST_CONFIG = {
    "CE": nn.CrossEntropyLoss(reduction='none'),
    "GCE": GCELoss(num_classes=args.num_class, reduction='none')
}

CIFAR10_CONFIG = {
    "CE": CELoss(reduction='none'),
    "FL": FocalLoss(gamma=0.5),
    "MAE": MAELoss(num_classes=10, reduction='none'),
    "GCE": GCELoss(num_classes=10, q=0.01, reduction='none'),
    "SCE": SCELoss(num_classes=10, a=0.1, b=1),
    # "NLNL": NLNL(train_loader, num_classes=10),
    "NFL": NormalizedFocalLoss(gamma=0.5, num_classes=10),
    "NGCE": NGCELoss(num_classes=10),
    "NCE": NCELoss(num_classes=10),
    "NFL+RCE": NFLandRCE(alpha=1, beta=1, num_classes=10, gamma=0.5),
    "NCEandMAE": NCEandMAE(alpha=1, beta=1, num_classes=10),
    "NCEandRCE": NCEandRCE(alpha=1, beta=1, num_classes=10),
}

CIFAR100_CONFIG = {
    "CE": nn.CrossEntropyLoss(),
    "FL": FocalLoss(gamma=0.5),
    "MAE": MAELoss(num_classes=100),
    "GCE": GCELoss(num_classes=100, q=0.001),
    "SCE": SCELoss(num_classes=100, a=6, b=0.1),
    # "NLNL": NLNL(train_loader, num_classes=10),
    "NFL": NormalizedFocalLoss(gamma=0.5, num_classes=100),
    "NGCE": NGCELoss(num_classes=100),
    "NCE": NCELoss(num_classes=100),
    "NFL+RCE": NFLandRCE(alpha=10, beta=1, num_classes=100, gamma=0.5),
    "NCEandMAE": NCEandMAE(alpha=10, beta=1, num_classes=100),
    "NCEandRCE": NCEandRCE(alpha=10, beta=1, num_classes=100),
}

#CIFAR10_PATH = '/home/zhh/code/data/'#path for cifar10, cifar100 etc.
CIFAR10_PATH = '/users6/ttwu/script/GoalstepRE/ml-data-parameters-master/dataset/'
CIFAR10_LB_PATH = '/users6/ttwu/script/Robustness/SLN-master/data/CIFAR10/'
CIFAR100_PATH = '/users6/ttwu/script/GoalstepRE/ml-data-parameters-master/dataset/'
CIFAR100_LB_PATH = '/users6/ttwu/script/Robustness/SLN-master/data/CIFAR100/'

#learning_rate_schedule = np.array([80, 100, 160])#for CIFAR10/CIFAR100

