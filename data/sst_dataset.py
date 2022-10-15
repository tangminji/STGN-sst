import os
from numpy import random
from torch.utils.data import DataLoader,Dataset
from cmd_args_sst import args, SST_CONFIG
import torch.nn as nn
import torch
from torch.nn.utils.rnn import pad_sequence
import numpy as np


import pytreebank
import torch
from transformers import BertTokenizer,BertConfig,BertForSequenceClassification
from torch.utils.data import Dataset

print("Loading the tokenizer")
tokenizer = BertTokenizer.from_pretrained(args.model_path) #"bert-large-uncased"

print("Loading SST")
sst = pytreebank.load_sst()

SEP = [tokenizer.sep_token_id] #[SEP]

# 其实完全可以用padding='max_length'实现
def rpad(array, n=70):
    """Right padding."""
    current_len = len(array)
    if current_len > n:
        return array[: n - 1] + SEP
    extra = n - current_len
    return array + ([0] * extra)


def get_binary_label(label):
    """Convert fine-grained label to binary label."""
    if label < 2:
        return 0
    if label > 2:
        return 1
    raise ValueError("Invalid label")


class SSTDataset(Dataset):
    """Configurable SST Dataset.
    
    Things we can configure:
        - split (train / val / test)
        - root / all nodes
        - binary / fine-grained
    """

    def __init__(self, split="train", binary=False, noisy_label=None):
        """Initializes the dataset with given configuration.

        Args:
            split: str
                Dataset split, one of [train, val, test]
            root: bool
                If true, only use root nodes. Else, use all nodes.
            binary: bool
                If true, use binary labels. Else, use fine-grained.
        """
        root = True #只用树根标签
        print(f"Loading SST {split} set")
        self.sst = sst[split]

        print("Tokenizing")
        self.data = []
        self.label = []

        if root and binary:
            for tree in self.sst:
                if tree.label != 2:
                    self.data.append(tree.to_lines()[0])
                    self.label.append(get_binary_label(tree.label))            
        elif root and not binary:
            for tree in self.sst:
                self.data.append(tree.to_lines()[0])
                self.label.append(tree.label)
        elif not root and not binary:
            for tree in self.sst:
                for label, line in tree.to_labeled_lines():
                    self.data.append(line)
                    self.label.append(label)
        else:
            for tree in self.sst:
                for label, line in tree.to_labeled_lines():
                    if label != 2:
                        self.data.append(line)
                        self.label.append(get_binary_label(label))
        self.data = tokenizer(self.data, padding='max_length', truncation=True, max_length=66, return_tensors='pt')
        self.noisy_label = noisy_label if noisy_label is not None else self.label
        assert len(self.label) == len(self.noisy_label), f" the noisy label of set {split} unmatched, {len(self.label)} datas {len(self.noisy_label)} noisy labels"

    def __len__(self):
        return len(self.label)

    # words, labels, target_gt, index
    def __getitem__(self, index):
        X = {k: v[index] for k,v in self.data.items()}
        y = self.label[index]
        noisy = self.noisy_label[index]
        return X, noisy, y, index

# # words, length, labels, gt, index
# def fn(data):
#     words, labels, gt, index = zip(*data)
#     return pad_sequence(words), list(map(len,words)), torch.tensor(labels), torch.tensor(gt), torch.tensor(index)

def load_sst_data(path, raw=False):
    words, labels = [],[]
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            label, sentence = line.strip().split('\t')
            word = sentence.split('|')
            if not raw:
                word = torch.LongTensor(list(map(int,word)))
                label = int(label)
            labels.append(label)
            words.append(word)
    return words, labels

def load_sst_noisy_data(path, raw=False):
    print(f'==> Load noisy_data from {path}')
    words, noisys, labels = [],[],[]
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            noisy, label, sentence = line.strip().split('\t')
            word = sentence.split('|')
            if not raw:
                word = torch.LongTensor(list(map(int,word)))
                label = int(label)
                noisy = int(noisy)
            labels.append(label)
            words.append(word)
            noisys.append(noisy)
    return words, noisys, labels

# 生成噪声样本
def generate_noisy_labels(args):
    print(f'==> Generate noisy labels with noise_rate={args.noise_rate}...')
    train_words, train_labels = load_sst_data(os.path.join(args.data_path,'train_idx.txt'))
    # 添加噪声部分,以后要重复训练的话可以删
    noise_rate = args.noise_rate
    total_train = len(train_labels)
    # 噪声样本, 这里是二分类，如果是五分类就复杂些
    noisy_ind = np.random.choice(total_train, int(total_train*noise_rate), False).tolist()
    noisy_ind.sort()
    train_noisy = []
    for i in range(total_train):
        if i in noisy_ind: #这里要考虑5分类问题
            # random.randint(lower,upper) [lower,upper] 都可能会取到
            noisy = random.randint(0,args.num_class-2) #噪声类别 0~x-1,x+1~k-1
            if noisy >= train_labels[i]:
                noisy += 1
            train_noisy.append(noisy) 
        else:
            train_noisy.append(train_labels[i])
    folder = os.path.join(args.data_path,'noisy')
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)
    noisy_file = os.path.join(folder,f'{args.noise_rate}.txt')
    with open(noisy_file,'w',encoding='utf-8') as f:
        lines = []
        for w, n, l in zip(train_words,train_noisy,train_labels):
            ids = '|'.join(list(map(lambda x:str(x),w.numpy().tolist())))
            lines.append(f'{n}\t{l}\t{ids}')
        f.writelines('\n'.join(lines))
    print(f'==> Noisy Label generated at {noisy_file}')

# 这里采用二分类
# noise_rate
# 这里的label是真实标签，noisy是噪声标签
# Dataset (words, noisy ,truth)
def get_sst_train_and_val_loader(args):
    print('==> Preparing data for sst..')

    if args.noise_rate == 0:
        train_words, train_labels = load_sst_data(os.path.join(args.data_path,'train_idx.txt'))
        train_noisy = train_labels
    else:
        noisy_file = os.path.join(args.data_path,'noisy',f'{args.noise_rate}.txt')
        if not os.path.exists(noisy_file):
            generate_noisy_labels(args)
        train_words, train_noisy, train_labels = load_sst_noisy_data(noisy_file)

    binary = args.num_class==2
    trainset = SSTDataset("train",binary=binary,noisy_label=train_noisy)
    valset = SSTDataset("dev",binary=binary)
    testset = SSTDataset("test",binary=binary)

    total_train = len(train_labels)
    noisy_ind, clean_ind = [],[]
    for i in range(total_train):
        if train_noisy[i]!=train_labels[i]:
            noisy_ind.append(i)
        else:
            clean_ind.append(i)
    
    batch_size = args.batch_size
    trainloader = DataLoader(trainset,batch_size=batch_size,shuffle=True)
    valloader = DataLoader(valset, batch_size=batch_size)
    testloader = DataLoader(testset, batch_size=batch_size)
    return trainloader, valloader, testloader, noisy_ind, clean_ind

def get_SST_model_and_loss_criterion(args):
    """Initializes DNN model and loss function.

    Args:
        args (argparse.Namespace):

    Returns:
        model (torch.nn.Module): DNN model.
        criterion (torch.nn.modules.loss): cross entropy loss
    """

    print('Loading Bert')
    args.arch = 'Bert'
    config = BertConfig.from_pretrained(args.model_path)
    config.num_labels = args.num_class
    model = BertForSequenceClassification.from_pretrained(args.model_path, config=config)
    model.to(args.device)
    criterion = SST_CONFIG[args.loss].to(args.device) #CE/GCE
    # 手动设置，因为加载json文件前就已经创建了Loss了
    if args.loss == 'GCE':
        criterion.q = args.q 
        criterion.num_classes = args.num_class
        print(f"Use GCE loss, q={criterion.q}, {type(criterion)}")
    criterion_val = nn.CrossEntropyLoss(reduction='none').to(args.device)
    return model, criterion, criterion_val

if __name__ == '__main__':
    from cmd_args_sst import args
    get_SST_model_and_loss_criterion(args)