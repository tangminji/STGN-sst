from genericpath import exists
import os
from numpy import random
from torch._C import device
from torch.utils.data import DataLoader,Dataset
from cmd_args_sst import SST_CONFIG
from models.sst_lstm import SSTLSTM
import torch.nn as nn
import torch
from torch.nn.utils.rnn import pad_sequence
import numpy as np

class SSTDataset(Dataset):
    def __init__(self, words, labels, targets_gt):
        super().__init__()
        self.words = words
        self.labels = labels
        self.targets_gt = targets_gt
    def __getitem__(self, index):
        return self.words[index], self.labels[index], self.targets_gt[index], index
    def __len__(self):
        return len(self.words)

# words, length, labels, gt, index
def fn(data):
    words, labels, gt, index = zip(*data)
    return pad_sequence(words), list(map(len,words)), torch.tensor(labels), torch.tensor(gt), torch.tensor(index)

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


def generate_noisy_labels(args):
    print(f'==> Generate noisy labels with noise_rate={args.noise_rate}...')
    train_words, train_labels = load_sst_data(os.path.join(args.data_path,'train_idx.txt'))

    noise_rate = args.noise_rate
    total_train = len(train_labels)

    noisy_ind = np.random.choice(total_train, int(total_train*noise_rate), False).tolist()
    noisy_ind.sort()
    train_noisy = []
    for i in range(total_train):
        if i in noisy_ind:
            noisy = random.randint(0,args.num_class-2)
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

    val_words, val_labels = load_sst_data(os.path.join(args.data_path,'validation_idx.txt'))
    test_words, test_labels = load_sst_data(os.path.join(args.data_path,'test_idx.txt'))
    trainset = SSTDataset(train_words, train_noisy, train_labels)
    valset = SSTDataset(val_words, val_labels, val_labels)
    testset = SSTDataset(test_words, test_labels, test_labels)

    total_train = len(train_labels)
    noisy_ind, clean_ind = [],[]
    for i in range(total_train):
        if train_noisy[i]!=train_labels[i]:
            noisy_ind.append(i)
        else:
            clean_ind.append(i)
    
    batch_size = args.batch_size
    trainloader = DataLoader(trainset,batch_size=batch_size,shuffle=True,collate_fn=fn)
    valloader = DataLoader(valset, batch_size=batch_size,collate_fn=fn)
    testloader = DataLoader(testset, batch_size=batch_size, collate_fn=fn)
    return trainloader, valloader, testloader, noisy_ind, clean_ind

def get_SST_model_and_loss_criterion(args):
    """Initializes DNN model and loss function.

    Args:
        args (argparse.Namespace):

    Returns:
        model (torch.nn.Module): DNN model.
        criterion (torch.nn.modules.loss): cross entropy loss
    """
    print('Building LSTM')
    args.arch = 'LSTM-FC'
    embed_weight = torch.load(os.path.join(args.data_path,'embedding.pt'))
    vocab_size, embed_dim = embed_weight.shape
    args.vocab_size = vocab_size
    args.embed_dim = embed_dim
    model = SSTLSTM(vocab_size, embed_dim, hidden_size=args.hidden_size, n_class=args.num_class,dropout=args.dropout,pretrained=embed_weight)
    model.to(args.device)
    criterion = SST_CONFIG[args.loss].to(args.device) #CE/GCE

    if args.loss == 'GCE':
        criterion.q = args.q 
        criterion.num_classes = args.num_class
    criterion_val = nn.CrossEntropyLoss(reduction='none').to(args.device)
    return model, criterion, criterion_val

if __name__ == '__main__':
    from cmd_args_sst import args
    get_SST_model_and_loss_criterion(args)