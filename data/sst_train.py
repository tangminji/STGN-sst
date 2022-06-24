from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import torch
from torch import nn
import sys
sys.path.insert(0,"..")
from models.sst_lstm import SSTLSTM
import json
import atexit

class SSTDataset(Dataset):
    def __init__(self, words, labels):
        super().__init__()
        self.words = words
        self.labels = labels
    def __getitem__(self, index):
        return self.words[index], self.labels[index]
    def __len__(self):
        return len(self.labels)

# words, length, labels
def fn(data):
    words, labels = zip(*data)
    return pad_sequence(words), list(map(len,words)), torch.tensor(labels)
# words, length, labels, ground_truth

# 先不加噪声
# 可以读proccess和raw data
def get_sst_data(path, raw=False):
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

train_words, train_labels = get_sst_data('./sst/train_idx.txt')
val_words, val_labels = get_sst_data('./sst/validation_idx.txt')
test_words, test_labels = get_sst_data('./sst/test_idx.txt')

trainset = SSTDataset(train_words, train_labels)
valset = SSTDataset(val_words, val_labels)
testset = SSTDataset(test_words, test_labels)

batch_size = 4

trainloader = DataLoader(trainset,batch_size=batch_size,shuffle=True,collate_fn=fn)
valloader = DataLoader(valset, batch_size=batch_size,collate_fn=fn)
testloader = DataLoader(testset, batch_size=batch_size, collate_fn=fn)

embed_weight = torch.load('./sst/embedding.pt')
vocab_size = len(embed_weight)

net = SSTLSTM(vocab_size=vocab_size, embed_dim=300, hidden_size=168, dropout=0.5, pretrained=embed_weight)

device = torch.device('cuda')
net.to(device)
parameters = list(filter(lambda x:x.requires_grad, net.parameters()))
optimizer = torch.optim.RMSprop(parameters, lr=0.001, alpha=0.9, weight_decay=0.0001)
criterion = nn.CrossEntropyLoss()

def val(net, loader, criterion):
    net.eval()
    acc_sum, loss_sum, n = 0.0,0.0,0
    with torch.no_grad():
        for words, length, labels in loader:
            words, length, labels = words.to(device), length, labels.to(device)
            logits = net(words,length)
            loss = criterion(logits, labels)
            ni = len(labels)
            acc_sum += (logits.argmax(1)==labels).detach().cpu().float().sum().item()
            loss_sum += loss.item()*ni
            n += ni
    return acc_sum/n, loss_sum/n

print('before train: val, test')
print(val(net,valloader,criterion), val(net,testloader,criterion))

def train(net, loader, criterion,optimizer):
    net.train()
    acc_sum, loss_sum, n = 0.0,0.0,0
    for words, length, labels in loader:
        words, length, labels = words.to(device), length, labels.to(device)
        logits = net(words,length)
        loss = criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ni = len(labels)
        acc_sum += (logits.argmax(1)==labels).detach().cpu().float().sum().item()
        loss_sum += loss.item()*ni
        n += ni
    return acc_sum/n, loss_sum/n

best_val = 0
best_acc, best_loss = 0,0
early_stop = 0
for epoch in range(1000):
    train_acc,train_loss = train(net,trainloader,criterion,optimizer)
    val_acc,val_loss = val(net,valloader,criterion)
    print(f'Epoch {epoch}: Train acc:{train_acc} loss:{train_loss} Val acc:{val_acc} loss:{val_loss}')
    if val_acc > best_val:
        early_stop = 0
        best_val = val_acc
        test_acc,test_loss = val(net, testloader, criterion)
        best_acc,best_loss = test_acc, test_loss
        print(f'best val {best_val}, Test acc: {test_acc}, Test_loss: {test_loss}')
    else:
        early_stop += 1
    if early_stop == 10:
        break

@atexit.register()
def log_result():
    print(f'best val {best_val}, Test acc: {best_acc}, Test_loss: {best_loss}')
    with open('result.txt','w') as f:
        f.write(f'best val {best_val}, Test acc: {best_acc}, Test_loss: {best_loss}')