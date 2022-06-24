import torch

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

if __name__ == '__main__':
    words,noisys,labels = load_sst_noisy_data('sst/noisy/0.6.txt')
    err = 0
    for n,l in zip(noisys, labels):
        if n!=l:
            err+=1
    print(err*1.0/len(labels))