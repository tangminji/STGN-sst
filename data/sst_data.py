from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence

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
    return pad_sequence(words), list(map(len,words)), labels
# words, length, labels, ground_truth

# 先不加噪声
# 可以读proccess和raw data
def get_sst_data(path):
    words, labels = [],[]
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            label, sentence = line.strip().split('\t')
            word = sentence.split('|')
            labels.append(label)
            words.append(word)
    return words, labels

train_words, train_labels = get_sst_data('./sst/train_idx.txt')
val_words, val_labels = get_sst_data('./sst/validation_idx.txt')
