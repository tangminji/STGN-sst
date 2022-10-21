import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from data.sst_dataset import tokenizer
import json
import datasets
import transformers
transformers.logging.set_verbosity_error()

class Glueset(Dataset):
    def __init__(self, dataset, noisy_label=None):
        self.label = torch.tensor(dataset["label"])
        self.noisy_label = torch.tensor(noisy_label) if noisy_label is not None else self.label
        self.data = {k:torch.tensor(dataset[k]) for k in ["input_ids","token_type_ids","attention_mask"]}
    def __len__(self):
        return len(self.label)
    def __getitem__(self, index):
        X = {k: v[index] for k,v in self.data.items()}
        y = self.label[index]
        noisy = self.noisy_label[index]
        return X, noisy, y, index

def generate_noisy_labels(args, dataset):
    print(f'==> Generate noisy labels with noise_rate={args.noise_rate}...')
    
    noise_rate = args.noise_rate
    total_train = len(dataset)
    train_labels = dataset["label"]
    rng = np.random.RandomState(42)
    noisy_ind = rng.choice(total_train, int(total_train*noise_rate), False).tolist()
    
    noisy_ind.sort()
    train_noisy = []
    for i in range(total_train):
        if i in noisy_ind:
            # numpy.random.randint(low,hight) [low, high)
            noisy = rng.randint(0,args.num_class-1) #0~x-1,x+1~k-1
            if noisy >= train_labels[i]:
                noisy += 1
            train_noisy.append(noisy)
        else:
            train_noisy.append(train_labels[i])
    folder = os.path.join(args.data_path,'noisy')
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)
    noisy_file = os.path.join(folder,f'{args.noise_rate}.json')
    with open(noisy_file, "w", encoding="utf-8") as f:
        json.dump(train_noisy, f)
    print(f'==> Noisy Label generated at {noisy_file}')

def get_glue_train_and_val_loader(args):
    processed_path = os.path.join(args.data_path, "processed")
    folder = processed_path
    if args.dataset == "MNLI":
        def mnli_process(example):
            return tokenizer(example["premise"], example["hypothesis"], padding='max_length', truncation=True, max_length=128)
        if os.path.exists(processed_path):
            mnli = datasets.load_from_disk(processed_path)
        else:
            mnli = datasets.load_dataset("glue","mnli")
            del mnli["test_matched"], mnli["test_mismatched"]
            mnli = mnli.map(mnli_process)
            mnli = mnli.remove_columns(["premise","hypothesis"])
            if not os.path.exists(folder):
                os.makedirs(folder, exist_ok=True)
            mnli.save_to_disk(processed_path)
        
        trainset = mnli["train"]
        valset = mnli["validation_matched"]
        testset = mnli["validation_mismatched"]
    
    elif args.dataset == "QQP":
        # num_class = 2
        def qqp_process(example):
            return tokenizer(example["question1"],example["question2"], padding='max_length', truncation=True, max_length=128)
        if os.path.exists(processed_path):
            qqp = datasets.load_from_disk(processed_path)
        else:
            qqp = datasets.load_dataset("glue","qqp")
            del qqp["test"]
            qqp = qqp.map(qqp_process)
            qqp = qqp.remove_columns(["question1","question2"])
            if not os.path.exists(folder):
                os.makedirs(folder, exist_ok=True)
            qqp.save_to_disk(processed_path)
        
        trainset = qqp["train"]
        valset = qqp["validation"]
        testset = None
    noisy_file = os.path.join(args.data_path,'noisy',f'{args.noise_rate}.json')
    train_labels = trainset["label"]
    if args.noise_rate == 0:
        train_noisy = train_labels
    else:
        if not os.path.exists(noisy_file):
            generate_noisy_labels(args, trainset)
        with open(noisy_file, "r") as f:
            train_noisy = json.load(f)
        
    total_train = len(train_labels)
    noisy_ind, clean_ind = [],[]
    for i in range(total_train):
        if train_noisy[i]!=train_labels[i]:
            noisy_ind.append(i)
        else:
            clean_ind.append(i)

    trainset = Glueset(trainset, noisy_label=train_noisy)
    valset = Glueset(valset)
    testset = Glueset(testset) if testset is not None else None
    batch_size = args.batch_size
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    valloader = DataLoader(valset, batch_size=batch_size)
    testloader = DataLoader(testset, batch_size=batch_size) if testset is not None else None
    return trainloader, valloader, testloader, noisy_ind, clean_ind