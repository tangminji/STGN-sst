# 这个脚本要单独运行，不然路径会出错

from datasets import load_dataset
from torchtext.vocab import GloVe
import torch
from torch import nn
import json
import os
import numpy as np
import random

all_type = ['train','validation','test']

def t5label(x):
    if x<=0.2:
        return 0
    elif x<=0.4:
        return 1
    elif x<=0.6:
        return 2
    elif x<=0.8:
        return 3
    else:
        return 4

def getsstdata():
    data_all = load_dataset('sst','default')
    label = {}
    for kind in all_type:
        label[kind] = list(map(t5label,data_all[kind]['label']))
    words = {}
    for kind in all_type:
        words[kind] = list(map(lambda x:x.lower().split('|'),data_all[kind]['tokens']))
    # 还可以加优化: 去掉标点，去掉过低频词
    del data_all

    # 改成2分类
    print('To 2 kinds...')
    for kind in all_type:
        new_label , new_words = [],[]
        for l, w in zip(label[kind],words[kind]):
            if l==2:
                continue
            else:
                new_label.append(0 if l<2 else 1)
                new_words.append(w)
            label[kind], words[kind] = new_label, new_words

    return label, words

def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
def processe_sst_data():

    if os.path.exists('./sst/embedding.pt'):
        print('SST data was processed!')
        return
    if not os.path.exists('./sst'):
        os.mkdir('./sst')
    print('SST data processing ...')

    label, words = getsstdata()

    print('Consturct the Vocab dictionary...')
    cache_dir = './glove'
    glove = GloVe(name='840B',dim=300,cache=cache_dir)
    sentence = sum(words.values(),[])
    vocab = set(sorted(sum(sentence,[])))
    vocab2id = {'<pad>':0}
    id2vocab = ['<pad>']
    for i,v in enumerate(vocab,1):
        vocab2id[v] = i
        id2vocab.append(v)
    vocab2id['<unk>'] = len(vocab2id)
    id2vocab.append('<unk>')

    print('Initialize the embedding...')
    set_seed(0)
    embed = torch.FloatTensor(len(id2vocab),300)
    nn.init.normal_(embed, 0, 0.1)
    nn.init.constant_(embed[0],0)
    OOV = 0
    for i,v in enumerate(vocab,1):
        if v in glove.stoi:
            embed[i] = glove[v]
        else:
            OOV += 1
    torch.save(embed, 'sst/embedding.pt')
    with open('sst/vocab_map.json','w') as f:
        json.dump(vocab2id, f)
    
    print('Save raw data...')
    # 保存原数据
    for type in all_type:
        with open(f'sst/{type}_raw.txt','w',encoding='utf-8') as f:
            lines = []
            for w, l in zip(words[type],label[type]):
                lines.append(f'{l}\t{"|".join(w)}')
            f.writelines('\n'.join(lines))
    
    print('Save processed data...')
    # 保存ids
    for type in all_type:
        with open(f'sst/{type}_idx.txt','w',encoding='utf-8') as f:
            lines = []
            for w, l in zip(words[type],label[type]):
                ids = '|'.join(list(map(lambda x:str(vocab2id[x]),w)))
                lines.append(f'{l}\t{ids}')
            f.writelines('\n'.join(lines))
    
    info = {'embedding_dim': 300, 'vocab_size': len(embed),'oov': OOV}
    with open('sst/info.json','w') as f:
        json.dump(info, f)

    print('Done')

if __name__ == '__main__':
    processe_sst_data()