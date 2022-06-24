from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

class SSTLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, n_class=5, dropout=0.5, pretrained=None):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        if not pretrained is None:
            self.embedding = self.embedding.from_pretrained(pretrained, freeze=False)
        # 这里可以用frompretrained加载预训练词表
        self.lstm = nn.LSTM(embed_dim, hidden_size, num_layers=1)
        self.drop = nn.Dropout(dropout)
        # 可以加其他参数 bidrectional, num_layers, dropout
        self.h_size = 1 * hidden_size
        # 这里暂时用一个全连接层
        self.fc = nn.Linear(self.h_size, n_class)
        
    def forward(self, x, length):
        emb_x = self.embedding(x)
        # embed 要不要接dropout
        pack_x = pack_padded_sequence(emb_x, length, enforce_sorted=False)
        self.lstm.flatten_parameters()
        output, (h,c) = self.lstm(pack_x)
        # h (num_layer, batch_size, hidden_size)
        h = self.drop(h)
        logits = self.fc(h.permute(1,0,2).view(-1,self.h_size))
        return logits
    
    def sentence_encode(self, x, length):
        emb_x = self.embedding(x)
        # embed 要不要接dropout
        pack_x = pack_padded_sequence(emb_x, length, enforce_sorted=False)
        self.lstm.flatten_parameters()
        output, (h,c) = self.lstm(pack_x)
        return h.permute(1,0,2).view(-1, self.h_size)