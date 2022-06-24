from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

class SSTLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, n_class=5, dropout=0.5, pretrained=None):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        if not pretrained is None:
            self.embedding = self.embedding.from_pretrained(pretrained, freeze=False)
        self.lstm = nn.LSTM(embed_dim, hidden_size, num_layers=1)
        self.drop = nn.Dropout(dropout)
        self.h_size = 1 * hidden_size
        self.fc = nn.Linear(self.h_size, n_class)
        
    def forward(self, x, length):
        emb_x = self.embedding(x)
        pack_x = pack_padded_sequence(emb_x, length, enforce_sorted=False)
        self.lstm.flatten_parameters()
        output, (h,c) = self.lstm(pack_x)
        h = self.drop(h)
        logits = self.fc(h.permute(1,0,2).view(-1,self.h_size))
        return logits
    
    def sentence_encode(self, x, length):
        emb_x = self.embedding(x)
        pack_x = pack_padded_sequence(emb_x, length, enforce_sorted=False)
        self.lstm.flatten_parameters()
        output, (h,c) = self.lstm(pack_x)
        return h.permute(1,0,2).view(-1, self.h_size)