import torch
import torch.nn as nn


class BERTEmbedding:
    def __init__(self, vocab_size, n_segments, max_len, embed_dim, dropout):
        super().__init__()

        self.token = nn.Embedding(vocab_size, embed_dim)
        self.segment = nn.Embedding(n_segments, embed_dim)
        self.position = nn.Embedding(max_len, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.pos_input = torch.tensor([i for i in range(max_len)], )
        
    def forward(self, seq, seg):
        embed_val = self.token(seq) + self.segment(seg) + self.position(self.pos_input)
        return embed_val
    
class BERT(nn.Module):
    def __init__(self, vocab_size, n_segments, max_len,embed_dim, n_layers, attn_heads, dropout):
        super().__init__()
        self.embedding = BERTEmbedding(vocab_size, n_segments, max_len,embed_dim, dropout)
        
        self.encoder_layer = nn.TransformerEncoderLayer(embed_dim, attn_heads)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, n_layers)
        
    def forward(self, seq, seg):
        out = self.embedding.forward(seq, seg)
        out = self.encoder(out)
        return out

if __name__ == "__main__":
    VOCAB_SIZE = 30000
    N_SEGMENTS = 3
    MAX_LEN = 512
    EMBED_DIM = 768
    N_LAYERS = 12
    ATTN_HEADS = 12
    DROPOUT = 0.1
    
    sample_seq = torch.randint(high = VOCAB_SIZE, size = [MAX_LEN, ])
    sample_seg = torch.randint(high = N_SEGMENTS, size = [MAX_LEN, ])
    
    embedding = BERTEmbedding(VOCAB_SIZE, N_SEGMENTS, MAX_LEN, EMBED_DIM, DROPOUT)
    tensor = embedding.forward(sample_seq, sample_seg)
    print(tensor.size())
    
    bert = BERT(VOCAB_SIZE, N_SEGMENTS, MAX_LEN, EMBED_DIM, N_LAYERS, ATTN_HEADS, DROPOUT)
    out = bert.forward(sample_seq, sample_seg)
    print(out)
    print(out.size())
    
