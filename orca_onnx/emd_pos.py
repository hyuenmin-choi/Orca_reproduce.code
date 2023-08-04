import torch
import torch.nn as nn
from torch.nn import functional as F

class GPT_Embedding(nn.Module):
    def __init__(
        self,
        *,
        vocab_size,
        n_embd,
        n_positions
    ):

        super().__init__()
        self.n_positions = n_positions
        self.token_emb = nn.Embedding(vocab_size, n_embd).to("cuda:0")
        self.pos_emb = nn.Embedding(n_positions, n_embd).to("cuda:0")

    def forward(self, input, pos):
        # pos 그냥 input이랑 size 똑같은 dummy tensor 넣으면 됨.
        
        token_emb = self.token_emb(input)  # (total_tokens, n_embd)
        pos_emb = self.pos_emb(pos)  # (total_tokens, n_embd)
        x = token_emb + pos_emb

        return x
