import torch
import torch.nn as nn
from torch.nn import functional as F
import math

class GPT_Attention(nn.Module):
    def __init__(
        self,
        n_heads,
        embed_dim,
        n_positions,
        ):
        
        super().__init__()

        self.n_embd = embed_dim
        self.n_head = n_heads
        self.n_pos = n_positions

        self.register_buffer("bias", torch.tril(torch.ones(self.n_pos, self.n_pos, device="cuda:0"))
                                     .view(1, 1, self.n_pos, self.n_pos))
        return

    # @torch.no_grad()
    def forward(self, q, k, v):

        # k = torch.concat([torch.rand(cache_num, self.n_embd, device='cuda:0'), k])
        # q = torch.concat([torch.rand(cache_num, self.n_embd, device='cuda:0'), q])
        # v = torch.concat([torch.rand(cache_num, self.n_embd, device='cuda:0'), v])
        
        k = k.view(-1, self.n_head, self.n_embd//self.n_head).transpose(0, 1) # (nh, T, hs)
        q = q.view(-1, self.n_head, self.n_embd//self.n_head).transpose(0, 1) # (nh, T, hs)
        v = v.view(-1, self.n_head, self.n_embd//self.n_head).transpose(0, 1) # (nh, T, hs)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.n_embd//self.n_head))
        # att = att.masked_fill(self.bias[:,:,:T_,:T_] == 0, float('-inf'))
        att = torch.nn.functional.softmax(att, dim=-1)
        
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(-1, self.n_embd)
        # y = y[cache_num:,:]

        return y