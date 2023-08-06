import torch
import torch.nn as nn
from torch.nn import functional as F

class GPT_Output(nn.Module):
    def __init__(self, n_embd, layer_norm_epsilon, vocab_size):
        super().__init__()
        self.ln = nn.LayerNorm(n_embd, eps=layer_norm_epsilon).to("cuda:0")

        #weight 안들어감
        self.head = nn.Linear(n_embd, vocab_size, bias=False).to("cuda:0")
        
    @torch.no_grad()
    def forward(self, x):
        x = self.ln(x)  # (total_tokens, n_embd)
        logits = self.head(x)  # (total_tokens, vocab_size)

        # logits를 output으로 받아서 밖에서 length에 따라 나눠주는 작업을 진행한다.
        return logits