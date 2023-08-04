import torch
import torch.nn as nn
from torch.nn import functional as F

class CustomGELU(nn.Module):
    @torch.no_grad()
    def forward(self, x):
        """Run forward pass."""
        return x * 0.5 * (1.0 + torch.erf(x / 1.41421))

class GPT_Projection(nn.module):
    def __init__(self, n_embd, layer_norm_epsilon):

        def init_normal(m):
            if type(m) == nn.Linear:
                nn.init.uniform_(m.weight)
                m.bias.data.fill_(0.1)

        self.c_proj = nn.Linear(n_embd, n_embd).to("cuda:0")
        self.c_proj.apply(init_normal)

        # self.resid_dropout = nn.Dropout(dropout).to("cuda:0")

        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            CustomGELU(),
            nn.Linear(4 * n_embd, n_embd),
            # nn.Dropout(resid_pdrop),
        ).to("cuda:0")

        self.ln = nn.LayerNorm(n_embd, eps=layer_norm_epsilon).to("cuda:0")

    def forward(self, x, y): # x: original input, y: attention input
        attn_out = self.c_proj(y)
        x = x + attn_out  # (total_tokens, n_embd)
        x = x + self.mlp(self.ln(x))  # (batch_size, n_tokens, n_embd)
        return x