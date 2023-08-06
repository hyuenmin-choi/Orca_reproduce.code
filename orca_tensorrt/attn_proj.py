import torch
import torch.nn as nn
from torch.nn import functional as F

class CustomGELU(nn.Module):
    # @torch.no_grad()
    def forward(self, x):
        """Run forward pass."""
        return x * 0.5 * (1.0 + torch.erf(x / 1.41421))

class GPT_Projection(nn.Module):
    def __init__(self, n_embd, layer_norm_epsilon):
        super().__init__()
        def init_normal(m):
            if type(m) == nn.Linear:
                nn.init.uniform_(m.weight)
                m.bias.data.fill_(0.1)

        self.c_proj = nn.Linear(n_embd, n_embd, device="cuda:0")
        self.c_proj.apply(init_normal)

        # self.resid_dropout = nn.Dropout(dropout) 
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd, device="cuda:0"),
            CustomGELU(),
            nn.Linear(4 * n_embd, n_embd, device="cuda:0"),
            # nn.Dropout(resid_pdrop),
        ).to("cuda:0")

        self.ln_1 = nn.LayerNorm(n_embd, eps=layer_norm_epsilon, device="cuda:0")
        self.ln_2 = nn.LayerNorm(n_embd, eps=layer_norm_epsilon, device="cuda:0")
        
    @torch.no_grad()
    def forward(self, x, y): # x: original input, y: attention input
        y = self.ln_1(y)
        attn_out = self.c_proj(y)
        x = x + attn_out  # (total_tokens, n_embd)
        x = x + self.mlp(self.ln_2(x))  # (batch_size, n_tokens, n_embd)
        return x