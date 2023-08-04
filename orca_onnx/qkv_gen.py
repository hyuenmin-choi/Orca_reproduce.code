import torch

class GPT_qkvgen(nn.Module):
    def __init__(
        self,
        *,
        embed_dim,):

        super().__init__()

        def init_normal(m):
            if type(m) == nn.Linear:
                nn.init.uniform_(m.weight)
                m.bias.data.fill_(0.1)

        self.c_attn = nn.Linear(embed_dim, 3 * embed_dim).to("cuda:0")
        self.c_attn.apply(init_normal)

        self.n_embd = embed_dim
    
    def forward(self, input):

        q,k,v = self.c_attn(x).split(self.n_embd, dim=1)

        return q, k, v