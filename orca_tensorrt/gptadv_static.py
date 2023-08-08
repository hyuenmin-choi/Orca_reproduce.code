# Implementation of non-autoregressive gpt
# Outsorce some layer using torch, transformer library from huggingface
# Reference : https://www.youtube.com/watch?v=d7IRM40VMYM

import torch
import math
import torch.nn as nn
from torch.nn import functional as F
import torch_tensorrt
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import math

from utils import copy_model, generate_token

import time

# from transformers.activations import gelu_new
acc_attn = 0
attn_time = 0

class CustomGELU(nn.Module):
    # @torch.no_grad()
    def forward(self, x):
        """Run forward pass."""
        return x * 0.5 * (1.0 + torch.erf(x / 1.41421))

class SelectiveAttention(nn.Module):
    def __init__(self,
        *,
        n_heads,
        embed_dim,
        n_positions,
        drop_out,):

        super().__init__()

        self.n_embd = embed_dim
        self.n_head = n_heads
        self.n_pos = n_positions
        self.attn_dropout = nn.Dropout(drop_out)

        self.register_buffer("bias", torch.tril(torch.ones(self.n_pos, self.n_pos)).to("cuda:0")
                                     .view(1, 1, self.n_pos, self.n_pos))
        
    # @torch.no_grad()
    def forward(self, q, k ,v):
        """
        여기서 caching은 In-sentence caching임. Not prompt caching
        """

        cache_num = 32

        # assume that assessing cache is just ignorable
        # TensorRT: No torch.rand() 
        k = torch.concat([torch.ones(cache_num, self.n_embd, device="cuda:0"), k])
        q = torch.concat([torch.ones(cache_num, self.n_embd, device="cuda:0"), q])
        v = torch.concat([torch.ones(cache_num, self.n_embd, device="cuda:0"), v])

        k = k.view(-1, self.n_head, self.n_embd//self.n_head).transpose(0, 1) # (nh, T, hs)
        q = q.view(-1, self.n_head, self.n_embd//self.n_head).transpose(0, 1) # (nh, T, hs)
        v = v.view(-1, self.n_head, self.n_embd//self.n_head).transpose(0, 1) # (nh, T, hs)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.n_embd//self.n_head))
        # att = att.masked_fill(self.bias[:,:,:T_,:T_] == 0, float('-inf'))
        att = torch.nn.functional.softmax(att, dim=-1)
        
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(-1, self.n_embd)
        y = y[cache_num:,:]

        return y
        


class CausalSelfAttention(nn.Module):

    def __init__(self,
        *,
        embed_dim,
        num_heads,
        dropout,
        n_positions):

        super().__init__()
        # assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        
        # n_embd = 768 # fixed for gpt-2
        def init_normal(m):
            if type(m) == nn.Linear:
                nn.init.uniform_(m.weight)
                m.bias.data.fill_(0.1)

        self.c_attn = nn.Linear(embed_dim, 3 * embed_dim).to("cuda:0")
        self.c_attn.apply(init_normal)
        # output projection
        self.c_proj = nn.Linear(embed_dim, embed_dim).to("cuda:0")
        self.c_proj.apply(init_normal)
        # regularization
        self.resid_dropout = nn.Dropout(dropout).to("cuda:0")

        self.sel_attn = SelectiveAttention(n_heads=num_heads,
                                            embed_dim=embed_dim,
                                            n_positions=n_positions,
                                            drop_out=dropout)

        self.n_embd = embed_dim
        

        # self.device_ids = ['cuda:0','cuda:1']
        # self.devices = len(self.device_ids)
        # self.replicas = nn.parallel.replicate(self.sel_attn, self.device_ids)

    @torch.no_grad()    
    def forward(self, x):
        # X size : (1, n_embd)
        # T, C = x.size() # total length, embedding dimensionality (n_embd)
        
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k ,v  = self.c_attn(x).split(self.n_embd, dim=1)

        y = self.sel_attn(q, k ,v)
        
        y = self.c_proj(y)
        return y


class Block(nn.Module):
    def __init__(
        self,
        *,
        n_embd,
        n_head,
        n_positions,
        attn_pdrop,
        resid_pdrop,
        layer_norm_epsilon,
    ):
        super().__init__()

        self.ln_1 = nn.LayerNorm(n_embd, eps=layer_norm_epsilon).to("cuda:0")
        self.ln_2 = nn.LayerNorm(n_embd, eps=layer_norm_epsilon).to("cuda:0")

        self.attention = CausalSelfAttention(
            embed_dim=n_embd,
            num_heads=n_head,
            dropout=attn_pdrop,
            n_positions=n_positions,
        )

        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            CustomGELU(),
            nn.Linear(4 * n_embd, n_embd),
            # nn.Dropout(resid_pdrop)
        ).to("cuda:0")
        
    @torch.no_grad()
    def forward(self, x):

        x_ = self.ln_1(x)  # (total_tokens, n_embd)  
        attn_out = self.attention(x_) # (total_tokens, n_embd)
        x = x + attn_out  # (total_tokens, n_embd)
        x = x + self.mlp(self.ln_2(x))  # (batch_size, n_tokens, n_embd)

        return x


class GPT(nn.Module):

    def __init__(
        self,
        *,
        vocab_size,
        n_layer,
        n_embd,
        n_head,
        n_positions,
        attn_pdrop,
        embd_pdrop,
        resid_pdrop,
        layer_norm_epsilon,
    ):
        super().__init__()
        self.n_positions = n_positions
        self.token_emb = nn.Embedding(vocab_size, n_embd).to("cuda:0")
        self.pos_emb = nn.Embedding(n_positions, n_embd).to("cuda:0")

        self.drop = nn.Dropout(embd_pdrop).to("cuda:0")

        self.blocks = nn.ModuleList(
            [
                Block(
                    n_embd=n_embd,
                    n_head=n_head,
                    n_positions=n_positions,
                    attn_pdrop=attn_pdrop,
                    resid_pdrop=resid_pdrop,
                    layer_norm_epsilon=layer_norm_epsilon,
                )
                for _ in range(n_layer)
            ]
        )
        self.ln = nn.LayerNorm(n_embd, eps=layer_norm_epsilon).to("cuda:0")
        self.head = nn.Linear(n_embd, vocab_size, bias=False).to("cuda:0")
        # self.decode : dict[int, int] = {-1:-1}
        self.eos = 50256

    # @torch.no_grad()
    def forward(self, idx, pos):
        """
        Example

        Input : [1, total_token]
        Input = ["hello I am (encoding phase)", "wow (decoding phase)", "my name is (encoding phase)"]
        
        pos : [1, total_token]
        pos = [0, 1, 2, n, 0, 1, 2] => input for positional encoding

        TODO: for loop 써도

        result : [1, batch_size]
        각 request 다음 token
        """

        token_emb = self.token_emb(idx)  # (total_tokens, n_embd)
        pos_emb = self.pos_emb(pos)  # (total_tokens, n_embd)
        x = token_emb + pos_emb  # (total_tokens, n_embd)

        # x = self.blocks(x, length, info)  # (total_tokens, n_embd)

        for layer in self.blocks:
            x = layer(x)

        x = self.ln(x)  # (total_tokens, n_embd)
        logits = self.head(x)  # (total_tokens, vocab_size)

        new_token = logits[-1].view(-1, 50257)
    
        probs = F.softmax(new_token, dim = 1)
       
        new_token_idx = torch.argmax(probs, dim = 1)

        return new_token_idx.view(-1, 1)
    
if __name__ == "__main__":
    model_official = AutoModelForCausalLM.from_pretrained("gpt2")
    config_official = model_official.config

    our_params = [
        "vocab_size",
        "n_layer",
        "n_embd",
        "n_head",
        "n_positions",
        "attn_pdrop",
        "embd_pdrop",
        "resid_pdrop",
        "layer_norm_epsilon",
    ]

    config_ours = {k: getattr(config_official, k) for k in our_params}

    model_ours = GPT(**config_ours)
    # model_ours = GPT(**config_ours)
    model_ours.eval()
    model_ours.zero_grad(True)

    copy_model(model_official, model_ours)
    # with torch_tensorrt.logging.debug():
    trt_model = torch_tensorrt.compile(model_ours, inputs = [
        torch_tensorrt.Input( # concated input
            min_shape=(1,),
            opt_shape=(8,),
            max_shape=(16,), 
            dtype=torch.int32),
        torch_tensorrt.Input( # concated input
            min_shape=(1,),
            opt_shape=(8,),
            max_shape=(16,), 
            dtype=torch.int32)],
        enabled_precisions = torch.float32, # Run with FP32
        workspace_size = 1 << 33,
        require_full_compilation = True
    )

    input1 = torch.tensor([15496,  2159,   318,  3666,  5181,   318,   437,  7777], dtype=torch.int32, device='cuda:0')
    input2 = torch.tensor([15496], dtype=torch.int32, device='cuda:0')
    pos1 = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7], dtype=torch.int32, device='cuda:0')
    pos2 = torch.tensor([0], dtype=torch.int32, device='cuda:0')

    # warm up
    for _ in range(30):
        trt_model(input1, pos1)
        trt_model(input2, pos2)

    start = time.time()
    output = trt_model(input1, pos1)
    for _ in range(31):
        output = trt_model(output[0], pos2)
    end = time.time()
    print("output :", output)
    print("total :", (end - start)*1000)