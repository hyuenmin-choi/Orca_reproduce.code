# Implementation of non-autoregressive gpt
# Outsorce some layer using torch, transformer library from huggingface
# Reference : https://www.youtube.com/watch?v=d7IRM40VMYM

import torch
import torch.nn as nn

from transformers.activations import gelu_new

import time
import math
from torch.nn import functional as F

attn_time = 0
input_len = 0
iteration = 0

class CustomGELU(nn.Module):
    """GELU implementation taken from the `transformers`."""

    def forward(self, x):
        """Run forward pass."""
        return gelu_new(x)
class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self,
                    *,
                    embed_dim,
                    num_heads,
                    dropout,
                    n_positions):

        super().__init__()
        
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(embed_dim, 3 * embed_dim)
        # output projection
        self.c_proj = nn.Linear(embed_dim, embed_dim)
        # regularization
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("bias", torch.tril(torch.ones(n_positions, n_positions))
                                     .view(1, 1, n_positions, n_positions))
        self.n_head = num_heads
        self.n_embd = embed_dim
        self.position = list()
        self.request = list()

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # print(x.shape)
        global input_len, iteration

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # print(self.c_attn(x).shape)
        q, k ,v  = self.c_attn(x).split(self.n_embd, dim=2)
        # print(q.shape)

        T_ = T
        cache_num = 0
        
        # with qkv caching
        # if(iteration > 0 and T == 1):
        #     # print("here")
        #     cache_num = input_len+iteration
        #     if(cache_num > 1023):
        #         cache_num = 1023
            
        #     T_ = T + cache_num

        #     # print(k.shape)
        #     k = torch.concat([torch.randn(B, cache_num, self.n_embd).to(x.device), k], dim = 1)
        #     q = torch.concat([torch.randn(B, cache_num, self.n_embd).to(x.device), q], dim = 1)
        #     v = torch.concat([torch.randn(B, cache_num, self.n_embd).to(x.device), v], dim = 1) 

        k = k.view(B, T_, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T_, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T_, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T_,:T_] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T_, C) # re-assemble all head outputs side by side
        
        y = y[:,(cache_num):,:]
        # output projection
        y = self.resid_dropout(self.c_proj(y))
        # print(y.shape)
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

        self.ln_1 = nn.LayerNorm(n_embd, eps=layer_norm_epsilon)
        self.ln_2 = nn.LayerNorm(n_embd, eps=layer_norm_epsilon)


        self.attention = CausalSelfAttention(
            embed_dim=n_embd,
            num_heads=n_head,
            dropout=attn_pdrop,
            n_positions=n_positions,
            # bias=True,
            # batch_first=True,
        )

        self.register_buffer(
            "mask",
            (1 - torch.tril(torch.ones(n_positions, n_positions))).to(
                dtype=torch.bool
            ),
        )

        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            CustomGELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(resid_pdrop),
        )

    # @torch.jit.script
    def forward(self, x):
        """Run forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape `(batch_size, n_tokens, n_embd)`.

        Returns
        -------
        torch.Tensor
            Output tensor of shape `(batch_size, n_tokens, n_embd)`.
        """
        batch_size, n_tokens, n_embd = x.shape
        
        global attn_time

        x_ = self.ln_1(x)  # (batch_size, n_tokens, n_embd)

        mask = self.mask[:n_tokens, :n_tokens]  # (n_tokens, n_tokens)
        start = time.time()
        # attn_out = self.attention(
        #     x_, x_, x_, attn_mask=mask, need_weights=False
        # )  
        attn_out = self.attention(x_)
        # (batch_size, n_tokens, n_embd)
        end = time.time()

        # print(f"attn time {end-start}")
        attn_time += (end-start)
        x = x + attn_out  # (batch_size, n_tokens, n_embd)
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

        self.blocks = nn.Sequential(
            *[
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
        ).to("cuda:0")
        self.ln = nn.LayerNorm(n_embd, eps=layer_norm_epsilon).to("cuda:0")
        self.head = nn.Linear(n_embd, vocab_size, bias=False).to("cuda:0")
        self.decode = dict()
        self.eos = 50256


    def forward(self, idx, user_ids):
        processed_input = list()
        processed_position = list()
        request_position = list()

        global attn_time
        global input_len
        global iteration

        iteration = (iteration % 32)
        # print(f"iteration : {iteration}, input_len : {input_len}")
        attn_time = 0

        idx = torch.tensor(idx, dtype=torch.int32, device = "cuda:0")
        user_ids = torch.tensor(user_ids, dtype=torch.int32, device = "cuda:0")
        
        batch_size, n_tokens = idx.shape
        if n_tokens > 1:
            input_len = n_tokens

        device = torch.device("cuda:0")

        if n_tokens > self.n_positions:
            raise ValueError("There are too many tokens in the input")
        

        for index in range(batch_size):
            #each request inside batch # (max_n_token)

            request = idx[index, :]
            request = request.tolist()
            user_id = user_ids[index].item()

            if self.eos in request:
                end = request.index(self.eos)
                request = request[:end]

            if self.decode.get(user_id) is not None:
                if request[0] != request[-1]:
                    raise ValueError("decode process need only one input")

                past = self.decode[user_id]
                self.decode.pop(user_id, None)

            
            request_position.append(len(request)-1)


        # request = torch.tensor(self.request_position)

        positions = torch.arange(n_tokens, device="cuda:0")  # (n_tokens,)

        token_emb = self.token_emb(idx)  # (total_tokens, n_embd)
        pos_emb = self.pos_emb(positions)  # (total_tokens, n_embd)
        x = self.drop(token_emb + pos_emb)  # (total_tokens, n_embd)
        # print(f"processed input shape", x.shape)

        x = self.blocks(x)  # (total_tokens, n_embd)


        x = self.ln(x)  # (total_tokens, n_embd)
        logits = self.head(x)  # (total_tokens, vocab_size)
        # print(logits.shape)
        # request_end = [i-1 for i in request_position]
        new_token = logits[: , -1 , :]/1.0
        # print(new_token.shape)
        probs = torch.nn.functional.softmax(new_token, dim= 1)
       
        new_token_ix = probs.argmax(dim = 1)
        # print(new_token_ix.shape)
 
        for i in range(len(new_token_ix)):
            # res = new_token_ix[i].item()
            self.decode[user_ids[i].item()] = request_position[i]
        
        # print(f"total attn time {attn_time}")
        # print(f"avg attn time {attn_time/12.0}")

        iteration += 1

        return new_token_ix.view(batch_size, 1).to("cpu")