# Implementation of non-autoregressive gpt
# Outsorce some layer using torch, transformer library from huggingface
# Reference : https://www.youtube.com/watch?v=d7IRM40VMYM

import torch
import math
import torch.nn as nn
from torch.nn import functional as F

import time

# from transformers.activations import gelu_new
acc_attn = 0
attn_time = 0

class CustomGELU(nn.Module):

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
        
    
    def forward(self, x):
        """
        여기서 caching은 In-sentence caching임. Not prompt caching
        """

        output = torch.empty(0, dtype=torch.float, device = x[0].device)
        # output = torch.zeros(0, dtype=torch.float, device = "cuda:0") # TensorRT

        # for x_ in x:
        x_ = x
        cache_num = int(x_[0, -1])
        qkv = x_[:, :-1]
        # print(qkv.shape)
        given_q, given_k, given_v = qkv.split(self.n_embd, dim=1)

        T,C = given_q.shape

        T_ = T
        
        # if(cache_num > 0): #attention with qkv cache 
        T_ = T + cache_num

        # assume that assessing cache is just ignorable
        # TensorRT: No torch.rand() 
        # k = torch.concat([torch.ones(cache_num, self.n_embd, device="cuda:0"), given_k])
        # q = torch.concat([torch.ones(cache_num, self.n_embd, device="cuda:0"), given_q])
        # v = torch.concat([torch.ones(cache_num, self.n_embd, device="cuda:0"), given_v])

        # No torch.cuda.Float
        k = torch.concat([torch.rand(int(cache_num), self.n_embd, device=x_.device), given_k])
        q = torch.concat([torch.rand(int(cache_num), self.n_embd, device=x_.device), given_q])
        v = torch.concat([torch.rand(int(cache_num), self.n_embd, device=x_.device), given_v])

        # Faster Version
        # k = torch.concat([torch.cuda.FloatTensor(int(cache_num), self.n_embd).normal_(), given_k])
        # q = torch.concat([torch.cuda.FloatTensor(int(cache_num), self.n_embd).normal_(), given_q])
        # v = torch.concat([torch.cuda.FloatTensor(int(cache_num), self.n_embd).normal_(), given_v])
            
        
        # else: #normal attention
        #     k = given_k
        #     q = given_q
        #     v = given_v 
        
        k = k.view(T_, self.n_head, C // self.n_head).transpose(0, 1) # (nh, T, hs)
        q = q.view(T_, self.n_head, C // self.n_head).transpose(0, 1) # (nh, T, hs)
        v = v.view(T_, self.n_head, C // self.n_head).transpose(0, 1) # (nh, T, hs)
        
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T_,:T_] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(T_, C)
        y = y[cache_num:,:]
        output = torch.concat([output, y])

    
        return output
        


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

        
    def forward(self, x, request_position, info):
        T, C = x.size() # total length, embedding dimensionality (n_embd)
        
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k ,v  = self.c_attn(x).split(self.n_embd, dim=1)

        attention_qkv = []
        # cache_require = [-1 for i in range(len(self.request))]

        prev = 0
        # for i in range(len(request_position)):
        for i in range(request_position.shape[0]): # TensorRT

            length = request_position[i].item()
            cache_require = torch.zeros(length - prev, 1, device='cuda:0')
        
            cache_require[0,0] = info[i].item()

            q_req = q[prev:length, :]
            k_req = k[prev:length, :]
            v_req = v[prev:length, :]
            qkv = torch.cat([q_req, k_req, v_req, cache_require], dim=1)
            attention_qkv.append(qkv)

            prev = length
        
        # Implementation of selective batching using naive for loop

        y = torch.empty(0, dtype=torch.float, device = "cuda:0")
        # y = torch.zeros(0, dtype=torch.float, device = "cuda:0")
        for inputs in attention_qkv:
            # print(inputs.shape)
            output = self.sel_attn(inputs)
            # print(output)
            y = torch.concat([y,output], dim=0)

        # implementation of selective batching using data parallel module
        # total 4 data path of parallel execution
        # I think batch size is up to 4 now.

        # TODO :  Data parallel library가 그냥 for 문으로 돌리는 것보다 느림 -> 이유 찾고 optimize 해야함.
        
        # attention_len = len(attention_qkv)
        # inputs = [[] for _ in range(self.devices)]
        # for i in range(attention_len):
        #     inputs[(i+1)%2].append(attention_qkv[i].to(self.device_ids[(i+1)%2]))

        # replicas = self.replicas[:attention_len]
        
        # outputs = nn.parallel.parallel_apply(replicas, inputs)
        # y = nn.parallel.gather(outputs, "cuda:0")

        # output projection
        y = self.resid_dropout(self.c_proj(y))
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
            nn.Dropout(resid_pdrop),
        ).to("cuda:0")

    def forward(self, x, request_position, info):

        x_ = self.ln_1(x)  # (total_tokens, n_embd)  
        attn_out = self.attention(x_, request_position, info) # (total_tokens, n_embd)
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


    def forward(self, idx, info, pos, length):
        """
        Example

        Input : [1, total_token]
        Input = ["hello I am (encoding phase)", "wow (decoding phase)", "my name is (encoding phase)"]
        
        info : [1, batch_size]
        info = [0, n, 0] => "n" is used for qkv cache generation, scheduler will record all requests' phase information
        
        pos : [1, total_token]
        pos = [0, 1, 2, n, 0, 1, 2] => input for positional encoding

        length : [1, batch_size]
        length = [3,4,7] => for spliting : each request's end point

        tensor.item() 

        tensor : GPU only로 생각
        TODO: for loop 써도

        result : [1, batch_size]
        각 request 다음 token
        """


        token_emb = self.token_emb(idx)  # (total_tokens, n_embd)
        pos_emb = self.pos_emb(pos)  # (total_tokens, n_embd)
        x = self.drop(token_emb + pos_emb)  # (total_tokens, n_embd)


        # x = self.blocks(x, length, info)  # (total_tokens, n_embd)
        for layer in self.blocks:
            x = layer(x, length, info)

        x = self.ln(x)  # (total_tokens, n_embd)
        logits = self.head(x)  # (total_tokens, vocab_size)

        # request_end = [i.item()-1 for i in length]
        request_end = [length[i].item() - 1 for i in range(length.shape[0])] # TensorRT
        new_token = logits[request_end]/1.0
    
        probs = torch.nn.functional.softmax(new_token, dim=1)
       
        new_token_idx = torch.argmax(probs, dim = 1)

        return new_token_idx.view(length.shape[0], 1).to("cpu")