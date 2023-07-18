# Implementation of non-autoregressive gpt
# Outsorce some layer using torch, transformer library from huggingface
# Reference : https://www.youtube.com/watch?v=d7IRM40VMYM

import torch
import math
import torch.nn as nn
from torch.nn import functional as F

from transformers.activations import gelu_new


class CustomGELU(nn.Module):
    """GELU implementation taken from the `transformers`."""

    def forward(self, x):
        """Run forward pass."""
        return gelu_new(x)

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
        self.register_buffer("bias", torch.tril(torch.ones(n_positions, n_positions))
                                     .view(1, 1, n_positions, n_positions))
        self.attn_dropout = nn.Dropout(drop_out)
        
    
    def forward(self, x):
        
        # print(x.shape)
        cache_num = int(x[0, -1])
        qkv = x[:, :-1]
        # print(qkv.shape)
        given_q, given_k, given_v = qkv.split(self.n_embd, dim=1)

        T,C = given_q.shape

        T_ = T
        
        if(cache_num > 0): #attention with qkv cache
            T_ = T + cache_num

            #assume that assessing cache is just ignorable
            k = torch.concat([torch.randn(int(cache_num), self.n_embd), given_k])
            q = torch.concat([torch.randn(int(cache_num), self.n_embd), given_q])
            v = torch.concat([torch.randn(int(cache_num), self.n_embd), given_v])
        
        else: #normal attention
            k = given_k
            q = given_q
            v = given_v
        
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

        return y
        


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
        # assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        
        # n_embd = 768 # fixed for gpt-2
        def init_normal(m):
            if type(m) == nn.Linear:
                nn.init.uniform_(m.weight)
                m.bias.data.fill_(0.1)

        self.c_attn = nn.Linear(embed_dim, 3 * embed_dim)
        self.c_attn.apply(init_normal)
        # output projection
        self.c_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj.apply(init_normal)
        # regularization
        self.resid_dropout = nn.Dropout(dropout)

        self.sel_attn = SelectiveAttention(n_heads=num_heads,
                                            embed_dim=embed_dim,
                                            n_positions=n_positions,
                                            drop_out=dropout)

        self.n_embd = embed_dim
        self.position = list()
        self.request = list()

        

    def forward(self, x):
        T, C = x.size() # total length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k ,v  = self.c_attn(x).split(self.n_embd, dim=1)

        attnetion_qkv = []
        cache_require = [-1 for i in range(len(self.request))]
        

        prev = 0
        for i in range(len(self.request)):

            length = self.request[i]
            cache_require = torch.zeros(length - prev, 1)
            # print(cache_require)
            
            if self.position[prev] != 0:
                # print(f"position prev :{self.position[prev]}")
                if prev == 0:
                    cache_require[0, 0] = self.position[prev]
                else:
                    cache_require[0, 0] = self.position[prev] - self.position[prev - 1]

            q_req = q[prev:length, :]
            k_req = k[prev:length, :]
            v_req = v[prev:length, :]
            qkv = torch.concat([q_req, k_req, v_req, cache_require], dim=1)
            attnetion_qkv.append(qkv)
            # print(qkv.shape)
            prev = length

        # selective_input = torch.stack(attnetion_qkv, dim=0)
        # print(selective_input.shape)

        #Implementation of selective batching using naive for loop

        y = torch.tensor([])

        for inputs in attnetion_qkv:
            output = self.sel_attn(inputs)
            y = torch.concat([y,output], dim=0)

        #implementation of selective batching using data parallel module
        #total 4 data path of parallel execution
        #I think batch size is up to 4 now.

        # TODO :  should be allow more flexible batching up to 32
        
        # device_ids = ['cuda:4','cuda:5','cuda:6','cuda:7']

        # replicas = nn.parallel.replicate(self.sel_attn, device_ids)
        # replicas = replicas[:len(attnetion_qkv)]
        # outputs = nn.parallel.parallel_apply(replicas, inputs)
        # nn.parallel.gather(outputs, output_device=None)

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class Block(nn.Module):
    """Decoder block.

    Parameters
    ----------
    n_embd : int
        Dimensionality of the embeddings.

    n_head : int
        Number of attention heads.

    n_positions : int
        Maximum number of tokens.

    attn_pdrop : float
        Probability of dropout on attention weights.

    resid_pdrop : float
        Probability of dropout after applying the MLP.

    layer_norm_epsilon : float
        Hyperparameter of layer normalization.

    Attributes
    ----------
    ln_1, ln_2 : nn.LayerNorm
        Layer norms.

    attention : nn.MultiHeadAttention
        Attention module.

    mlp : nn.Sequential
        Multilayer perceptron.

    """

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
            Input tensor of shape `(1, total_tokens, n_embd)`.

        Returns
        -------
        torch.Tensor
            Output tensor of shape `(batch_size, n_tokens, n_embd)`.
        """
        # batch_size, n_tokens, n_embd = x.shape

        # is layer normalization is really critical?
        x_ = self.ln_1(x)  # (total_tokens, n_embd)  


        attn_out = self.attention(x_) # (total_tokens, n_embd)
        x = x + attn_out  # (total_tokens, n_embd)
        x = x + self.mlp(self.ln_2(x))  # (batch_size, n_tokens, n_embd)

        return x


class GPT(nn.Module):
    """Entire GPT model.

    Parameters
    ----------
    vocab_size : int
        Number of tokens in the vocabulary.

    n_layer : int
        Number of decoder blocks to include.

    n_embd : int
        Dimensionality of the embeddings.

    n_head : int
        Number of attention heads.

    n_positions : int
        Maximum number of tokens.

    attn_pdrop : float
        Probability of dropout on attention weights.

    embd_pdrop : float
        Probability of dropout on the sum of embeddings.

    resid_pdrop : float
        Probability of dropout after applying the MLP.

    layer_norm_epsilon : float
        Hyperparameter of layer normalization.

    Attributes
    ----------
    token_emb : nn.Embedding
        Token embeddings.

    pos_emb : nn.Embedding
        Positional embedding.

    drop : nn.Dropout
        Dropout module to be applied on the sum of embeddings.

    blocks : nn.Sequential
        List of decoder blocks.

    ln : nn.LayerNorm
        Layer norm applied before applying `head`.

    head : nn.Linear
        Final linear layer.
    """

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
        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(n_positions, n_embd)

        self.drop = nn.Dropout(embd_pdrop)

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
        )
        self.ln = nn.LayerNorm(n_embd, eps=layer_norm_epsilon)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)
        self.decode = dict()
        self.eos = 50256


    def forward(self, idx, user_ids):
        """Run forward pass.

        Parameters
        ----------
        idx : torch.Tensor
            Integer tensor of shape `(batch_size, n_tokens)` where each
            element is in the range `[0, vocab_size)`.

        Returns
        -------
        logits : torch.Tensor
            Tensor of shape `(batch_size, n_tokens, vocab_size)`.
        """
        processed_input = list()
        processed_position = list()
        request_position = list()
        
        batch_size, n_tokens = idx.shape
        device = idx.device
        device = torch.device("cuda:0")

        if n_tokens > self.n_positions:
            raise ValueError("There are too many tokens in the input")

        
        for index in range(batch_size):
            #each request inside batch # (max_n_token)

            request = idx[index]
            request = request.tolist()
            user_id = user_ids[index]

            if self.eos in request:
                end = request.index(self.eos)
                request = request[:end]

            if self.decode.get(user_id) is not None:
                
                if request[0] != request[-1]:
                    raise ValueError("decode process need only one input")

                past = self.decode[user_id]
                processed_input.append(request[-1])
                processed_position.append(past)
                self.decode.pop(user_id, None)

            else:
                index = len(request)
                processed_input.extend(request)
                processed_position.extend(list(range(index)))
            
            request_position.append(len(processed_position))



        processed_input = torch.tensor(processed_input).to(device)
        positions = torch.tensor(processed_position).to(device)
        # request = torch.tensor(self.request_position)

        # positions = torch.arange(n_tokens, device=device)  # (n_tokens,)

        token_emb = self.token_emb(processed_input)  # (total_tokens, n_embd)
        pos_emb = self.pos_emb(positions)  # (total_tokens, n_embd)
        x = self.drop(token_emb + pos_emb)  # (total_tokens, n_embd)
        # print(f"processed input shape", x.shape)

        for name, child in self.blocks.named_children():
            child.attention.position = []
            child.attention.position.extend(processed_position)
            child.attention.request = []
            child.attention.request.extend(request_position)

        x = self.blocks(x)  # (total_tokens, n_embd)


        x = self.ln(x)  # (total_tokens, n_embd)
        logits = self.head(x)  # (total_tokens, vocab_size)

        request_end = [i-1 for i in request_position]
        new_token = logits[request_end]/1.0
        probs = torch.nn.functional.softmax(new_token, dim=1)
       
        new_token_ix = probs.argmax(dim = 1)
 
        for i in range(len(new_token_ix)):
            # res = new_token_ix[i].item()
            self.decode[user_ids[i]] = request_position[i]

        return new_token_ix.view(batch_size, 1)