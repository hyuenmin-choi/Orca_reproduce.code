import torch
from torch.onnx import export
import torch_tensorrt
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from attn_proj import GPT_Projection
from attn import GPT_Attention
from emd_pos import GPT_Embedding
from out_gen import GPT_Output
from qkv_gen import GPT_QKVgen
import time

class GPT_TRT():
    def __init__(self):
        super().__init__()# model config
        
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

        self.config_ours = {k: getattr(config_official, k) for k in our_params}

        # initialize models
        emd_pos = GPT_Embedding(self.config_ours['vocab_size'], self.config_ours['n_embd'], self.config_ours['n_positions'])
        qkv_gen = GPT_QKVgen(self.config_ours['n_embd'])
        attn = GPT_Attention(self.config_ours['n_head'], self.config_ours['n_embd'], self.config_ours['n_positions'])
        attn_proj = GPT_Projection(self.config_ours['n_embd'], self.config_ours['layer_norm_epsilon'])
        out_gen = GPT_Output(self.config_ours['n_embd'], self.config_ours['layer_norm_epsilon'], self.config_ours['vocab_size'])

        # export models to tensorrt

        # embedding
        self.emd_model = torch_tensorrt.compile(emd_pos, inputs = [
            torch_tensorrt.Input( # concated input
                min_shape=(1,),
                opt_shape=(8,),
                max_shape=(16,), 
                dtype=torch.int32), 
            torch_tensorrt.Input( # pos
                min_shape=(1,),
                opt_shape=(8,),
                max_shape=(16,), 
                dtype=torch.int32)],
            enabled_precisions = torch.float32, # Run with FP32
            workspace_size = 1 << 33,
            require_full_compilation = True
        )

        input = torch.tensor([15496,  2159,   318,  3666,  5181,   318,  1324,  2342], dtype=torch.int32, device='cuda:0')
        pos = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7], dtype=torch.int32, device='cuda:0')

        # warm up
        for _ in range(30):
            self.emd_model(input, pos)

        # start = time.time()
        output1 = self.emd_model(input, pos)
        # end = time.time()
        # print("emb", (end-start)*1000)

        # print(output1)
        # print(output1.shape)

        # qkv generation
        self.qkv_model = torch_tensorrt.compile(qkv_gen, inputs = [
            torch_tensorrt.Input( # concated input
                min_shape=(1, self.config_ours['n_embd']),
                opt_shape=(8, self.config_ours['n_embd']),
                max_shape=(16, self.config_ours['n_embd']), 
                dtype=torch.float32)],
            enabled_precisions = torch.float32, # Run with FP32
            workspace_size = 1 << 33,
            require_full_compilation = True
        )

        # warm up
        for _ in range(30):
            self.qkv_model(output1)

        # start = time.time()
        output2 = self.qkv_model(output1)
        # end = time.time()
        # print("qkv", (end-start)*1000)

        # print(output2) # output is tuple
        # print(output2[0].shape)
        # print(output2[1].shape)
        # print(output2[2].shape)

        # attatch cache


        # attention
        self.attn_model = torch_tensorrt.compile(attn, inputs = [
            torch_tensorrt.Input( # concated input
                min_shape=(1, self.config_ours['n_embd']),
                opt_shape=(8, self.config_ours['n_embd']),
                max_shape=(16, self.config_ours['n_embd']), 
                dtype=torch.float32),
            torch_tensorrt.Input( # concated input
                min_shape=(1, self.config_ours['n_embd']),
                opt_shape=(8, self.config_ours['n_embd']),
                max_shape=(16, self.config_ours['n_embd']), 
                dtype=torch.float32),
            torch_tensorrt.Input( # concated input
                min_shape=(1, self.config_ours['n_embd']),
                opt_shape=(8, self.config_ours['n_embd']),
                max_shape=(16, self.config_ours['n_embd']), 
                dtype=torch.float32)],
            enabled_precisions = torch.float32, # Run with FP32
            workspace_size = 1 << 33,
            require_full_compilation = True
        )

        # warm up
        for _ in range(30):
            self.attn_model(*output2)

        # projection
        self.proj_model = torch_tensorrt.compile(attn_proj, inputs = [
            torch_tensorrt.Input( # concated input
                min_shape=(1, self.config_ours['n_embd']),
                opt_shape=(8, self.config_ours['n_embd']),
                max_shape=(16, self.config_ours['n_embd']), 
                dtype=torch.float32),
            torch_tensorrt.Input( # concated input
                min_shape=(1, self.config_ours['n_embd']),
                opt_shape=(8, self.config_ours['n_embd']),
                max_shape=(16, self.config_ours['n_embd']), 
                dtype=torch.float32)],
            enabled_precisions = torch.float32, # Run with FP32
            workspace_size = 1 << 33,
            require_full_compilation = True
        )

        output3 = torch.empty((8, self.config_ours['n_embd']), device="cuda:0")
        for _ in range(30):
            self.proj_model(output3, output1)

        # start = time.time()
        output4 = self.proj_model(output3, output1)
        # end = time.time()
        # print("proj", (end-start)*1000)

        # logits
        self.out_model = torch_tensorrt.compile(out_gen, inputs = [
            torch_tensorrt.Input( # concated input
                min_shape=(1, self.config_ours['n_embd']),
                opt_shape=(8, self.config_ours['n_embd']),
                max_shape=(16, self.config_ours['n_embd']), 
                dtype=torch.float32)],
            enabled_precisions = torch.float32, # Run with FP32
            workspace_size = 1 << 33,
            require_full_compilation = True
        )

        # warm up
        for _ in range(30):
            self.out_model(output4)

        # start = time.time()
        logits = self.out_model(output4)
        # end = time.time()
        # print("out", (end-start)*1000)

    def forward(self, input, pos):

        layer = self.config_ours["n_layer"]
        
        res = self.emd_model(input, pos)
        for _ in range(layer):
            # start = time.time()
            q, k, v= self.qkv_model(res)
            ## cache processing
            # k = torch.concat([torch.rand(cache_num,, self.n_embd, device='cuda:0'), k])
            # q = torch.concat([torch.rand(cache_num,, self.n_embd, device='cuda:0'), q])
            # v = torch.concat([torch.rand(cache_num, self.n_embd, device='cuda:0'), v])            
            attn = self.attn_model(q, k, v)
            # attn = attn[cache_num:,:]
            ## attention post-processing
            proj_res = self.proj_model(res, attn)
            res = torch.add(res, proj_res)
            # end = time.time()
            # print(f"layer{i} : {(end-start)*1000}")
            # print(res)
        
        logits = self.out_model(res)
        ## post-processing of logits
        new_token = logits[-1].view(-1, 50257)
        # print(new_token.shape)
        probs = torch.nn.functional.softmax(new_token, dim = 1)
        new_token_idx = torch.argmax(probs, dim = 1)

        return new_token_idx.view(-1, 1)

if __name__ == "__main__":
    our_model = GPT_TRT()
    input = torch.tensor([15496,  2159,   318,  3666,  5181,   318,   437,  7777], dtype=torch.int32, device='cuda:0')
    # input = torch.tensor([15496], dtype=torch.int32, device='cuda:0')
    pos = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7], dtype=torch.int32, device='cuda:0')
    # pos = torch.tensor([0], dtype=torch.int32, device='cuda:0')
    start = time.time()
    for _ in range(32):
        output = our_model(input, pos)
    end = time.time()
    print("total :", (end - start)*1000)