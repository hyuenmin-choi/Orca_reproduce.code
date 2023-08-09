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

iter_time = 0.285 * 37

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
                opt_shape=(512,),
                max_shape=(1024,), 
                dtype=torch.int32), 
            torch_tensorrt.Input( # pos
                min_shape=(1,),
                opt_shape=(512,),
                max_shape=(1024,), 
                dtype=torch.int32)],
            enabled_precisions = torch.float32, # Run with FP32
            workspace_size = 1 << 33,
            require_full_compilation = True
        )

        # torch.save(self.emd_model, "emb_model")
        # self.emd_model = torch.load("emb_model")

        input1 = torch.tensor([15496], dtype=torch.int32, device='cuda:0')
        pos1 = torch.tensor([0], dtype=torch.int32, device='cuda:0')
        input2 = torch.tensor([15496 * 1024], dtype=torch.int32, device='cuda:0')
        pos2 = torch.tensor([0 * 1024], dtype=torch.int32, device='cuda:0')

        # warm up
        for _ in range(30):
            self.emd_model(input1, pos1)
            self.emd_model(input2, pos2)

        self.qkv_model = torch_tensorrt.compile(qkv_gen, inputs = [
            torch_tensorrt.Input( # concated input
                min_shape=(1, self.config_ours['n_embd']),
                opt_shape=(512, self.config_ours['n_embd']),
                max_shape=(1024, self.config_ours['n_embd']), 
                dtype=torch.float32)],
            enabled_precisions = torch.float32, # Run with FP32
            workspace_size = 1 << 33,
            require_full_compilation = True
        )

        # torch.save(self.qkv_model, "qkv_model")
        # self.qkv_model = torch.load("qkv_model")

        input3 = torch.cuda.FloatTensor(1, self.config_ours['n_embd'])
        input4 = torch.cuda.FloatTensor(1024, self.config_ours['n_embd'])

        # warm up
        for _ in range(30):
            self.qkv_model(input3)
            self.qkv_model(input4)

        self.attn_model = torch_tensorrt.compile(attn, inputs = [
            torch_tensorrt.Input( # concated input
                min_shape=(1, self.config_ours['n_embd']),
                opt_shape=(512, self.config_ours['n_embd']),
                max_shape=(1024, self.config_ours['n_embd']), 
                dtype=torch.float32),
            torch_tensorrt.Input( # concated input
                min_shape=(1, self.config_ours['n_embd']),
                opt_shape=(512, self.config_ours['n_embd']),
                max_shape=(1024, self.config_ours['n_embd']), 
                dtype=torch.float32),
            torch_tensorrt.Input( # concated input
                min_shape=(1, self.config_ours['n_embd']),
                opt_shape=(512, self.config_ours['n_embd']),
                max_shape=(1024, self.config_ours['n_embd']), 
                dtype=torch.float32)],
            enabled_precisions = torch.float32, # Run with FP32
            workspace_size = 1 << 33,
            require_full_compilation = True
        )

        # torch.save(self.attn_model, "attn_model")
        # self.attn_model = torch.load("attn_model")


        q1 = torch.cuda.FloatTensor(1, self.config_ours['n_embd'])
        k1 = torch.cuda.FloatTensor(1, self.config_ours['n_embd'])
        v1 = torch.cuda.FloatTensor(1, self.config_ours['n_embd'])

        q2 = torch.cuda.FloatTensor(1024, self.config_ours['n_embd'])
        k2 = torch.cuda.FloatTensor(1024, self.config_ours['n_embd'])
        v2 = torch.cuda.FloatTensor(1024, self.config_ours['n_embd'])

        # warm up
        for i in range(30):
            self.attn_model(q1, k1, v1)
            self.attn_model(q2, k2, v2)

        # projection
        self.proj_model = torch_tensorrt.compile(attn_proj, inputs = [
            torch_tensorrt.Input( # concated input
                min_shape=(1, self.config_ours['n_embd']),
                opt_shape=(512, self.config_ours['n_embd']),
                max_shape=(1024, self.config_ours['n_embd']), 
                dtype=torch.float32),
            torch_tensorrt.Input( # concated input
                min_shape=(1, self.config_ours['n_embd']),
                opt_shape=(512, self.config_ours['n_embd']),
                max_shape=(1024, self.config_ours['n_embd']), 
                dtype=torch.float32)],
            enabled_precisions = torch.float32, # Run with FP32
            workspace_size = 1 << 33,
            require_full_compilation = True
        )

        # torch.save(self.proj_model, "proj_model")
        # self.proj_model = torch.load("proj_model")

        for _ in range(30):
            self.proj_model(input3, input3)
            self.proj_model(input4, input4)

        # logits
        self.out_model = torch_tensorrt.compile(out_gen, inputs = [
            torch_tensorrt.Input( # concated input
                min_shape=(1, self.config_ours['n_embd']),
                opt_shape=(512, self.config_ours['n_embd']),
                max_shape=(1024, self.config_ours['n_embd']), 
                dtype=torch.float32)],
            enabled_precisions = torch.float32, # Run with FP32
            workspace_size = 1 << 33,
            require_full_compilation = True
        )
        
        # torch.save(self.out_model, "out_model")
        # self.out_model = torch.load("out_model")

        # warm up
        for _ in range(30):
            self.out_model(input3)
            self.out_model(input4)


    def forward(self, input, pos, cache_num, index):

        layer = self.config_ours["n_layer"]
        
        res = self.emd_model(input, pos)
        for _ in range(layer):
            # start = time.time()
            q, k, v = self.qkv_model(res)
            ## cache processing
            ## cache num : max cache + input length
            q = torch.cuda.FloatTensor(cache_num, self.config_ours['n_embd'])
            k = torch.cuda.FloatTensor(cache_num, self.config_ours['n_embd'])
            v = torch.cuda.FloatTensor(cache_num, self.config_ours['n_embd'])
            attn = self.attn_model(q, k, v)
            # attn = attn[cache_num:,:]
            ## attention post-processing
            proj_res = self.proj_model(res, res)
            res = torch.add(res, proj_res)
            # end = time.time()
            # print(f"layer{i} : {(end-start)*1000}")
            # print(res)
        
        logits = self.out_model(res)
        ## post-processing of logits
        new_token = logits[index].view(-1, 50257)
        # print(new_token.shape)
        probs = torch.nn.functional.softmax(new_token, dim = 1)
        new_token_idx = torch.argmax(probs, dim = 1)

        return new_token_idx.view(-1, 1)

if __name__ == "__main__":
    model = GPT_TRT()
    input1 = torch.tensor([15496,  2159,   318,  3666,  5181,   318,   437,  7777], dtype=torch.int32, device='cuda:0')
    input2 = torch.tensor([15496], dtype=torch.int32, device='cuda:0')
    pos1 = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7], dtype=torch.int32, device='cuda:0')
    pos2 = torch.tensor([0], dtype=torch.int32, device='cuda:0')
    index1 = torch.tensor([7], dtype=torch.int32, device='cuda:0')
    index2 = torch.tensor([0], dtype=torch.int32, device='cuda:0')
    start = time.time()
    output = model.forward(input1, pos1, 8, index1)
    for i in range(32):
        output = model.forward(input2, pos2, 9 + i, index2)
    end = time.time()
    # global iter_time
    print("total :", (end - start)*1000 - iter_time * 32)