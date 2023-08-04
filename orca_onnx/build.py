import torch
from torch.onnx import export
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from attn_proj import GPT_Projection
from attn import GPT_Attention
from emd_pos import GPT_Embedding
from out_gen import GPT_Output
from qkv_gen import GPT_QKVgen

# model config
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

# initialize models
emd_pos = GPT_Embedding(config_ours['vocab_size'], config_ours['n_embd'], config_ours['n_positions'])
qkv_gen = GPT_QKVgen(config_ours['n_embd'])
attn = GPT_Attention()
attn_proj = GPT_Projection(config_ours['n_embd'], config_ours['layer_norm_epsilon'])
out_gen = GPT_Output(config_ours['n_embd'], config_ours['layer_norm_epsilon'])

# export models to onnx tensorrt
dummy_input = (
    torch.tensor([15496,  2159,   318,  3666,  5181,   318], dtype=torch.int64, device='cuda:0'),
    torch.tensor([0, 1, 2, 0, 1, 2], dtype=torch.int64, device='cuda:0')
)

emd_model = export(
    emd_pos,
    dummy_input,
    "/workspace/experiment_env/emd_pos.onnx",
    input_names=['input_ids', 'pos'],
    output_names=['output'],
    dynamic_axes={
        "input_ids": {0: "total_token"},
        "pos": {0: "total_token"}
    },
    opset_version=16,
    # verbose=True
)

dummy_input = (
    torch.tensor([0, 1, 2, 3, 4, 4, 5, 6, 7, 8], dtype=torch.int64, device='cuda:0'),
)

qkv_model = export(
    qkv_gen,
    dummy_input,
    "/workspace/experiment_env/emd_pos.onnx",
    input_names=['embd'],
    output_names=['q', 'k', 'v'],
    dynamic_axes={
        "embd": {0: "total_token"},
        "q": {0: "total_token"},
        "k": {0: "total_token"},
        "v": {0: "total_token"}
    },
    opset_version=16,
    # verbose=True
)

