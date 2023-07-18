import torch
import time

# from gpt import GPT
from gptwiththread import GPThread
from gptadv import GPT
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import copy_model, generate_token
from transformers import GPT2Tokenizer, GPT2LMHeadModel

def main(argv=None):

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

    # model_ours = GPT(**config_ours)
    model_ours = GPT(**config_ours)
    model_ours.eval()

    copy_model(model_official, model_ours)

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    input = tokenizer(["hello everyone i am hmchoi hello everyone i am hmchoi hello everyone i am hmchoi hello everyone i am hmchoi hello everyone i am hmchoi hello everyone i am hmchoi hello everyone i am hmchoi hello everyone i am hmchoi hello everyone i am hmchoi hello everyone i am hmchoi hello everyone i am hmchoi hello everyone i am hmchoi hello everyone i am hmchoi hello everyone i am hmchoi hello everyone i am hmchoi hello everyone i am hmchoi hello everyone i am hmchoi hello everyone i am hmchoi", "wow wow wow", "whats going on"], return_tensors="pt", padding = "longest")["input_ids"]
    
    start = time.time()
    output = model_ours(input, [0, 1, 2])
    end = time.time()

    # print(output)
    print(output.shape)

    print(f"{end - start:.5f} sec")

    start = time.time()
    new_output = model_ours(output, [0,1,2])
    end = time.time()

    print(f"{end - start:.5f} sec")

    print(output)
    # print(output[None, ...])
    # model_ours(output[None, None, ...])
    # print(tokenizer.decode(output))


if __name__ == "__main__":
    main()