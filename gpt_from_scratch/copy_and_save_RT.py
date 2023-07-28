import torch
import time

# from gpt import GPT
from gptadv import GPT
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import copy_model, generate_token
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from torch.profiler import profile, record_function, ProfilerActivity
import torch_tensorrt

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

    model_ours = GPT(**config_ours)

    model_ours.eval()

    copy_model(model_official, model_ours)

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    input = tokenizer(["""hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello
                        hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello
                        hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello
                        hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello
                        """,
                        """hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello
                        hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello
                        hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello
                        hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello
                        """,
                        """hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello
                        hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello
                        hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello
                        hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello
                        """,
                        """hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello
                        hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello
                        hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello
                        hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello
                        """,
                        """hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello
                        hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello
                        hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello
                        hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello
                        """,
                        """hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello
                        hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello
                        hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello
                        hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello
                        """,
                        """hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello
                        hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello
                        hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello
                        hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello
                        """,
                        """hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello
                        hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello
                        hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello
                        hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello
                        """,
                    #    "hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello",
                    #    "hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello",
                    #    "hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello",
                    #    "hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello",
                    #    "hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello",
                    #    "hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello",
                    #    "hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello",
                    #    "hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello",
                    #    "hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello",
                    #    "hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello",
                    #    "hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello",
                    #    "hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello",
                    #    "hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello",
                    #    "hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello",
                    #    "hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello",
                       ]
                       , return_tensors="pt", padding = "longest")["input_ids"].to("cuda:0")
    total_avg = 0

    # torch-script 사용
    traced_model = torch.jit.script(model_ours)
    model_ours = traced_model

    # torch-tensorrt 사용
    trt_model = torch_tensorrt.compile(model_ours, inputs = [torch_tensorrt.Input(
            min_shape=(1, 1),
            opt_shape=(16, 32),
            max_shape=(32, 32), 
            dtype=torch.float32), 
            torch_tensorrt.Input(
            min_shape=(1,),
            opt_shape=(16,),
            max_shape=(32,), 
            dtype=torch.int32)],
        enabled_precisions = torch.float32, # Run with FP32
        workspace_size = 1 << 33
    )

    model_ours = trt_model




    for z in range(100):
        # print(z)
        total_start = time.time()
        start = time.time()
        x = len(input)*z
        # print(x)
        output = model_ours(input, torch.tensor([x+i for i in range(len(input))], dtype=torch.int32))
        # output = model_ours(input)
        end = time.time()

        # print(output)
        # print(output.shape)

        # print(f"{end - start:.5f} sec")
        # acc = 0
        for i in range(32):
            new_output = model_ours(output, torch.tensor([x+i for i in range(len(input))], dtype=torch.int32))
        total_end = time.time()

        total_avg += (total_end - total_start)
        print(f"total {total_end - total_start:.5f} sec")

    # print(f"total {total_avg/100:.5f} sec")

    #     start = time.time()
    # with profile(with_stack=True, record_shapes=True) as prof:
    #     with record_function("model_inference"):
    #         output = model_ours(input, [5000, 5001, 5002, 5003, 5004, 5005, 5006, 5007])
         
    #         for i in range(32):
    #             new_output = model_ours(output, [5000, 5001, 5002, 5003, 5004, 5005, 5006, 5007])
    
    # print(prof.key_averages(group_by_stack_n=5).table(sort_by="cpu_time_total", row_limit=20))
    # print(prof.key_averages(group_by_stack_n=5).table(sort_by="cuda_time_total", row_limit=20))
    # print(prof.key_averages(group_by_stack_n=5).table(sort_by="self_cpu_time_total", row_limit=20))
    # print(prof.key_averages(group_by_stack_n=5).table(sort_by="self_cuda_time_total", row_limit=20))

    # prof.export_chrome_trace("trace2.json")
    #     # output = model_ours(input)
    #     end = time.time()
    #     acc += (end-start)
    # total_end = time.time()
    # print(f"avg engine {acc/100.0} z {z}")
    # print(f"total {total_end - total_start:.5f} sec")



    # print(output[None, ...])
    # model_ours(output[None, None, ...])
    # print(tokenizer.decode(output))


if __name__ == "__main__":
    main()