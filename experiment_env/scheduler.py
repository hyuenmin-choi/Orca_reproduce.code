import torch
import time

# from gpt import GPT
from gptadv_RT import GPT
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import copy_model, generate_token
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from torch.profiler import profile, record_function, ProfilerActivity

from queue import PriorityQueue, Queue

import torch_tensorrt
from torch.onnx import export
import onnx

class Request:
    def __init__(self, input_ids, output_len, user_id):
        
        if (isinstance(input_ids, int)):
            # print("integer")
            self.input_ids = torch.randint(1, 50255, (1, input_ids), dtype=torch.int32, device="cuda:0")
            # print(self.input_ids)
        
        else:
            self.input_ids = input_ids
        
        self.req_time = 0
        self.max_length = output_len
        self.output_length = 0
        self.user_id = user_id
        self.next_token = None
    
    def __lt__(self, other):
        return self.req_time < other.req_time


class Scheduler:
    def __init__(self, model, tokenizer, max_batch_size, batching_delay):
        self.model = model
        self.tokenizer = tokenizer
        self.pool = PriorityQueue()
        # self.running = PriorityQueue() 우리는 한개의 GPU사용 가정
        self.finish = PriorityQueue()

        self.batching_delay = batching_delay
        self.max_batch_size = max_batch_size
        
        self.total_req = 0
        self.done = 0
        self.latency = 0
        self.norm_latency = 0
        self.measure_start = 0
        self.endflag = 0

    # pool에 request add
    def addRequest(self, request):
        request.req_time = time.time()
        if self.total_req == 0:
            self.measure_start = request.req_time

        self.pool.put(request)
        # print("Request Adding Complete")
        self.total_req += 1
        
        return

    # one iteration
    def schedule(self):

        # Select from pool queue
        batch = []
        user_ids = []
        batch_size = 0
        current_time = time.time()
        max_len = 1
        # batching에 들어갈 input 확인
        # print("Waiting For Batching")
        while batch_size < self.max_batch_size and time.time() - current_time < self.batching_delay:
            # print("Batching...")
            if self.pool.empty():
                continue
            request = self.pool.get()

            # 처음 들어오는 token
            if request.next_token == None:
                max_len = max(max_len, request.input_ids.shape[1])
                batch.append(request)
                user_ids.append(request.user_id)
            # 처음이 아닌 token
            else:
                batch.append(request)
                user_ids.append(request.user_id)
            
            batch_size += 1
            request.output_length += 1

        # timer 초과
        if batch_size == 0:
            return
        
        # print("Batching Done Batch length:", str(max_len))
        input_ids = torch.empty(0, dtype=torch.int64, device = 'cuda:0')
        pos = torch.empty(0, dtype=torch.int64, device = 'cuda:0')
        info = torch.empty(0, dtype=torch.int64, device = 'cuda:0')
        length = torch.empty(0, dtype=torch.int64, device = 'cuda:0')

        print("Padding Start")
        # batching 없이 하나로 뭉쳐서 보냄
        for i in range(batch_size):
            # max_len보다 작을 경우 차이만큼 패딩
            if batch[i].next_token != None:

                input_ids = torch.cat((input_ids, batch[i].next_token), dim=1)
                pos = torch.cat((pos, torch.tensor([batch[i].input_ids.shape[1]], device='cuda:0')), dim=0)
                info = torch.cat((info, torch.tensor([batch[i].input_ids.shape[1]], device='cuda:0')), dim=0)
                length = torch.cat([length, torch.tensor([pos.shape[0]], device='cuda:0')])
            
                
            else:
                
                input_ids = torch.cat((input_ids, batch[i].input_ids), dim=1)
                pos = torch.cat((pos, torch.arange(batch[i].input_ids.shape[1], device='cuda:0')), dim=0)
                info = torch.cat((info, torch.tensor([0], device='cuda:0')), dim=0)
                length = torch.cat([length, torch.tensor([pos.shape[0]], device='cuda:0')])
                
            
        input_ids = input_ids.squeeze()
        
        print("Padding Done")

        # 모든 input 준비완료, model inference 시작
        print("Inference Start")
        print(input_ids)

        # with profile(with_stack=True, record_shapes=True) as prof:
        #     with record_function("model_inference"):
        output = self.model(input_ids, info, pos, length)
        # prof.export_chrome_trace("trace.json")

        # print(output)
        # print("Inference Done")

        # request update
        for i, out in zip(range(batch_size), output.split(1)):
            if batch[i].next_token != None:
                batch[i].input_ids = torch.cat((batch[i].input_ids, batch[i].next_token), dim=1)
            batch[i].next_token = out.cuda()
            # print(batch[i].next_token)
            if batch[i].output_length >= batch[i].max_length:
                self.finish.put(batch[i])
            else:
                self.pool.put(batch[i])
        
        # TODO response 처리를 어떻게 할 것인가?
        while True:
            if self.finish.empty():
                # print("Finished Request Empty")
                break
            request = self.finish.get()
            
            response = torch.cat((request.input_ids, request.next_token), dim=1)
            response_time = time.time()
            # print("Output Has Returned")
            # print(response)
            # print(f"response time {response_time - request.req_time}")
            self.norm_latency += (response_time - request.req_time)/request.output_length
            self.done += 1
            self.latency += (response_time - request.req_time)

            if(self.done == self.total_req):
                self.measure_end = response_time
                self.endflag = 1

        # print("Finish One Iteration")
        return
    
    def run(self):
        while(1):
            print("runing")
            self.schedule()

            if(self.endflag):
                self.eval_metric()
                break
    
    def eval_metric(self):
        print("evaluation metric")
        print(f"thorugh put : {(self.total_req / (self.measure_end - self.measure_start))} req/sec")
        print(f"normalized latency : {(self.norm_latency / self.total_req)*1000} ms")
        print(f"latency : {(self.latency / self.total_req)*1000} ms")

# testing scheduler
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

    copy_model(model_official, model_ours)

    with torch_tensorrt.logging.debug():

        trt_model = torch_tensorrt.compile(model_ours, inputs = [
        torch_tensorrt.Input( # concated input
            min_shape=(1,),
            opt_shape=(128,),
            max_shape=(512,), 
            dtype=torch.int32), 
        torch_tensorrt.Input( # info
            min_shape=(1,),
            opt_shape=(2,),
            max_shape=(2,), 
            dtype=torch.int32),
        torch_tensorrt.Input( # pos
            min_shape=(1,),
            opt_shape=(128,),
            max_shape=(512,), 
            dtype=torch.int32),
        torch_tensorrt.Input( # length
            min_shape=(1,),
            opt_shape=(2,),
            max_shape=(2,),
            dtype=torch.int32),
            ],
        enabled_precisions = torch.float32, # Run with FP32
        workspace_size = 1 << 33,
        require_full_compilation = True
    )

    model_ours = trt_model

    # dummy_input = [
    #     torch.randint(768, size=[1,], dtype=torch.int32).cuda(),
    #     torch.ones([1,], dtype=torch.int32).cuda(),
    #     torch.ones([1,], dtype=torch.int32).cuda(),
    #     torch.ones([1,], dtype=torch.int32).cuda()
    # ]

    # onnx_model = export(
    #     model_ours,
    #     dummy_input,
    #     "/workspace/experiment_env/gptadv.onnx",
    #     input_names=['input_ids', 'info', 'pos', 'length'],
    #     output_names=['output_ids'],
    #     dynamic_axes={
    #         "input_ids": {0: "total_token"},
    #         "info": {0: "batch"},
    #         "pos": {0: "total_token"},
    #         "legnth" : {0: "batch"},
    #         "output_ids": {0: "batch", 1: "sequence"}
    #     }
    # )

    # import onnxruntime
    # ort_session = onnxruntime.InferenceSession("/workspace/experiment_env/gptadv.onnx")

    # rt_model = torch2trt(model_ours, dummy_input, use_onnx=True)
    # model_ours = rt_model

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # print(tokenizer.eos_token)

    scheduler = Scheduler(model_ours, tokenizer, 2, 1)

    dummy_input1 = tokenizer(["Hello World my name"], return_tensors="pt", padding = "longest")["input_ids"].to("cuda:0")
    dummy_input2 = tokenizer(["My Cat"], return_tensors="pt", padding = "longest")["input_ids"].to("cuda:0")

    dummy_req1 = Request(dummy_input1, 10, 0)
    dummy_req2 = Request(dummy_input2, 5, 1)

    # print(dummy_req1.input_ids.shape[1])

    scheduler.addRequest(dummy_req1)
    scheduler.addRequest(dummy_req2)

    scheduler.schedule()
    scheduler.schedule()
    scheduler.schedule()
    scheduler.schedule()
    scheduler.schedule() #5 request2 should be end
    scheduler.schedule()
    scheduler.schedule()
    scheduler.schedule()
    scheduler.schedule()
    scheduler.schedule() #10 request1 should be end
