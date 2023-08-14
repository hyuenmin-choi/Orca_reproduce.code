import torch
import time

# from gpt import GPT
from build import GPT_TRT
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import copy_model, generate_token
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from torch.profiler import profile, record_function, ProfilerActivity
import numpy as np

from queue import PriorityQueue, Queue

iter_time = 0.285 * 37 * 0.001

class Request:
    def __init__(self, input_ids, output_len, user_id):
        
        if isinstance(input_ids, int):
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
    def __init__(self, model, tokenizer, max_batch_size, req_num):
        self.model = model
        self.tokenizer = tokenizer
        self.pool = PriorityQueue()
        # self.running = PriorityQueue() 우리는 한개의 GPU사용 가정
        self.finish = PriorityQueue()

        # self.batching_delay = batching_delay
        self.max_batch_size = max_batch_size
        
        self.total_req = 0
        self.done = 0
        self.deadline = req_num
        self.latency = 0
        self.norm_latency = 0
        self.measure_start = 0
        self.endflag = 0
        self.iter = 0

        self.test = []

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
        # print("iter start", time.time())
        # Select from pool queue
        batch = []
        user_ids = []
        batch_size = 0
        current_time = time.time()
        max_len = 1
        # batching에 들어갈 input 확인
        # print("Waiting For Batching")
        self.batching_delay = 0.0001
        while batch_size < self.max_batch_size and time.time() - current_time < self.batching_delay:
        # while batch_size < self.max_batch_size:
            # print("Batching...")
            if self.pool.empty():
                continue
                # break # batching delay를 제거, 빈 경우 그냥 진행
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
        
        self.iter += 1
        
        # print("Batching Done Batch length:", str(max_len))
        input_ids = torch.empty(0, dtype=torch.int32, device = 'cuda:0')
        pos = torch.empty(0, dtype=torch.int32, device = 'cuda:0')
        cache_num = 0
        index = []

        # print("Padding Start")
        # batching 없이 하나로 뭉쳐서 보냄
        for i in range(batch_size):
            # max_len보다 작을 경우 차이만큼 패딩
            if batch[i].next_token != None:

                input_ids = torch.cat((input_ids, batch[i].next_token), dim=1)
                pos = torch.cat((pos, torch.tensor([batch[i].input_ids.shape[1]], dtype=torch.int32, device='cuda:0')), dim=0)
                cache_num = max(cache_num, (batch[i].input_ids.shape[1] + 1))

            else:
                input_ids = torch.cat((input_ids, batch[i].input_ids), dim=1)
                pos = torch.cat((pos, torch.arange(batch[i].input_ids.shape[1], dtype=torch.int32, device='cuda:0')), dim=0)
                cache_num = max(cache_num, batch[i].input_ids.shape[1])
            
            index.append(input_ids.shape[1]-1)
        
        input_ids = input_ids.view(-1)

        # print("Padding Done")

        # 모든 input 준비완료, model inference 시작
        # print("Inference Start")
        # print(input_ids)
        # print(pos)
        # print(cache_num)

        output = self.model.forward(input_ids, pos, cache_num, index)
        output = output.int()
        
        # print(output)
        # print("Inference Done")

        # request update
        for i, out in zip(range(batch_size), output.split(1)):
        # for i, out in zip(range(batch_size), output): # ONNX
            if batch[i].next_token != None:
                batch[i].input_ids = torch.cat((batch[i].input_ids, batch[i].next_token), dim=1)
            batch[i].next_token = out.cuda()
            # batch[i].next_token = torch.from_numpy(out).cuda() # ONNX
            # print(batch[i].next_token)
            if batch[i].output_length >= batch[i].max_length:
                self.finish.put(batch[i])
            else:
                self.pool.put(batch[i])
        
        # TODO response 처리를 어떻게 할 것인가?
        while not self.finish.empty():
            request = self.finish.get()
            
            response = torch.cat((request.input_ids, request.next_token), dim=1)
            response_time = time.time()
            # print("Output Has Returned")
            # print(response)
            # print(request.output_length)
            # print("res_time", response_time)
            # print("req_time", request.req_time)

            ################################ ONLY FOR TESTING ######################################
            # self.test.append(response_time - request.req_time - (iter_time * request.output_length))
            ########################################################################################

            self.norm_latency += (response_time - request.req_time - (iter_time * request.output_length))/request.output_length
            self.done += 1
            self.latency += (response_time - request.req_time - (iter_time * request.output_length))

            if(self.done == self.deadline):
                self.measure_end = response_time
                self.endflag = 1

        # print("Finish One Iteration")
        return
    
    def run(self):
        while(1):
            # print("runing")
            self.schedule()

            if(self.endflag):
                self.eval_metric()
                break
    
    def eval_metric(self):
        # print("evaluation metric")
        print(f"thorugh put : {(self.total_req / (self.measure_end - self.measure_start - self.iter * iter_time))} req/sec")
        print(f"normalized latency : {(self.norm_latency / self.total_req)*1000} ms")
        print(f"latency : {(self.latency / self.total_req)*1000} ms")

# testing scheduler
if __name__ == "__main__":
    model_ours = GPT_TRT()

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    scheduler = Scheduler(model_ours, tokenizer, 1, 1)

    dummy_input1 = tokenizer(["Hello World is"], return_tensors="pt", padding = "longest")["input_ids"].to("cuda:0").int()
    dummy_input2 = tokenizer(["My Cat is"], return_tensors="pt", padding = "longest")["input_ids"].to("cuda:0").int()

    dummy_req1 = Request(dummy_input1, 32, 1)
    dummy_req2 = Request(dummy_input2, 5, 2)


    # print(dummy_req1.input_ids.shape[1])

    scheduler.addRequest(dummy_req1)
    # scheduler.addRequest(dummy_req2)

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

    scheduler.schedule()
    scheduler.schedule()
    scheduler.schedule()
    scheduler.schedule()
    scheduler.schedule()
    scheduler.schedule()
    scheduler.schedule()
    scheduler.schedule()
    scheduler.schedule()
    scheduler.schedule()

    scheduler.schedule()
    scheduler.schedule()
    scheduler.schedule()
    scheduler.schedule()
    scheduler.schedule()
    scheduler.schedule()
    scheduler.schedule()
    scheduler.schedule()
    scheduler.schedule()
    scheduler.schedule()

    scheduler.schedule()
    scheduler.schedule()

    for laten in scheduler.test:
        print(laten*1000)
