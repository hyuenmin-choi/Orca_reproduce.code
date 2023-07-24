import torch
import time

# from gpt import GPT
from gptadv import GPT
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import copy_model, generate_token
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from torch.profiler import profile, record_function, ProfilerActivity

from queue import PriorityQueue, Queue

class Request:
    def __init__(self, input_ids, req_time, output_len, user_id):
        self.input_ids = input_ids # tensor of token_id
        self.req_time = req_time
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

    # pool에 request add
    def addRequest(self, request):
        self.pool.put(request)
        print("Request Adding Complete")
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
        print("Waiting For Batching")
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
        
        print("Batching Done Batch length:", str(max_len))
        input_ids = torch.empty(0, max_len, dtype=torch.int).cuda()

        print("Padding Start")
        # model input 패딩하여 만들기
        for i in range(batch_size):
            length = batch[i].input_ids.shape[1]
            # max_len보다 작을 경우 차이만큼 패딩
            if batch[i].next_token != None:
                if length < max_len:
                    input = torch.cat((torch.tensor([[self.tokenizer.eos_token_id for _ in range(max_len - length)]]).cuda(), batch[i].next_token), dim=1)
                    print(input)
                else:
                    input = batch[i].next_token
                    print(input)
            else:
                if length < max_len:
                    input = torch.cat((torch.tensor([[self.tokenizer.eos_token_id for _ in range(max_len - length)]]).cuda(), batch[i].input_ids), dim=1)
                    print(input)
                else:
                    input = batch[i].input_ids
                    print(input)
            
            input_ids = torch.cat((input_ids, input), dim=0)
        print("Padding Done")

        # 모든 input 준비완료, model inference 시작
        print("Inference Start")
        print(input_ids)
        print(user_ids)
        output = self.model(input_ids, user_ids)
        print(output)
        print("Inference Done")

        # request update
        for i, out in zip(range(batch_size), output.split(1)):
            if batch[i].next_token != None:
                batch[i].input_ids = torch.cat((batch[i].input_ids, batch[i].next_token), dim=1)
            batch[i].next_token = out.cuda()
            print(batch[i].next_token)
            if batch[i].output_length >= batch[i].max_length:
                self.finish.put(batch[i])
            else:
                self.pool.put(batch[i])
        
        # TODO response 처리를 어떻게 할 것인가?
        while True:
            if self.finish.empty():
                print("Finished Request Empty")
                break
            request = self.finish.get()
            response = torch.cat((request.input_ids, request.next_token), dim=1)

            print("Output Has Returned")
            print(response)

        print("Finish One Iteration")
        return

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

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # print(tokenizer.eos_token)

    scheduler = Scheduler(model_ours, tokenizer, 2, 1)

    dummy_input1 = tokenizer(["Hello World"], return_tensors="pt", padding = "longest")["input_ids"].to("cuda:0")
    dummy_input2 = tokenizer(["My Cat"], return_tensors="pt", padding = "longest")["input_ids"].to("cuda:0")

    dummy_req1 = Request(dummy_input1, time.time(), 10, 0)
    dummy_req2 = Request(dummy_input2, time.time(), 5, 1)

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
