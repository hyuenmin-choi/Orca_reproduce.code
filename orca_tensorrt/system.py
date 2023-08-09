from scheduler import *
from utils import *
from build import GPT_TRT
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import threading
import argparse
import numpy as np
import random
import time

class Server(threading.Thread):
    def __init__(self, model, batch_size, batching_delay, request_num):

        threading.Thread.__init__(self)

        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token

        self.scheduler = Scheduler(model, tokenizer, batch_size, batching_delay, request_num)

    def addRequest(self, request):
        self.scheduler.addRequest(request)
    
    def run(self):
        self.scheduler.run()


if __name__ == "__main__":

    model = GPT_TRT()

    batch = [4]
    request = [500, 700]

    for batch_size in batch:

        for request_num in request:
            print("----------------------------------------------------------")
            print(f"--                      batch size {batch_size}                    --")
            print(f"--                       req num {request_num}                     --")
            print("----------------------------------------------------------")

            myserver = Server(model, batch_size, 0.0001, request_num)
            
            request_list = []

            for i in range(request_num):
                new = Request(random.randint(32, 128), 64, i+1)
                request_list.append(new)
            
            # print(i)

            myserver.start()

            while(len(request_list) != 0):
                req = request_list.pop()
                myserver.addRequest(req)
                sec = np.random.poisson(lam=150)/1000
                time.sleep(sec)

            myserver.join()
        
