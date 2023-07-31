from scheduler import *
from utils import *
from gptadv import GPT
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import threading
import argparse
import numpy as np
import random
import time

class Server(threading.Thread):
    def __init__(self, batch_size, batching_delay):

        threading.Thread.__init__(self)

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

        for param in model_ours.parameters():
            param.grad = None

        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token

        copy_model(model_official, model_ours)

        self.scheduler = Scheduler(model_ours, tokenizer, batch_size, batching_delay)

    def addRequest(self, request):
        self.scheduler.addRequest(request)
    
    def run(self):
        self.scheduler.run()


if __name__ == "__main__":

    myserver = Server(32, 0.1)
    print("here")

    #TODO : request manager => tokenizing 이후에 user id 부여
    #TODO : response 제대로 처리하기
    #TODO : request random generation code
    #TODO : request-response time 측정, thorghput 측정 방법 개발
    request_num = 2000

    request_list = []
    # input_len = np.random.random_integers(32, 512, request_num)
    # output_len = np.random.random_integers(1, 128, request_num)

    for i in range(request_num):
        new = Request(random.randint(32, 512 + 1), random.randint(32, 128 + 1), i+1)
        request_list.append(new)
    
    # print(i)

    myserver.start()

    while(len(request_list) != 0):
        req = request_list.pop()
        myserver.addRequest(req)
        sec = np.random.poisson(lam=50)/1000
        time.sleep(sec)


    
