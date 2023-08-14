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
import pandas as pd

class Server(threading.Thread):
    def __init__(self, model, batch_size, request_num):

        threading.Thread.__init__(self)

        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token

        self.scheduler = Scheduler(model, tokenizer, batch_size, request_num)

    def addRequest(self, request):
        self.scheduler.addRequest(request)
    
    def run(self):
        self.scheduler.run()


if __name__ == "__main__":

    # load dataset
    df_NQ = pd.read_csv('/workspace/dataset/NQ_data.csv')

    # get first 1000 rows that have non zero short answer length
    df_NQ = df_NQ.loc[df_NQ['short answer length'] > 0].iloc[:1000]

    # divide question, long, short
    df_question = df_NQ.loc[:,['question length']]
    df_long = df_NQ.loc[:,['long answer length']]
    df_short = df_NQ.loc[:,['short answer length']]

    # print(df_question.iloc[0, 0].item())

    # load and warm up model
    model = GPT_TRT()

    req_rate = [500 / 1000, 300 / 1000, 100 / 1000]
    batch = [1] # TODO make until 64 batches
    request = [1000] # fix to 1000

    for sec in req_rate:
        for batch_size in batch:
            for request_num in request:
                print("----------------------------------------------------------")
                print(f"--                      batch size {batch_size}                    --")
                print(f"--                      req num {request_num}                    --")
                print(f"--                      req rate {sec * 1000}                    --")
                print("----------------------------------------------------------")

                myserver = Server(model, batch_size, request_num)
                
                request_list = []

                for i in range(request_num):
                    # new = Request(random.randint(32, 128), 64, i+1)
                    new = Request(df_question.iloc[i, 0].item(), df_long.iloc[i, 0].item(), i+1)
                    request_list.append(new)

                myserver.start()

                while(len(request_list) != 0):
                    req = request_list.pop()
                    myserver.addRequest(req)
                    # sec = np.random.poisson(lam=150)/1000
                    time.sleep(sec)

                myserver.join()
        
