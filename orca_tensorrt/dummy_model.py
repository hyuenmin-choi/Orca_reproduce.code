import torch
import torch.nn as nn
from torch.nn import functional as F
import torch_tensorrt
import time

class Dummy(nn.Module):
    def __init__(self):
        super().__init__()
        return

    # @torch.no_grad()
    def forward(self, x, y):
        z = torch.add(x, y)
        return z
    
if __name__ == "__main__":
    model = Dummy()
    model.eval()
    model.zero_grad(True)
    trt_model = torch_tensorrt.compile(model, inputs = [
        torch_tensorrt.Input( # concated input
            min_shape=(1,),
            opt_shape=(512,),
            max_shape=(1024,), 
            dtype=torch.int32),
        torch_tensorrt.Input( # concated input
            min_shape=(1,),
            opt_shape=(512,),
            max_shape=(1024,), 
            dtype=torch.int32)],
        enabled_precisions = torch.float32, # Run with FP32
        workspace_size = 1 << 33,
        require_full_compilation = True
    )

    input1 = torch.tensor([15496 * 32], dtype=torch.int32, device='cuda:0')
    input2 = torch.tensor([15496], dtype=torch.int32, device='cuda:0')
    pos1 = torch.tensor([0 * 32], dtype=torch.int32, device='cuda:0')
    pos2 = torch.tensor([0], dtype=torch.int32, device='cuda:0')

    # warm up
    for _ in range(100):
        trt_model(input1, pos1)
        # trt_model(input2, pos2)
    
    record = []

    for i in range(100000):
        start = time.time()
        output = trt_model(input1, pos1)
        end = time.time()
        record.append((end - start)*1000)
        # print("total", i ,":", (end - start)*1000)

    print("total avg : ", sum(record)/len(record))