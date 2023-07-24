# Copyright 2020-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import sys
# import torch
import numpy as np
import tritonclient.http as httpclient
from tritonclient.utils import *
import time

model_name = "encoding"
shape = [1, 4]

with httpclient.InferenceServerClient("localhost:8000") as client:
    input0_data = np.random.randint(0,12312, size=(4,32)).astype(np.int32)
    input1_data = np.array(range(4)).astype(np.int32)
    inputs = [
        httpclient.InferInput(
            "INPUT0", input0_data.shape, np_to_triton_dtype(input0_data.dtype)
        ),
        httpclient.InferInput(
            "INPUT1", input1_data.shape, np_to_triton_dtype(input1_data.dtype)
        ),
    ]

    inputs[0].set_data_from_numpy(input0_data)
    inputs[1].set_data_from_numpy(input1_data)

    outputs = [
        httpclient.InferRequestedOutput("OUTPUT0"),
    ]

    # acc = 0.0
    # for z in range(100):
    start = time.time()
        # for i in range(128):
        #     print(f"i is {i}")
    response = client.infer(model_name, inputs, request_id=str(1), outputs=outputs)
    result1 = response.get_response()
        # response = client.infer(model_name, inputs, request_id=str(1), outputs=outputs)
    # end = time.time()

    # print(f"time {end - start}")
    #     acc += end - start
    #     result2 = response.get_response()
    

    # acc = acc/100.0
    # print(f"avg {acc:.5f} sec")
    
    for i in range(127):
        output0_data = response.as_numpy("OUTPUT0").astype(np.int32)
        print(output0_data)
        
        inputs = [
            httpclient.InferInput(
                "INPUT0", output0_data.shape, np_to_triton_dtype(output0_data.dtype)
            ),
            httpclient.InferInput(
                "INPUT1", input1_data.shape, np_to_triton_dtype(input1_data.dtype)
            ),
        ]

        inputs[0].set_data_from_numpy(output0_data)
        inputs[1].set_data_from_numpy(input1_data)
        # start = time.time()
        response = client.infer(model_name, inputs, request_id=str(1), outputs=outputs)
        result1 = response.get_response()
    
    
    end = time.time()

    print(f"time {end - start}")

    print("PASS: pytorch")
    sys.exit(0)
