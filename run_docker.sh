# Build in Docker because of compability issue
# Follow instructions in https://github.com/NVIDIA/FasterTransformer/blob/main/docs/gpt_guide.md
# Optimized code will be also located outside of docker
nvidia-docker run -ti --shm-size 5g -v `pwd`:/workspace nvcr.io/nvidia/pytorch:22.09-py3 bash