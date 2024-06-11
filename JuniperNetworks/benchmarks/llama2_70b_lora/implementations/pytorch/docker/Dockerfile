ARG FROM_IMAGE_NAME=nvcr.io/nvidia/pytorch:24.01-py3
FROM ${FROM_IMAGE_NAME}

RUN apt-get update && apt-get install -y \
    curl \
    gnupg \
    && rm -rf /var/lib/apt/lists/*

RUN curl https://developer.download.nvidia.com/hpc-sdk/ubuntu/DEB-GPG-KEY-NVIDIA-HPC-SDK | gpg --dearmor -o /usr/share/keyrings/nvidia-hpcsdk-archive-keyring.gpg

RUN echo 'deb [signed-by=/usr/share/keyrings/nvidia-hpcsdk-archive-keyring.gpg] https://developer.download.nvidia.com/hpc-sdk/ubuntu/amd64 /' | tee /etc/apt/sources.list.d/nvhpc.list

RUN apt-get update
RUN apt-get install -y nvhpc-23-11

WORKDIR /workspace/ft-llm

ADD configs /workspace/ft-llm
ADD scripts /workspace/ft-llm
ADD requirements.txt /workspace/ft-llm

RUN DS_BUILD_FUSED_ADAM=1 pip install deepspeed==0.13.2
RUN pip install -r requirements.txt
RUN pip install flash-attn==2.4.1 --no-build-isolation