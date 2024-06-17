ARG CUDA_VERSION=12.1.0
ARG from=nvidia/cuda:${CUDA_VERSION}-cudnn8-devel-ubuntu20.04

FROM ${from} as base

RUN <<EOF
apt update -y && apt upgrade -y && apt install -y --no-install-recommends  \
    git \
    git-lfs \
    python3 \
    python3-pip \
    python3-dev \
    wget \
    vim \
&& rm -rf /var/lib/apt/lists/*
EOF

RUN ln -s /usr/bin/python3 /usr/bin/python

RUN git lfs install

FROM base as dev

WORKDIR /

RUN mkdir -p /data/shared/Qwen

WORKDIR /data/shared/Qwen/

FROM dev as bundle_req
RUN pip install --no-cache-dir networkx==3.1
RUN pip3 install --no-cache-dir torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121
RUN pip3 install --no-cache-dir transformers==4.40.2 accelerate tiktoken einops scipy
    
FROM bundle_req as bundle_finetune
ARG BUNDLE_FINETUNE=true

RUN <<EOF
if [ "$BUNDLE_FINETUNE" = "true" ]; then
    cd /data/shared/Qwen

    # Full-finetune / LoRA.
    pip3 install --no-cache-dir "deepspeed==0.14.2" "peft==0.11.1"

    # Q-LoRA.
    apt update -y && DEBIAN_FRONTEND=noninteractive apt install -y --no-install-recommends \
        libopenmpi-dev openmpi-bin \
        && rm -rf /var/lib/apt/lists/*
    pip3 install --no-cache-dir "optimum==1.20.0" "auto-gptq==0.7.1" "autoawq==0.2.5" mpi4py
fi
EOF

FROM bundle_finetune as bundle_vllm
ARG BUNDLE_VLLM=true

RUN <<EOF
if [ "$BUNDLE_VLLM" = "true" ]; then
    cd /data/shared/Qwen

    pip3 install --no-cache-dir vllm==0.4.3 "fschat[model_worker,webui]==0.2.36"
fi
EOF

FROM bundle_vllm as bundle_flash_attention
ARG BUNDLE_FLASH_ATTENTION=true

RUN <<EOF 
if [ "$BUNDLE_FLASH_ATTENTION" = "true" ]; then
    pip3 install --no-cache-dir flash-attn==2.5.8 --no-build-isolation
fi
EOF

FROM bundle_flash_attention as final

COPY ../examples/sft/* ./
COPY ../examples/demo/* ./

EXPOSE 80
