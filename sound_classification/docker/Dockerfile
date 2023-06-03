ARG PYTORCH="1.13.1"
ARG CUDA="11.6"
ARG CUDNN="8"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

ARG DEBIAN_FRONTEND=noninteractive

LABEL maintainer="@hirune924"

RUN apt-get update --fix-missing && apt-get install -y \
    git \
    libopencv-dev \
    python-openslide \
    unzip build-essential \
    && \
    apt-get install -y libopenmpi-dev && \
    apt-get clean && \
    apt-get autoremove && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

# Install python package.
ADD requirements.txt  /tmp/requirements.txt
RUN pip install --upgrade pip && \
    pip install -r /tmp/requirements.txt

# for pytorch-lightning==1.9
RUN pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117

# Language settings
ENV LANG C.UTF-8
ENV LANGUAGE en_US