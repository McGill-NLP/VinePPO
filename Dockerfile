# Use the official PyTorch image as the base image
FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-devel


ENV TZ=America/New_York
ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda

# Update the package list and install necessary packages
RUN apt-get update && apt-get install -y --no-install-recommends --fix-missing \
    curl \
    unzip \
    software-properties-common \
    apt-transport-https ca-certificates gnupg lsb-release gnupg \
    make bash-completion tree vim less tmux nmap nano wget \
    build-essential libfreetype6-dev \
    pkg-config \
    graphviz \
    git \
    htop \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Remove the requirements.txt file
RUN rm requirements.txt


