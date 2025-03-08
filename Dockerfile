FROM nvidia/cuda:12.3.2-devel-ubuntu22.04

LABEL maintainer="SJTUYoimiya"
LABEL org.opencontainers.image.licenses="Apache-2.0"
LABEL org.opencontainers.image.authors="SJTUYoimiya sjtuyoimiya@icloud.com"

# install dependencies
RUN apt update -y
RUN apt install -y git python3.11-venv cudnn9-cuda-12

# construct python virtual environment
WORKDIR /root
RUN python3.11 -m venv ferminet

# install ferminet
COPY . ferminet
RUN ferminet/bin/pip install -e ferminet/ferminet
RUN ferminet/bin/pip install ferminet/kfac-jax
RUN ferminet/bin/pip install -e ferminet/ferminet'[testing]'

RUN echo 'source /root/ferminet/bin/activate' >> /root/.bashrc

ENTRYPOINT [ "/bin/bash" ]
