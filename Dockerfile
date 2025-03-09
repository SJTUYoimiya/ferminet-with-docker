FROM ubuntu:jammy

LABEL maintainer="SJTUYoimiya"
LABEL org.opencontainers.image.licenses="Apache-2.0"
LABEL org.opencontainers.image.authors="SJTUYoimiya sjtuyoimiya@icloud.com"

# install dependencies
RUN apt update -y && apt install -y git python3.11-venv

# construct python virtual environment
WORKDIR /root
RUN python3.11 -m venv ferminet_env

# install ferminet
COPY . ferminet
RUN ferminet_env/bin/pip install -e ferminet/ferminet
RUN ferminet_env/bin/pip install ferminet/kfac-jax
RUN ferminet_env/bin/pip install -e ferminet/ferminet'[testing]'

RUN echo 'source /root/ferminet/bin/activate' >> /root/.bashrc

ENTRYPOINT [ "/bin/bash" ]
