FROM nvidia/cuda:12.3.2-devel-ubuntu22.04

LABEL maintainer="SJTUYoimiya"
LABEL org.opencontainers.image.licenses="Apache-2.0"
LABEL org.opencontainers.image.authors="SJTUYoimiya sjtuyoimiya@icloud.com"

COPY . /root/

WORKDIR /root

# install dependencies
RUN sed -i 's|http://archive.ubuntu.com/ubuntu|https://mirror.sjtu.edu.cn/ubuntu|g' /etc/apt/sources.list && \
    apt update --allow-insecure-repositories -y && \
    apt install -y ca-certificates && \
    apt update -y && apt upgrade -y && \
    apt install -y zsh git wget curl nano build-essential zlib1g

SHELL [ "/bin/zsh", "-c" ]

# use zsh
RUN chsh -s /bin/zsh && \
    sh -c "$(wget -O- https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" && \
    git clone https://github.com/zsh-users/zsh-syntax-highlighting.git /root/.oh-my-zsh/custom/plugins/zsh-syntax-highlighting && \
    git clone https://github.com/zsh-users/zsh-autosuggestions.git /root/.oh-my-zsh/custom/plugins/zsh-autosuggestions  && \
    sed -i 's/plugins=(git)/plugins=(git zsh-syntax-highlighting zsh-autosuggestions git z vscode)/g' ~/.zshrc && \
    sed -i 's/ZSH_THEME="robbyrussell"/ZSH_THEME="bureau"/g' ~/.zshrc

# install cudnn9
RUN dpkg -i /root/cuda-keyring_1.1-1_all.deb && \
    apt update && \
    apt install -y cudnn9-cuda-12

# install miniconda
RUN mkdir /root/miniconda3 && \
    bash /root/miniconda.sh -b -u -p /root/miniconda3 && \
    rm -rf /root/miniconda.sh && \
    source /root/miniconda3/bin/activate && \
    conda init --all

ENV PATH="/root/miniconda3/bin:$PATH"

# install ferminet
RUN conda create -n ferminet python=3.11 -y && \
    conda run -n ferminet pip install -e /root/ferminet && \
    conda run -n ferminet pip install /root/kfac-jax && \
    conda run -n ferminet pip install -e /root/ferminet'[testing]' && \
    conda run -n base conda install numpy scipy matplotlib pandas openpyxl seaborn scikit-learn -y

ENTRYPOINT [ "/bin/zsh" ]
