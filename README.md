# ferminet

Google DeepMind FermiNet with Docker

## Usage

```shell
docker build -t ferminet:0.2-cuda12-cudnn9 .
docker run --gpus=all -it ferminet:0.0.2-cuda12-cudnn9
```

## Usage of ferminet

[README.md - ferminet](./ferminet/README.md)

## Test Environment

- System: Ubuntu 22.04 LTS (x86_64)
- GPU:  
  - NVIDIA GeForce RTX 4060
  - NVIDIA Driver 550.127.05
- Docker: 27.3.1, build ce12230
