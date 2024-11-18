# ferminet

Google DeepMind FermiNet with Docker

## Usage

```shell
docker build -t ferminet:0.0.2-cuda12-cudnn9 .
docker run --gpus=all -it ferminet:0.0.2-cuda12-cudnn9
```
