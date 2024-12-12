# [Google DeepMind FermiNet](https://github.com/google-deepmind/ferminet) 在 Docker 中的部署实现

由于 DeepMind 在优化器库 [`kfac_jax`](https://github.com/google-deepmind/kfac-jax) 库在 2024/10/25 10:34:33 的 commit `5a97d14a4429310caef1add6898122a66d87ee17` 中添加了 `sharding` 参数，导致了库版本冲突的问题

```git log
commit 5a97d14a4429310caef1add6898122a66d87ee17
Author: Yash Katariya <yashkatariya@google.com>
Date:   Fri Oct 25 10:34:33 2024 -0700

    Add sharding rules to some more primitives so that backward pass of minformer passes. There are a couple of changes here:
    
    * Handled transpose of `dot_general` correctly with shardings
    * Handled transpose of `reduce_sum` correctly with shardings
    * `ShapedArray.to_tangent_aval` now sets the sharding of the tangent (not handling unreduced yet).
    * `ConcreteArray.aval` correctly sets the sharding which is extracted from the `val` attribute.
    * (Paired with Dougal!) Added sharding rule for `reshape_p` only when singleton dims are added/removed.
    * Added sharding rule for `select_n_p` because it gets called during `jax.grad` of minformer.
    * Added `sharding` attribute to `broadcast_in_dim` because we need to provide the correct sharding to it during `full` and transpose of `reduce_sum`.
    
    PiperOrigin-RevId: 689837320

 kfac_jax/_src/tag_graph_matcher.py | 1 +
 1 file changed, 1 insertion(+)
```

本仓库是在 DeepMind FermiNet 的基础上，将 `kfac_jax` 库的版本降级到此 commit 之前的版本，以解决库版本冲突的问题，同时将 FermiNet 部署在 Docker 中，以隔离运行环境.

> ⚠️ **注意**：在安装 `ferminet` 前，请务必检查 [`setup.py`](./ferminet/setup.py) 文件的 `REQUIRED_PACKAGES` 字段中 `jax` 版本是否与系统环境匹配

## 使用方式

在配置前，请确保已经安装了 Docker 和 NVIDIA Container Toolkit，安装方式请参考官方教程或[本人 blog 记录](https://sjtuyoimiya.github.io/tech/docker-engine-部署/)

将本仓库克隆到本地后，在此仓库目录路径下运行以下命令构建 Docker 镜像

```shell
docker build -t ferminet:0.2-cuda12-cudnn9 .
docker run --gpus=all -it ferminet:0.0.2-cuda12-cudnn9
```

### Dockerfile 说明

- 使用 `nvidia/cuda:12.3.2-devel-ubuntu22.04` 作为基础镜像
- 安装基本生产环境：zsh, git, wget, e.g.
- 安装 cuDNN 9 加速库
- 安装 miniconda3 作为 Python 环境管理工具，并配置 ferminet 环境

### ferminet 使用方式

请参考原仓库说明：[README.md - ferminet](./ferminet/README.md)

## 测试环境

### 系统信息

- CPU: Intel i5-14600KF
- GPU: NVIDIA RTX 4070 Ti SUPER 16GB

  - NVIDIA 驱动版本: 550.127.08
  
- RAM: 32GB DDR5 6400MHz
- OS: Ubuntu 22.04 LTS

- Docker Engine: 27.3.1, build ce12230

```shell
Client: Docker Engine - Community
 Version:           27.3.1
 API version:       1.47
 Go version:        go1.22.7
 Git commit:        ce12230
 Built:             Fri Sep 20 11:41:00 2024
 OS/Arch:           linux/amd64
 Context:           default

Server: Docker Engine - Community
 Engine:
  Version:          27.3.1
  API version:      1.47 (minimum version 1.24)
  Go version:       go1.22.7
  Git commit:       41ca978
  Built:            Fri Sep 20 11:41:00 2024
  OS/Arch:          linux/amd64
  Experimental:     false
 containerd:
  Version:          1.7.23
  GitCommit:        57f17b0a6295a39009d861b89e3b3b87b005ca27
 runc:
  Version:          1.1.14
  GitCommit:        v1.1.14-0-g2c9f560
 docker-init:
  Version:          0.19.0
  GitCommit:        de40ad0
 rootlesskit:
  Version:          2.3.1
  ApiVersion:       1.1.1
  NetworkDriver:    slirp4netns
  PortDriver:       builtin
  StateDir:         /run/user/1000/dockerd-rootless
 slirp4netns:
  Version:          1.0.1
  GitCommit:        6a7b16babc95b6a3056b33fb45b74a6f62262dd4
```

### Docker 容器环境

- 上游镜像: `nvidia/cuda:12.3.2-devel-ubuntu22.04`
- CUDA: 12.3.107

  - cuDNN: 9.5.1

- Python:

  - Conda: 24.9.2
  - Python(ferminet): 3.11.10
  - JAX: `cuda12_local` + 0.4.34

```shell
# packages in environment at /root/miniconda3/envs/ferminet:
#
# Name                    Version                   Build  Channel
_libgcc_mutex             0.1                        main  
_openmp_mutex             5.1                       1_gnu  
absl-py                   2.1.0                    pypi_0    pypi
astroid                   3.3.5                    pypi_0    pypi
attrs                     24.2.0                   pypi_0    pypi
bzip2                     1.0.8                h5eee18b_6  
ca-certificates           2024.9.24            h06a4308_0  
chex                      0.1.87                   pypi_0    pypi
cloudpickle               3.1.0                    pypi_0    pypi
decorator                 5.1.1                    pypi_0    pypi
dill                      0.3.9                    pypi_0    pypi
distrax                   0.1.5                    pypi_0    pypi
dm-tree                   0.1.8                    pypi_0    pypi
etils                     1.10.0                   pypi_0    pypi
ferminet                  0.2                       dev_0    <develop>
flake8                    7.1.1                    pypi_0    pypi
folx                      0.2.12                   pypi_0    pypi
gast                      0.6.0                    pypi_0    pypi
h5py                      3.12.1                   pypi_0    pypi
immutabledict             4.2.1                    pypi_0    pypi
importlab                 0.8.1                    pypi_0    pypi
iniconfig                 2.0.0                    pypi_0    pypi
isort                     5.13.2                   pypi_0    pypi
jax                       0.4.34                   pypi_0    pypi
jax-cuda12-pjrt           0.4.34                   pypi_0    pypi
jax-cuda12-plugin         0.4.34                   pypi_0    pypi
jaxlib                    0.4.34                   pypi_0    pypi
jaxtyping                 0.2.36                   pypi_0    pypi
jinja2                    3.1.4                    pypi_0    pypi
kfac-jax                  0.0.6                    pypi_0    pypi
ld_impl_linux-64          2.40                 h12ee557_0  
libcst                    1.5.1                    pypi_0    pypi
libffi                    3.4.4                h6a678d5_1  
libgcc-ng                 11.2.0               h1234567_1  
libgomp                   11.2.0               h1234567_1  
libstdcxx-ng              11.2.0               h1234567_1  
libuuid                   1.41.5               h5eee18b_0  
markupsafe                3.0.2                    pypi_0    pypi
mccabe                    0.7.0                    pypi_0    pypi
ml-collections            1.0.0                    pypi_0    pypi
ml-dtypes                 0.5.0                    pypi_0    pypi
msgspec                   0.18.6                   pypi_0    pypi
ncurses                   6.4                  h6a678d5_0  
networkx                  3.4.2                    pypi_0    pypi
ninja                     1.11.1.1                 pypi_0    pypi
numpy                     2.1.3                    pypi_0    pypi
openssl                   3.0.15               h5eee18b_0  
opt-einsum                3.4.0                    pypi_0    pypi
optax                     0.2.4                    pypi_0    pypi
packaging                 24.2                     pypi_0    pypi
pandas                    2.2.3                    pypi_0    pypi
parameterized             0.9.0                    pypi_0    pypi
pip                       24.2            py311h06a4308_0  
platformdirs              4.3.6                    pypi_0    pypi
pluggy                    1.5.0                    pypi_0    pypi
pyblock                   0.6                      pypi_0    pypi
pycnite                   2024.7.31                pypi_0    pypi
pycodestyle               2.12.1                   pypi_0    pypi
pydot                     3.0.2                    pypi_0    pypi
pyflakes                  3.2.0                    pypi_0    pypi
pylint                    3.3.1                    pypi_0    pypi
pyparsing                 3.2.0                    pypi_0    pypi
pyscf                     2.7.0                    pypi_0    pypi
pytest                    8.3.3                    pypi_0    pypi
python                    3.11.10              he870216_0  
python-dateutil           2.9.0.post0              pypi_0    pypi
pytype                    2024.10.11               pypi_0    pypi
pytz                      2024.2                   pypi_0    pypi
pyyaml                    6.0.2                    pypi_0    pypi
readline                  8.2                  h5eee18b_0  
scipy                     1.14.1                   pypi_0    pypi
setuptools                75.1.0          py311h06a4308_0  
six                       1.16.0                   pypi_0    pypi
sqlite                    3.45.3               h5eee18b_0  
tabulate                  0.9.0                    pypi_0    pypi
tensorflow-probability    0.25.0                   pypi_0    pypi
tk                        8.6.14               h39e8969_0  
toml                      0.10.2                   pypi_0    pypi
tomlkit                   0.13.2                   pypi_0    pypi
toolz                     1.0.0                    pypi_0    pypi
typing-extensions         4.12.2                   pypi_0    pypi
tzdata                    2024.2                   pypi_0    pypi
wheel                     0.44.0          py311h06a4308_0  
xz                        5.4.6                h5eee18b_1  
zlib                      1.2.13               h5eee18b_1  
```
