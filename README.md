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

在配置前，请确保已经安装了 Docker 和 NVIDIA Container Toolkit

将本仓库克隆到本地后，在此仓库目录路径下运行以下命令构建 Docker 镜像

```shell
docker build -t ferminet:0.2 .
docker run --gpus=all -it ferminet:0.2
```

### Dockerfile 说明

- 使用 `nvidia/cuda:12.3.2-devel-ubuntu22.04` 作为基础镜像
- 基本环境：python3.11 & JAX with CUDA 12 & cuDNN 9
- 配置 ferminet 环境

### ferminet 使用方式

请参考原仓库说明：[README.md - ferminet](./ferminet/README.md)
