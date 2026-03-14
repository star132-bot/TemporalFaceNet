# Cross Temporal Face Recognition - Docker Image
# 基于 PyTorch 官方 CUDA 镜像

FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

# 设置工作目录
WORKDIR /workspace

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# 复制项目文件
COPY . /workspace/

# 安装 Python 依赖
COPY requirements.txt /workspace/
RUN pip install --no-cache-dir -r /workspace/requirements.txt

# 设置环境变量
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0

# 默认命令
CMD ["/bin/bash"]
