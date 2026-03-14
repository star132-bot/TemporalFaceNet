# Docker 使用指南

## 环境要求

- Docker Desktop (Windows/Mac/Linux)
- NVIDIA Docker (Linux) 或 NVIDIA Container Toolkit
- GPU: NVIDIA GPU with CUDA 12.1+

## 构建镜像

```bash
# 构建 Docker 镜像
docker build -t cross-temporal-face-rec .
```

## 使用 Docker Compose

### 训练模型

```bash
# 启动训练
docker-compose up training

# 后台运行
docker-compose up -d training

# 查看训练日志
docker-compose logs -f training
```

### 评估模型

```bash
# 评估模型
docker-compose run evaluate
```

## 直接使用 Docker

### 训练

```bash
# 训练猩猩模型
docker run --gpus all -v $(pwd):/workspace cross-temporal-face-rec \
    python train_orangutan.py --epochs 30

# 继续训练 AgeDB 模型
docker run --gpus all -v $(pwd):/workspace cross-temporal-face-rec \
    python train_agedb.py --resume models/orangutan_best.pth --epochs 30
```

### 评估

```bash
# 评估模型
docker run --gpus all -v $(pwd):/workspace cross-temporal-face-rec \
    python evaluate.py --checkpoint models/agedb_best.pth --dataset agedb
```

## Windows Docker Desktop 注意事项

1. 启用 WSL2 后端（推荐）
2. 分配足够的资源（CPU、内存、磁盘）
3. Windows 上使用 GPU 需要启用 "Use the Windows Docker Backend" 并安装 NVIDIA Container Toolkit

## 数据目录结构

确保宿主机上有以下目录结构：

```
project/
├── data/
│   ├── orangutan/
│   └── human/
│       └── agedb/
├── models/
├── outputs/
├── splits/
│   ├── orangutan_train.csv
│   ├── orangutan_val.csv
│   ├── agedb_train.csv
│   ├── agedb_val.csv
│   └── agedb_test.csv
├── train_orangutan.py
├── train_agedb.py
├── evaluate.py
└── Dockerfile
```
