# Cross Temporal Face Recognition

## 项目背景

**竞赛命题**：西北大学 - 跨时间域生物面部识别算法研发

### 整体背景
生物面部识别技术已广泛应用于身份认证、安防监控、智能终端交互等多个领域。然而，生物面部特征会随时间维度发生自然变化（如成长发育、衰老、环境影响等），传统面部识别算法在面对跨时间跨度的面部匹配任务时，识别精度大幅下降，存在"时间域泛化能力不足"的行业痛点。

### 业务需求
- 安防领域：历史监控画面与当前采集图像比对追踪目标
- 公共服务领域：核验多年前录入的身份信息与当前申请人面部特征的一致性
- 特殊场景：基于不同时期的生物面部数据完成身份追溯

### 任务要求
1. **数据集构建**：
   - 非人类生物数据集：时间跨度≥3年，训练-测试时期完全互斥
   - 人脸数据集： CACD、FG-NET、AgeDB（按时域划分，时期互斥）

2. **模型设计**：
   - 基于 Swin Transformer / ResNet / Vision Transformer
   - 时间对比学习（Temporal Contrastive Learning）
   - 非人类生物训练 + 人类面部迁移验证

3. **评估指标**（必须）：
   - `accuracy` ≥ 80%（跨时识别）
   - `TAR@FAR=0.1`
   - `Rank-1`
   - 人脸迁移泛化 `accuracy` ≥ 65%

---

## 目录结构

```
CrossTemporalFaceRecognition/
├── data/
│   ├── orangutan/
│   │   ├── raw/           # 原始猩猩面部图像
│   │   └── processed/     # 预处理后的数据
│   └── human/
│       ├── cacd/          # CACD 数据集
│       ├── fg_net/        # FG-NET 数据集
│       ├── agedb/         # AgeDB 数据集
│       ├── raw/
│       └── processed/
├── splits/                # 训练/测试划分
│   ├── orangutan_split.csv
│   ├── cacd_split.csv
│   ├── fg_net_split.csv
│   └── agedb_split.csv
├── models/                # 训练权重
│   ├── orangutan_swin.pth
│   └── human_finetuned.pth
├── outputs/
│   ├── metrics/           # 评估指标
│   └── visualizations/    # 可视化结果
├── src/
│   ├── __init__.py
│   ├── config.py          # 配置管理
│   ├── dataset.py         # 数据集加载
│   ├── augmentation.py    # 数据增强
│   ├── model.py           # Swin Transformer + TCL
│   ├── loss.py            # 时间对比损失
│   ├── train.py           # 训练脚本
│   ├── finetune.py        # 迁移微调
│   └── evaluate.py        # 评估脚本
├── configs/
│   └── default.yaml       # 默认配置
├── train_orangutan.py     # 猩猩训练入口
├── finetune_human.py      # 人类迁移入口
├── evaluate.py            # 评估入口
└── requirements.txt      # 依赖
```

---

## 数据集说明

### 非人类生物数据集（Orangutan）
- 时间跨度：≥3年
- 划分要求：训练集与测试集时期完全互斥
- 测试集必须包含训练阶段从未出现过的时期

### 人脸数据集
| 数据集 | 下载地址 | 说明 |
|--------|----------|------|
| CACD | http://bcsiriuschen.com/CARC/ | 13年间断年份 |
| FG-NET | https://yanweifu.github.io/FG_NET_dataset/ | 跨年龄 |
| AgeDB | http://www.conceptolvers.com/agedb/ | 多年份 |

---

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 准备数据集

```bash
# 下载并处理数据集
python src/dataset.py --download
python src/preprocess.py
```

### 3. 训练猩猩模型

```bash
python train_orangutan.py --config configs/default.yaml
```

### 4. 迁移到人脸数据集

```bash
python finetune_human.py --config configs/default.yaml --checkpoint models/orangutan_swin.pth
```

### 5. 评估模型

```bash
python evaluate.py --checkpoint models/human_finetuned.pth --dataset cacd
python evaluate.py --checkpoint models/human_finetuned.pth --dataset fg_net
python evaluate.py --checkpoint models/human_finetuned.pth --dataset agedb
```

---

## 模型架构

- **Backbone**: Swin Transformer (Tiny/Small)
- **Head**: Temporal Contrastive Learning Head
- **Loss**: InfoNCE + Triplet Loss + Cross-Entropy

---

## 评估指标

| 指标 | 非人类生物 | 人类迁移 |
|------|------------|----------|
| Accuracy | ≥80% | ≥65% |
| TAR@FAR=0.1 | - | - |
| Rank-1 | - | - |

---

## 参考

- Swin Transformer: https://arxiv.org/abs/2103.14030
- Temporal Contrastive Learning: https://arxiv.org/abs/...
