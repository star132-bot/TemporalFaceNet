# ChimpTai 数据处理模块

## 概述

本模块用于处理 ChimpTai（黑猩猩）数据集，将其转换为适合跨时间人脸识别模型训练的格式。

## 数据集信息

- **原始数据位置**: `d:\Pycharmxm\pytorch_main\Project_KSJ\data\data_CTai`
- **图像数量**: 3905 张图片，共 5078 个人脸
- **可用人脸**: 4377 张（去除标注不可靠的）
- **身份数量**: 78 个独立身份
- **年龄范围**: 使用年龄作为时间维度

## 处理流程

### 1. 数据解析
解析 `annotations_ctai.txt` 标注文件，提取：
- `filename`: 图片路径
- `identity_id`: 身份名称（如 Fredy, Victor）
- `age`: 年龄
- `age_group`: 年龄组（Infant, Juvenile, Adult, etc.）
- `gender`: 性别

### 2. 数据采样
按要求采样 **10%** 的数据，按身份分层采样确保每个身份都有样本。

### 3. 时间域划分
**核心要求**：训练集和测试集的时期必须完全互斥

- **训练集**: 包含早期年份的数据
- **验证集**: 用于调参
- **测试集**: 包含训练阶段从未出现过的年份（最新年份）

### 4. 输出文件
生成以下 CSV 文件（位于 `splits/` 目录）：
- `orangutan_train.csv`: 训练集划分
- `orangutan_val.csv`: 验证集划分
- `orangutan_test.csv`: 测试集划分

CSV 格式：
```
filename,identity_id,year,age,age_group,gender
face_images/img-id1-object-1.png,Fredy,32,32,Adult,Male
```

## 使用方法

### 基本用法（采样10%）
```bash
python data_processing/process_chimpTai.py
```

### 完整参数
```bash
python data_processing/process_chimpTai.py \
    --sample_ratio 0.1 \
    --train_ratio 0.7 \
    --val_ratio 0.15 \
    --min_samples 2 \
    --copy_images \
    --seed 42
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--sample_ratio` | 0.1 | 采样比例 (0.1 = 10%) |
| `--train_ratio` | 0.7 | 训练集比例 |
| `--val_ratio` | 0.15 | 验证集比例 |
| `--min_samples` | 2 | 每个身份最少样本数 |
| `--copy_images` | False | 是否复制图像到处理后目录 |
| `--seed` | 42 | 随机种子 |

## 输出结果示例

处理完成后会显示：
```
Split statistics:
  Train: XXX samples, XX identities
  Val:   XXX samples, XX identities
  Test:  XXX samples, XX identities

✓ Train/Test years are disjoint: {1, 2, 3, 4} vs {5, 6, 7}
```

## 时间域划分说明

本项目采用**年龄作为时间维度**进行跨时间识别：
- 不同年龄的黑猩猩面部特征会发生变化
- 训练集使用较年轻的样本
- 测试集使用较年长的样本（训练时未见过的年龄段）
- 这模拟了真实的跨时间人脸识别场景

## 竞赛要求对应

| 竞赛要求 | 本项目实现 |
|----------|------------|
| 时间跨度≥3年 | ChimpTai 数据集年龄范围满足 |
| 训练-测试时期互斥 | 按年龄划分实现 |
| 测试集包含未见时期 | 使用最新年龄作为测试 |
