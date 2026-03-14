#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ChimpTai (黑猩猩) 数据集处理脚本
功能：
1. 解析 annotations_ctai.txt 标注文件
2. 采样 10% 数据
3. 按时间域划分训练/测试集（确保时期互斥）
4. 生成 CSV 划分文件

作者：AI Assistant
日期：2026-03-12
"""

import os
import sys
import argparse
import random
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import shutil


# ==================== 配置 ====================
DATA_ROOT = r"d:\Pycharmxm\pytorch_main\Project_KSJ\data\data_CTai"
ANNOTATION_FILE = "annotations_ctai.txt"
FACE_IMAGES_DIR = "face_images"
OUTPUT_ROOT = r"d:\Pycharmxm\pytorch_main\Project_KSJ\CrossTemporalFaceRecognition\data\orangutan"


# ==================== 数据解析 ====================
def parse_annotation_line(line: str) -> dict:
    """
    解析一行标注信息

    格式: Filename face_images/img-id1-object-1.png Name Fredy Age 32 Age_Group Adult Gender Male ...
    或:   Filename face_images/img-id107-object-1.png Name Adult Age NaN Age_Group SubAdult ...

    Returns:
        dict: 包含 filename, identity_id, age, age_group, gender 等字段
    """
    parts = line.strip().split()

    # 解析Filename
    filename = parts[1]  # face_images/img-id1-object-1.png

    # 解析Name
    name_idx = parts.index("Name")
    identity_name = parts[name_idx + 1]

    # 解析Age (可能是数字或NaN)
    age_idx = parts.index("Age")
    age_str = parts[age_idx + 1]
    try:
        age = int(age_str) if age_str != "NaN" else None
    except ValueError:
        age = None

    # 解析Age_Group
    age_group_idx = parts.index("Age_Group")
    age_group = parts[age_group_idx + 1]

    # 解析Gender
    gender_idx = parts.index("Gender")
    gender = parts[gender_idx + 1]

    # 从文件名提取ID
    # img-id1-object-1.png -> img_id=1, object=1
    basename = os.path.basename(filename)
    # img-id1-object-1 -> id1, object1
    parts_name = basename.replace(".png", "").split("-")
    img_id = int(parts_name[1].replace("id", ""))
    obj_id = int(parts_name[3].replace("object", ""))

    # 如果年龄为None，使用Age_Group作为时间维度
    year = age if age is not None else 0

    return {
        "filename": filename,
        "image_id": img_id,
        "object_id": obj_id,
        "identity_id": identity_name,
        "identity_idx": 0,  # 后续会被替换为索引
        "age": age,
        "age_group": age_group,
        "gender": gender,
        "year": year  # 使用年龄作为时间维度
    }


def load_annotations(annotation_path: str) -> pd.DataFrame:
    """
    加载并解析所有标注信息
    过滤掉年龄为NaN的记录
    """
    print(f"Loading annotations from: {annotation_path}")

    data_list = []
    with open(annotation_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                parsed = parse_annotation_line(line)
                # 过滤掉年龄为None的记录
                if parsed['age'] is not None:
                    data_list.append(parsed)
            except Exception as e:
                print(f"Warning: Failed to parse line: {line[:50]}... Error: {e}")

    df = pd.DataFrame(data_list)

    print(f"Loaded {len(df)} face annotations (filtered out NaN ages)")
    print(f"Unique identities: {df['identity_id'].nunique()}")
    print(f"Age range: {df['age'].min()} - {df['age'].max()}")
    print(f"Age groups: {df['age_group'].unique()}")

    return df


# ==================== 数据采样 ====================
def sample_data(df: pd.DataFrame, sample_ratio: float = 0.1, seed: int = 42) -> pd.DataFrame:
    """
    采样数据

    按身份分层采样，确保每个身份都有样本

    Args:
        df: 完整数据集
        sample_ratio: 采样比例 (0.1 = 10%)
        seed: 随机种子

    Returns:
        pd.DataFrame: 采样后的数据集
    """
    print(f"\nSampling {sample_ratio*100}% of data...")

    random.seed(seed)
    np.random.seed(seed)

    # 按身份分层采样
    sampled_dfs = []
    for identity_id, group in df.groupby('identity_id'):
        n_samples = max(1, int(len(group) * sample_ratio))
        sampled = group.sample(n=n_samples, random_state=seed)
        sampled_dfs.append(sampled)

    sampled_df = pd.concat(sampled_dfs, ignore_index=True)

    print(f"Sampled {len(sampled_df)} faces from {sampled_df['identity_id'].nunique()} identities")
    print(f"Samples per identity: min={sampled_df.groupby('identity_id').size().min()}, "
          f"max={sampled_df.groupby('identity_id').size().max()}, "
          f"mean={sampled_df.groupby('identity_id').size().mean():.1f}")

    return sampled_df


# ==================== 时间域划分 ====================
def create_temporal_split(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    min_samples_per_identity: int = 2,
    seed: int = 42
) -> tuple:
    """
    按时间域划分训练/验证/测试集

    核心要求：训练集和测试集的时期必须完全互斥
    测试集必须包含训练阶段从未出现过的时期

    Args:
        df: 数据集
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        min_samples_per_identity: 每个身份最少样本数
        seed: 随机种子

    Returns:
        tuple: (train_df, val_df, test_df)
    """
    print(f"\nCreating temporal split (train:{train_ratio}, val:{val_ratio}, test:{1-train_ratio-val_ratio})...")

    random.seed(seed)
    np.random.seed(seed)

    # 获取所有年份并排序
    years = sorted(df['year'].unique())
    print(f"Available years: {years}")

    # 划分年份：早期作为训练，后期作为测试
    # 确保测试集的年份在训练集中不会出现
    n_years = len(years)

    # 测试集使用最新的 20% 年份（这些年份在训练中不会出现）
    n_test_years = max(1, n_years // 5)
    test_years = years[-n_test_years:]
    train_val_years = years[:-n_test_years]

    # 训练集和验证集划分：从剩余年份中划分
    n_val_years = max(1, len(train_val_years) // 5)
    val_years = train_val_years[-n_val_years:]
    train_years = train_val_years[:-n_val_years]

    print(f"Train years: {train_years}")
    print(f"Val years: {val_years}")
    print(f"Test years: {test_years}")

    # 创建划分
    train_df = df[df['year'].isin(train_years)].copy()
    val_df = df[df['year'].isin(val_years)].copy()
    test_df = df[df['year'].isin(test_years)].copy()

    # 过滤：每个身份在每个划分中至少有 min_samples_per_identity 个样本
    def filter_by_min_samples(data_df, min_samples):
        identity_counts = data_df.groupby('identity_id').size()
        valid_identities = identity_counts[identity_counts >= min_samples].index
        return data_df[data_df['identity_id'].isin(valid_identities)]

    train_df = filter_by_min_samples(train_df, 1)  # 训练集可以少一些
    val_df = filter_by_min_samples(val_df, min_samples_per_identity)
    test_df = filter_by_min_samples(test_df, min_samples_per_identity)

    print(f"\nSplit statistics:")
    print(f"  Train: {len(train_df)} samples, {train_df['identity_id'].nunique()} identities")
    print(f"  Val:   {len(val_df)} samples, {val_df['identity_id'].nunique()} identities")
    print(f"  Test:  {len(test_df)} samples, {test_df['identity_id'].nunique()} identities")

    # 验证时间互斥
    train_years_set = set(train_df['year'].unique())
    test_years_set = set(test_df['year'].unique())
    overlap = train_years_set & test_years_set

    if overlap:
        print(f"WARNING: Year overlap detected! {overlap}")
    else:
        print(f"[OK] Train/Test years are disjoint: {train_years_set} vs {test_years_set}")

    return train_df, val_df, test_df


def assign_identity_indices(df: pd.DataFrame) -> pd.DataFrame:
    """
    为每个身份分配数字索引
    """
    unique_identities = sorted(df['identity_id'].unique())
    identity_to_idx = {id_: idx for idx, id_ in enumerate(unique_identities)}
    df['identity_idx'] = df['identity_id'].map(identity_to_idx)
    return df


# ==================== 数据复制 ====================
def copy_images(df: pd.DataFrame, source_dir: str, dest_dir: str):
    """
    复制图像到目标目录，保持原有结构
    """
    print(f"\nCopying images to {dest_dir}...")

    os.makedirs(dest_dir, exist_ok=True)

    success_count = 0
    error_count = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Copying"):
        src_path = os.path.join(source_dir, row['filename'])
        dst_path = os.path.join(dest_dir, row['filename'])

        # 创建目标目录
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)

        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
            success_count += 1
        else:
            print(f"Warning: Source file not found: {src_path}")
            error_count += 1

    print(f"Copied {success_count} images, {error_count} errors")


# ==================== 保存划分文件 ====================
def save_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: str
):
    """
    保存划分CSV文件
    """
    print(f"\nSaving split files to {output_dir}...")

    os.makedirs(output_dir, exist_ok=True)

    # 准备保存的DataFrame（只保留需要的列）
    columns = ['filename', 'identity_id', 'year', 'age', 'age_group', 'gender']

    # 保存为CSV
    train_df[columns].to_csv(
        os.path.join(output_dir, 'orangutan_train.csv'),
        index=False
    )
    val_df[columns].to_csv(
        os.path.join(output_dir, 'orangutan_val.csv'),
        index=False
    )
    test_df[columns].to_csv(
        os.path.join(output_dir, 'orangutan_test.csv'),
        index=False
    )

    print(f"  Saved orangutan_train.csv ({len(train_df)} samples)")
    print(f"  Saved orangutan_val.csv ({len(val_df)} samples)")
    print(f"  Saved orangutan_test.csv ({len(test_df)} samples)")


# ==================== 主函数 ====================
def main(args):
    """
    主处理流程
    """
    print("=" * 60)
    print("ChimpTai Dataset Processing")
    print("=" * 60)

    # 1. 加载标注
    annotation_path = os.path.join(DATA_ROOT, ANNOTATION_FILE)
    df = load_annotations(annotation_path)

    # 2. 采样 10%
    if args.sample_ratio < 1.0:
        df = sample_data(df, sample_ratio=args.sample_ratio, seed=args.seed)

    # 3. 分配身份索引
    df = assign_identity_indices(df)

    # 4. 时间域划分
    train_df, val_df, test_df = create_temporal_split(
        df,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        min_samples_per_identity=args.min_samples,
        seed=args.seed
    )

    # 5. 复制图像（可选）
    if args.copy_images:
        source_dir = DATA_ROOT
        dest_dir = os.path.join(OUTPUT_ROOT, 'processed')

        # 复制所有划分的图像
        all_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
        copy_images(all_df, source_dir, dest_dir)

    # 6. 保存划分文件
    splits_dir = os.path.join(OUTPUT_ROOT, '..', '..', 'splits')
    save_splits(train_df, val_df, test_df, splits_dir)

    print("\n" + "=" * 60)
    print("Processing Complete!")
    print("=" * 60)
    print(f"\nOutput files:")
    print(f"  - {splits_dir}/orangutan_train.csv")
    print(f"  - {splits_dir}/orangutan_val.csv")
    print(f"  - {splits_dir}/orangutan_test.csv")

    # 显示统计摘要
    print(f"\nDataset Summary:")
    print(f"  Total samples: {len(train_df) + len(val_df) + len(test_df)}")
    print(f"  Train samples: {len(train_df)}")
    print(f"  Val samples: {len(val_df)}")
    print(f"  Test samples: {len(test_df)}")
    print(f"  Unique identities: {df['identity_id'].nunique()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process ChimpTai dataset for cross-temporal face recognition"
    )
    parser.add_argument('--sample_ratio', type=float, default=0.1,
                        help='Sampling ratio (default: 0.1 = 10%)')
    parser.add_argument('--train_ratio', type=float, default=0.7,
                        help='Training set ratio (default: 0.7)')
    parser.add_argument('--val_ratio', type=float, default=0.15,
                        help='Validation set ratio (default: 0.15)')
    parser.add_argument('--min_samples', type=int, default=2,
                        help='Minimum samples per identity (default: 2)')
    parser.add_argument('--copy_images', action='store_true',
                        help='Copy images to processed directory')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')

    args = parser.parse_args()
    main(args)
