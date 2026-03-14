# Cross Temporal Face Recognition
# Dataset Module

import os
import torch
import pandas as pd
import numpy as np
from PIL import Image
from typing import Optional, List, Tuple, Dict
from torch.utils.data import Dataset, Subset


class TemporalFaceDataset(Dataset):
    """跨时间人脸数据集基类"""

    def __init__(
        self,
        root: str,
        split_file: str,
        transform=None,
        years: Optional[List[int]] = None
    ):
        self.root = root
        self.transform = transform
        self.years = years

        # Load split file
        self.split_df = pd.read_csv(split_file)

        # Filter by years if specified
        if years is not None:
            self.split_df = self.split_df[self.split_df['year'].isin(years)]

        # Get unique identities
        self.identity_ids = sorted(self.split_df['identity_id'].unique())
        self.identity_to_idx = {id_: idx for idx, id_ in enumerate(self.identity_ids)}

    def __len__(self) -> int:
        return len(self.split_df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, int]:
        row = self.split_df.iloc[idx]

        # Handle different column names in CSV files
        if 'image_path' in self.split_df.columns:
            img_path = os.path.join(self.root, row['image_path'])
        elif 'filename' in self.split_df.columns:
            # For orangutan data: filename is like "face_images/img-xxx.png"
            # but actual images are directly in train/val/test directories (no face_images subfolder)
            filename = row['filename']
            # Remove "face_images/" prefix if present
            if filename.startswith('face_images/'):
                filename = filename.replace('face_images/', '')
            img_path = os.path.join(self.root, filename)

        # Load image
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        identity_id = self.identity_to_idx[row['identity_id']]
        year = row['year']

        return image, identity_id, year


class OrangutanDataset(TemporalFaceDataset):
    """猩猩面部数据集"""

    def __init__(self, root: str, split_file: str, transform=None, years: Optional[List[int]] = None):
        super().__init__(root, split_file, transform, years)
        self.dataset_name = "orangutan"


class HumanDataset(TemporalFaceDataset):
    """人类面部数据集（支持 CACD, FG-NET, AgeDB）"""

    def __init__(
        self,
        root: str,
        split_file: str,
        transform=None,
        years: Optional[List[int]] = None,
        dataset_type: str = "cacd"
    ):
        super().__init__(root, split_file, transform, years)
        self.dataset_name = dataset_type


def create_train_val_test_split(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    min_samples_per_identity: int = 2,
    seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    按时间域划分训练/验证/测试集
    确保测试集包含训练集未见过的时间点
    """
    np.random.seed(seed)

    # Get unique years and identities
    years = sorted(df['year'].unique())
    identities = df['identity_id'].unique()

    # Split years for train vs test (time-wise disjoint)
    n_test_years = max(1, len(years) // 5)  # 20% years for test
    test_years = years[-n_test_years:]
    train_val_years = years[:-n_test_years]

    # Split train_val years into train and val
    n_val_years = max(1, len(train_val_years) // 5)
    val_years = train_val_years[-n_val_years:]
    train_years = train_val_years[:-n_val_years]

    # Create splits
    train_df = df[df['year'].isin(train_years)]
    val_df = df[df['year'].isin(val_years)]
    test_df = df[df['year'].isin(test_years)]

    # Filter identities with enough samples
    def filter_by_samples(data_df, min_samples):
        identity_counts = data_df.groupby('identity_id').size()
        valid_identities = identity_counts[identity_counts >= min_samples].index
        return data_df[data_df['identity_id'].isin(valid_identities)]

    train_df = filter_by_samples(train_df, 1)
    val_df = filter_by_samples(val_df, min_samples_per_identity)
    test_df = filter_by_samples(test_df, min_samples_per_identity)

    return train_df, val_df, test_df


def save_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: str,
    dataset_name: str
):
    """保存划分结果到 CSV"""
    os.makedirs(output_dir, exist_ok=True)

    train_df.to_csv(os.path.join(output_dir, f"{dataset_name}_train.csv"), index=False)
    val_df.to_csv(os.path.join(output_dir, f"{dataset_name}_val.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, f"{dataset_name}_test.csv"), index=False)


def get_dataset_info(dataset: Dataset) -> Dict:
    """获取数据集信息"""
    if hasattr(dataset, 'split_df'):
        df = dataset.split_df
        return {
            'num_samples': len(df),
            'num_identities': df['identity_id'].nunique(),
            'year_range': (df['year'].min(), df['year'].max()),
            'years': sorted(df['year'].unique()),
        }
    return {}


class DatasetDownloader:
    """数据集下载器"""

    @staticmethod
    def download_cacd(output_dir: str):
        """下载 CACD 数据集"""
        # CACD 下载链接: http://bcsiriuschen.com/CARC/
        # 需要手动下载或使用脚本
        print(f"Please download CACD dataset manually from:")
        print(f"  http://bcsiriuschen.com/CARC/")
        print(f"  Save to: {output_dir}")

    @staticmethod
    def download_fg_net(output_dir: str):
        """下载 FG-NET 数据集"""
        print(f"Please download FG-NET dataset manually from:")
        print(f"  https://yanweifu.github.io/FG_NET_dataset/")
        print(f"  Save to: {output_dir}")

    @staticmethod
    def download_agedb(output_dir: str):
        """下载 AgeDB 数据集"""
        print(f"Please download AgeDB dataset manually from:")
        print(f"  http://www.conceptolvers.com/agedb/")
        print(f"  Save to: {output_dir}")
