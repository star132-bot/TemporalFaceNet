"""
Orangutan 数据准备脚本
从原始数据目录复制图片到训练/验证/测试目录
"""

import os
import shutil
import pandas as pd
from pathlib import Path


def setup_directories(base_dir: str) -> dict:
    """创建数据目录结构"""
    directories = {
        'train': os.path.join(base_dir, 'train'),
        'val': os.path.join(base_dir, 'val'),
        'test': os.path.join(base_dir, 'test')
    }
    
    for dir_path in directories.values():
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"创建目录: {dir_path}")
    
    return directories


def get_filenames_from_csv(csv_path: str) -> list:
    """从CSV文件提取图片文件名"""
    df = pd.read_csv(csv_path)
    # 提取文件名，去掉 "face_images/" 前缀
    filenames = df['filename'].str.replace('face_images/', '', regex=False).tolist()
    return filenames


def copy_images(source_dir: str, target_dir: str, filenames: list) -> tuple:
    """
    复制图片文件
    返回: (成功数量, 失败数量, 失败的文件名列表)
    """
    success_count = 0
    failed_count = 0
    failed_files = []
    
    for filename in filenames:
        source_path = os.path.join(source_dir, filename)
        target_path = os.path.join(target_dir, os.path.basename(filename))
        
        if os.path.exists(source_path):
            shutil.copy2(source_path, target_path)
            success_count += 1
        else:
            failed_count += 1
            failed_files.append(filename)
    
    return success_count, failed_count, failed_files


def main():
    # 配置路径
    project_root = r"d:\Pycharmxm\pytorch_main\Project_KSJ\CrossTemporalFaceRecognition"
    
    # 源目录 - 根据实际数据位置修正
    source_dir = r"d:\Pycharmxm\pytorch_main\Project_KSJ\data\data_CTai\face_images"
    
    target_base_dir = os.path.join(project_root, "data", "orangutan")
    splits_dir = os.path.join(project_root, "splits")
    
    # CSV文件路径
    csv_files = {
        'train': os.path.join(splits_dir, 'orangutan_train.csv'),
        'val': os.path.join(splits_dir, 'orangutan_val.csv'),
        'test': os.path.join(splits_dir, 'orangutan_test.csv')
    }
    
    print("=" * 50)
    print("Orangutan 数据准备脚本")
    print("=" * 50)
    print(f"源目录: {source_dir}")
    print(f"目标目录: {target_base_dir}")
    print()
    
    # 创建目标目录
    directories = setup_directories(target_base_dir)
    print()
    
    # 处理每个数据集
    total_success = 0
    total_failed = 0
    all_failed_files = []
    
    for split_name, csv_path in csv_files.items():
        print(f"处理 {split_name} 数据集...")
        
        # 获取文件名
        filenames = get_filenames_from_csv(csv_path)
        print(f"  - CSV记录数: {len(filenames)}")
        
        # 复制图片
        target_dir = directories[split_name]
        success, failed, failed_list = copy_images(source_dir, target_dir, filenames)
        
        print(f"  - 成功复制: {success} 张")
        print(f"  - 复制失败: {failed} 张")
        
        if failed_list:
            print(f"  - 失败文件: {failed_list[:5]}{'...' if len(failed_list) > 5 else ''}")
        
        total_success += success
        total_failed += failed
        all_failed_files.extend(failed_list)
        print()
    
    # 总结
    print("=" * 50)
    print("数据准备完成!")
    print(f"总共成功复制: {total_success} 张")
    print(f"总共复制失败: {total_failed} 张")
    print("=" * 50)
    
    # 验证目录内容
    print("\n目录文件统计:")
    for split_name, dir_path in directories.items():
        count = len([f for f in os.listdir(dir_path) if f.endswith('.png')])
        print(f"  - {split_name}: {count} 张图片")


if __name__ == "__main__":
    main()
