#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Cross Temporal Face Recognition
训练入口 - AgeDB人类数据集训练脚本

Usage:
    python train_agedb.py --checkpoint models/orangutan_best.pth
    python train_agedb.py --checkpoint models/orangutan_best.pth --epochs 30
"""

import sys
import os

# Suppress warnings before any imports
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.train import main as train_main
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Train AgeDB Cross-Temporal Face Recognition Model')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                        help='Path to config file')
    parser.add_argument('--dataset', type=str, default='agedb',
                        help='Dataset to train on (agedb)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size (overrides config)')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of epochs (overrides config)')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate (overrides config)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Resume from checkpoint (pretrained model)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint (for continuing training)')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train_main(args)
