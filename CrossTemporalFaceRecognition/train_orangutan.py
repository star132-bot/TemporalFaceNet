#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Cross Temporal Face Recognition
训练入口 - 猩猩数据集训练脚本

Usage:
    python train_orangutan.py --config configs/default.yaml
    python train_orangutan.py --config configs/default.yaml --epochs 50 --batch_size 16
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
    parser = argparse.ArgumentParser(description='Train Orangutan Cross-Temporal Face Recognition Model')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                        help='Path to config file')
    parser.add_argument('--dataset', type=str, default='orangutan',
                        help='Dataset to train on')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size (overrides config)')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of epochs (overrides config)')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate (overrides config)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train_main(args)
