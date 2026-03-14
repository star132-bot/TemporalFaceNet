#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Cross Temporal Face Recognition
迁移微调入口 - 人类数据集微调脚本

Usage:
    python finetune_human.py --checkpoint models/orangutan_best.pth --dataset cacd
    python finetune_human.py --checkpoint models/orangutan_best.pth --dataset fg_net
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.finetune import main as finetune_main
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Fine-tune on Human Face Datasets')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                        help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to pretrained checkpoint')
    parser.add_argument('--dataset', type=str, default='cacd',
                        choices=['cacd', 'fg_net', 'agedb'],
                        help='Human dataset to fine-tune on')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size (overrides config)')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of epochs (overrides config)')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate (overrides config)')
    parser.add_argument('--freeze_backbone', action='store_true',
                        help='Freeze backbone first')
    parser.add_argument('--freeze_epochs', type=int, default=5,
                        help='Epochs to train with frozen backbone')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    finetune_main(args)
