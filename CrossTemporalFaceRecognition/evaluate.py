#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Cross Temporal Face Recognition
评估入口 - 评估模型性能

Usage:
    python evaluate.py --checkpoint models/orangutan_best.pth --dataset orangutan
    python evaluate.py --checkpoint models/human_cacd_best.pth --dataset cacd
"""

import sys
import os
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.evaluate import main as evaluate_main
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate Cross-Temporal Face Recognition Model')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                        help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--dataset', type=str, default='orangutan',
                        choices=['orangutan', 'cacd', 'fg_net', 'agedb'],
                        help='Dataset to evaluate on')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size (overrides config)')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    evaluate_main(args)
