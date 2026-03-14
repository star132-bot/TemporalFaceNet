# Cross Temporal Face Recognition
# Fine-tuning Script for Human Datasets

import os
import sys
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import logging

# Get project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from src.config import load_config, get_default_config
from src.dataset import HumanDataset, get_dataset_info
from src.augmentation import build_transforms
from src.model import build_model
from src.loss import build_loss


def set_seed(seed: int = 42):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_logging(output_dir: str):
    """设置日志"""
    os.makedirs(output_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(output_dir, 'finetune.log')),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def finetune_one_epoch(
    model,
    train_loader,
    criterion,
    optimizer,
    scaler,
    device,
    epoch,
    config,
    logger,
    freeze_backbone: bool = False
):
    """微调一个 epoch"""
    model.train()

    if freeze_backbone:
        # Freeze backbone, only train head
        for name, param in model.named_parameters():
            if 'backbone' in name:
                param.requires_grad = False
            else:
                param.requires_grad = True

    total_loss = 0
    num_batches = 0

    pbar = tqdm(train_loader, desc=f'Fine-tune Epoch {epoch}')
    for batch_idx, (images, labels, years) in enumerate(pbar):
        images = images.to(device)
        labels = labels.to(device)
        years = years.to(device)

        optimizer.zero_grad()

        with autocast(enabled=config.train.use_amp):
            outputs = model(images, return_features=True)

            if isinstance(outputs, tuple):
                logits, features, tcl_features = outputs
                loss, loss_dict = criterion(
                    logits, features, labels, years, tcl_features
                )
            else:
                logits = outputs
                loss, loss_dict = criterion(logits, logits, labels)

        scaler.scale(loss).backward()

        if config.train.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                config.train.grad_clip
            )

        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        num_batches += 1

        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'lr': optimizer.param_groups[0]['lr']
        })

        if batch_idx % config.train.print_freq == 0:
            logger.info(
                f"Epoch [{epoch}][{batch_idx}/{len(train_loader)}] "
                f"Loss: {loss.item():.4f}"
            )

    avg_loss = total_loss / num_batches
    return avg_loss


@torch.no_grad()
def validate(model, val_loader, criterion, device, config, logger):
    """验证"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(val_loader, desc='Validation')
    for images, labels, years in pbar:
        images = images.to(device)
        labels = labels.to(device)
        years = years.to(device)

        with autocast(enabled=config.train.use_amp):
            outputs = model(images, return_features=True)

            if isinstance(outputs, tuple):
                logits, features, _ = outputs
                loss, _ = criterion(logits, features, labels)
            else:
                logits = outputs
                loss, _ = criterion(logits, logits, labels)

        total_loss += loss.item()

        _, predicted = torch.max(logits, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    avg_loss = total_loss / len(val_loader)
    accuracy = 100 * correct / total

    logger.info(f"Validation Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

    return avg_loss, accuracy


def main(args):
    # Load config
    if args.config:
        config = load_config(args.config)
    else:
        config = get_default_config()

    # Convert relative paths to absolute paths based on PROJECT_ROOT
    config.data.orangutan_root = os.path.join(PROJECT_ROOT, config.data.orangutan_root)
    config.data.human_root = os.path.join(PROJECT_ROOT, config.data.human_root)
    config.output_dir = os.path.join(PROJECT_ROOT, config.output_dir)
    config.model_dir = os.path.join(PROJECT_ROOT, config.model_dir)
    config.log_dir = os.path.join(PROJECT_ROOT, config.log_dir)

    # Override with command line arguments
    if args.device:
        config.device = args.device
    if args.batch_size:
        config.data.batch_size = args.batch_size
    if args.epochs:
        config.train.epochs = args.epochs
    if args.lr:
        config.train.lr = args.lr

    # Set seed
    set_seed(config.seed)

    # Setup
    device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
    logger = setup_logging(config.output_dir)

    logger.info(f"Using device: {device}")
    logger.info(f"Fine-tuning on human dataset: {args.dataset}")

    # Create output directories
    os.makedirs(config.model_dir, exist_ok=True)
    os.makedirs(config.output_dir, exist_ok=True)

    # Build transforms
    train_transform = build_transforms(
        image_size=config.data.image_size,
        is_training=True
    )
    val_transform = build_transforms(
        image_size=config.data.image_size,
        is_training=False
    )

    # Build datasets
    # Use absolute path for splits directory
    splits_dir = os.path.join(PROJECT_ROOT, 'splits')

    train_dataset = HumanDataset(
        root=os.path.join(config.data.human_root, args.dataset),
        split_file=os.path.join(splits_dir, f'{args.dataset}_train.csv'),
        transform=train_transform,
        dataset_type=args.dataset
    )
    val_dataset = HumanDataset(
        root=os.path.join(config.data.human_root, args.dataset),
        split_file=os.path.join(splits_dir, f'{args.dataset}_val.csv'),
        transform=val_transform,
        dataset_type=args.dataset
    )

    # Get number of identities
    num_identities = len(train_dataset.identity_ids)
    logger.info(f"Number of identities: {num_identities}")
    logger.info(f"Dataset info: {get_dataset_info(train_dataset)}")

    # Build dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=True,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=True
    )

    # Build model - load pretrained checkpoint
    model = build_model(
        backbone=config.model.backbone,
        num_identities=num_identities,
        use_temporal_cl=config.model.use_temporal_cl,
        temperature=config.model.temperature,
        pretrained=False  # Load from checkpoint
    )

    # Load pretrained weights
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        logger.info(f"Loaded pretrained checkpoint: {args.checkpoint}")

    model = model.to(device)
    logger.info(f"Model: {config.model.backbone}")

    # Build criterion
    criterion = build_loss(config)

    # Build optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.train.lr,
        weight_decay=config.train.weight_decay
    )

    # Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.train.epochs,
        eta_min=config.train.lr * 0.01
    )

    # Mixed precision scaler
    scaler = GradScaler(enabled=config.train.use_amp)

    # Fine-tuning loop
    best_acc = 0

    # Phase 1: Freeze backbone (optional)
    if args.freeze_backbone:
        logger.info("Phase 1: Training with frozen backbone")
        for epoch in range(args.freeze_epochs):
            train_loss = finetune_one_epoch(
                model, train_loader, criterion, optimizer,
                scaler, device, epoch, config, logger,
                freeze_backbone=True
            )
            val_loss, val_acc = validate(
                model, val_loader, criterion, device, config, logger
            )
            scheduler.step()

    # Phase 2: Fine-tune all layers
    logger.info("Phase 2: Fine-tuning all layers")
    for epoch in range(config.train.epochs):
        train_loss = finetune_one_epoch(
            model, train_loader, criterion, optimizer,
            scaler, device, epoch, config, logger,
            freeze_backbone=False
        )

        val_loss, val_acc = validate(
            model, val_loader, criterion, device, config, logger
        )

        scheduler.step()

        # Save checkpoint
        if (epoch + 1) % config.train.save_freq == 0:
            checkpoint_path = os.path.join(
                config.model_dir,
                f'human_{args.dataset}_epoch_{epoch+1}.pth'
            )
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc,
            }, checkpoint_path)
            logger.info(f"Saved checkpoint: {checkpoint_path}")

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            best_path = os.path.join(
                config.model_dir,
                f'human_{args.dataset}_best.pth'
            )
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc,
            }, best_path)
            logger.info(f"Saved best model: {best_path}")

    logger.info(f"Fine-tuning completed! Best accuracy: {best_acc:.2f}%")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to pretrained checkpoint')
    parser.add_argument('--dataset', type=str, default='cacd',
                        choices=['cacd', 'fg_net', 'agedb'])
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--freeze_backbone', action='store_true',
                        help='Freeze backbone first')
    parser.add_argument('--freeze_epochs', type=int, default=5,
                        help='Epochs to train with frozen backbone')
    args = parser.parse_args()

    main(args)
