# Cross Temporal Face Recognition
# Evaluation Script - 评估模型性能

import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve
import pandas as pd

# Get project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from src.config import load_config, get_default_config
from src.dataset import OrangutanDataset, HumanDataset
from src.augmentation import build_transforms
from src.model import build_model


def set_seed(seed: int = 42):
    """设置随机种子"""
    np.random.seed(seed)
    torch.manual_seed(seed)


def setup_logging(output_dir: str):
    """设置日志"""
    os.makedirs(output_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(output_dir, 'eval.log')),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def extract_features(model, data_loader, device):
    """提取特征向量"""
    model.eval()
    features_list = []
    labels_list = []
    years_list = []

    with torch.no_grad():
        for images, labels, years in tqdm(data_loader, desc='Extracting features'):
            images = images.to(device)
            features = model.extract_features(images)
            features_list.append(features.cpu().numpy())
            labels_list.append(labels.numpy())
            years_list.append(years.numpy())

    features = np.concatenate(features_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    years = np.concatenate(years_list, axis=0)

    return features, labels, years


def compute_accuracy(predictions, labels):
    """计算准确率"""
    return accuracy_score(labels, predictions) * 100


def compute_rank_k(features, labels, k=1, batch_size=500):
    """
    计算 Rank-K 准确率
    对于每个查询样本，找到最近的 K 个样本，如果其中有正确标签则计为成功
    使用分块计算来节省内存
    """
    n = features.shape[0]
    features = features.astype(np.float32)
    
    correct = 0
    for i in range(n):
        # Compute distances to all other samples in batches
        query = features[i:i+1]  # Shape: (1, dim)
        
        # Compute distances in batches to save memory
        min_distances = []
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_features = features[start:end]
            # Compute Euclidean distance
            dist = np.sqrt(((batch_features - query) ** 2).sum(axis=1))
            min_distances.append((start, end, dist))
        
        # Get all distances and sort
        all_distances = []
        all_indices = []
        for start, end, dist in min_distances:
            for j, d in enumerate(dist):
                if start + j != i:  # Exclude self
                    all_distances.append(d)
                    all_indices.append(start + j)
        
        # Sort and get top-k
        sorted_order = np.argsort(all_distances)[:k]
        top_k_indices = [all_indices[idx] for idx in sorted_order]
        
        # Check if any of the top-k has the same label
        top_k_labels = [labels[idx] for idx in top_k_indices]
        if labels[i] in top_k_labels:
            correct += 1

    return correct / n * 100


def compute_tar_at_far(features, labels, far_target=0.1, batch_size=500):
    """
    计算 TAR @ FAR
    TAR (True Accept Rate) = True Positive / (True Positive + False Negative)
    FAR (False Accept Rate) = False Positive / (False Positive + True Negative)

    简化版本：使用余弦相似度和阈值，使用分块计算节省内存
    """
    n = features.shape[0]
    features = F.normalize(torch.tensor(features), dim=1).numpy()
    
    # Compute similarities in batches to save memory
    genuine_scores = []
    impostor_scores = []
    
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_features = features[start:end]
        sim_batch = np.dot(batch_features, features.T)
        
        for i in range(sim_batch.shape[0]):
            global_idx = start + i
            for j in range(n):
                if global_idx < j:  # Avoid duplicates
                    if labels[global_idx] == labels[j]:
                        genuine_scores.append(sim_batch[i, j])
                    else:
                        impostor_scores.append(sim_batch[i, j])
    
    if len(genuine_scores) == 0 or len(impostor_scores) == 0:
        return 0.0
    
    genuine_scores = np.array(genuine_scores)
    impostor_scores = np.array(impostor_scores)
    
    # Compute FAR
    n_impostor = len(impostor_scores)
    far_thresholds = np.sort(impostor_scores)[int(n_impostor * (1 - far_target)):]
    far_threshold = far_thresholds[0] if len(far_thresholds) > 0 else impostor_scores.min()
    
    # Compute TAR at this FAR
    tar = (genuine_scores >= far_threshold).mean() * 100
    
    return tar


def compute_temporal_metrics(features, labels, years):
    """
    计算跨时间域的评估指标
    按时间区间分组评估
    """
    unique_years = sorted(np.unique(years))
    year_pairs = []

    for i in range(len(unique_years)):
        for j in range(i + 1, len(unique_years)):
            year_pairs.append((unique_years[i], unique_years[j]))

    results = {}
    for year1, year2 in year_pairs:
        mask1 = years == year1
        mask2 = years == year2
        combined_mask = mask1 | mask2

        if combined_mask.sum() < 2:
            continue

        year_features = features[combined_mask]
        year_labels = labels[combined_mask]

        # Compute pairwise similarities
        year_features_norm = F.normalize(torch.tensor(year_features), dim=1).numpy()
        similarities = np.dot(year_features_norm, year_features_norm.T)

        # Get predictions
        predictions = np.argmax(similarities, axis=1)

        acc = compute_accuracy(predictions, year_labels)
        results[f"{year1}_vs_{year2}"] = acc

    return results


@torch.no_grad()
def evaluate(model, test_loader, device, config, logger):
    """评估模型"""
    model.eval()

    # Extract features
    features, labels, years = extract_features(model, test_loader, device)

    logger.info(f"Extracted {len(features)} features")

    # Compute overall accuracy using classification
    all_predictions = []
    all_labels = []

    for i in range(len(features)):
        feature = torch.tensor(features[i]).unsqueeze(0).to(device)
        # Use cosine similarity for prediction
        all_features = torch.tensor(features).to(device)
        similarities = F.cosine_similarity(feature, all_features)
        prediction = similarities.argmax().item()
        all_predictions.append(prediction)
        all_labels.append(labels[i])

    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)

    # Accuracy
    accuracy = compute_accuracy(all_predictions, all_labels)
    logger.info(f"Accuracy: {accuracy:.2f}%")

    # Rank-1
    rank_1 = compute_rank_k(features, labels, k=1)
    logger.info(f"Rank-1: {rank_1:.2f}%")

    # Rank-5
    rank_5 = compute_rank_k(features, labels, k=5)
    logger.info(f"Rank-5: {rank_5:.2f}%")

    # TAR @ FAR=0.1
    try:
        tar_at_far = compute_tar_at_far(features, labels, far_target=0.1)
        logger.info(f"TAR@FAR=0.1: {tar_at_far:.2f}%")
    except Exception as e:
        logger.warning(f"Could not compute TAR@FAR: {e}")
        tar_at_far = 0.0

    # Temporal metrics
    temporal_results = compute_temporal_metrics(features, labels, years)
    logger.info("Temporal Results:")
    for year_pair, acc in temporal_results.items():
        logger.info(f"  {year_pair}: {acc:.2f}%")

    # Save results
    results = {
        'accuracy': accuracy,
        'rank_1': rank_1,
        'rank_5': rank_5,
        'tar_at_far': tar_at_far,
        'temporal': temporal_results
    }

    return results


def main(args):
    # Load config
    if args.config:
        if not os.path.isabs(args.config):
            args.config = os.path.join(PROJECT_ROOT, args.config)
    else:
        config = get_default_config()

    # Convert relative paths to absolute paths based on PROJECT_ROOT
    config.data.orangutan_root = os.path.join(PROJECT_ROOT, config.data.orangutan_root)
    config.data.human_root = os.path.join(PROJECT_ROOT, config.data.human_root)
    config.output_dir = os.path.join(PROJECT_ROOT, config.output_dir)
    config.model_dir = os.path.join(PROJECT_ROOT, config.model_dir)
    config.log_dir = os.path.join(PROJECT_ROOT, config.log_dir)

    # Override
    if args.device:
        config.device = args.device
    if args.batch_size:
        config.data.batch_size = args.batch_size

    # Set seed
    set_seed(config.seed)

    # Setup
    device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
    logger = setup_logging(config.output_dir)

    logger.info(f"Using device: {device}")
    logger.info(f"Evaluating on dataset: {args.dataset}")

    # Build transforms
    test_transform = build_transforms(
        image_size=config.data.image_size,
        is_training=False
    )

    # Build dataset
    # Use absolute path for splits directory
    splits_dir = os.path.join(PROJECT_ROOT, 'splits')

    if args.dataset == 'orangutan':
        # Data is in train/val/test directories (not processed)
        test_dataset = OrangutanDataset(
            root=os.path.join(config.data.orangutan_root, 'test'),
            split_file=os.path.join(splits_dir, 'orangutan_test.csv'),
            transform=test_transform
        )
    else:
        test_dataset = HumanDataset(
            root=os.path.join(config.data.human_root, args.dataset),
            split_file=os.path.join(splits_dir, f'{args.dataset}_test.csv'),
            transform=test_transform,
            dataset_type=args.dataset
        )

    logger.info(f"Test dataset size: {len(test_dataset)}")
    logger.info(f"Number of identities: {len(test_dataset.identity_ids)}")

    # Build dataloader
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=True
    )

    # Build model
    model = build_model(
        backbone=config.model.backbone,
        num_identities=len(test_dataset.identity_ids),
        use_temporal_cl=False,
        pretrained=False
    )

    # Load checkpoint
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        # Only load backbone weights, skip head layer (different num_identities)
        state_dict = checkpoint['model_state_dict']
        model_state = model.state_dict()
        filtered_state = {k: v for k, v in state_dict.items() 
                         if k in model_state and model_state[k].shape == v.shape}
        model.load_state_dict(filtered_state, strict=False)
        logger.info(f"Loaded checkpoint: {args.checkpoint} (backbone only, skipped head)")
    else:
        logger.warning("No checkpoint provided!")

    model = model.to(device)

    # Evaluate
    results = evaluate(model, test_loader, device, config, logger)

    # Save results
    results_path = os.path.join(
        config.output_dir,
        f'{args.dataset}_results.csv'
    )

    # Convert results to DataFrame
    results_df = pd.DataFrame([{
        'dataset': args.dataset,
        'accuracy': results['accuracy'],
        'rank_1': results['rank_1'],
        'rank_5': results['rank_5'],
        'tar_at_far': results['tar_at_far'],
    }])
    results_df.to_csv(results_path, index=False)
    logger.info(f"Results saved to: {results_path}")

    # Print summary
    logger.info("=" * 50)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 50)
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Accuracy: {results['accuracy']:.2f}%")
    logger.info(f"Rank-1: {results['rank_1']:.2f}%")
    logger.info(f"Rank-5: {results['rank_5']:.2f}%")
    logger.info(f"TAR@FAR=0.1: {results['tar_at_far']:.2f}%")
    logger.info("=" * 50)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--dataset', type=str, default='orangutan',
                        choices=['orangutan', 'cacd', 'fg_net', 'agedb'])
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--batch_size', type=int, default=None)
    args = parser.parse_args()

    main(args)
