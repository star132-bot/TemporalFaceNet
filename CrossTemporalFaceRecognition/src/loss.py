# Cross Temporal Face Recognition
# Loss Functions Module

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class LabelSmoothingCrossEntropy(nn.Module):
    """Label smoothing cross entropy loss"""

    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, pred, target):
        n_classes = pred.size(-1)
        log_probs = F.log_softmax(pred, dim=-1)

        # Smooth labels
        with torch.no_grad():
            smooth_labels = torch.zeros_like(pred)
            smooth_labels.fill_(self.smoothing / (n_classes - 1))
            smooth_labels.scatter_(1, target.unsqueeze(1), 1 - self.smoothing)

        loss = (-smooth_labels * log_probs).sum(dim=-1).mean()
        return loss


class TripletLoss(nn.Module):
    """Triplet loss with margin"""

    def __init__(self, margin: float = 0.3, distance: str = "cosine"):
        super().__init__()
        self.margin = margin
        self.distance = distance

    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor
    ) -> torch.Tensor:
        if self.distance == "cosine":
            pos_dist = 1 - F.cosine_similarity(anchor, positive, dim=-1)
            neg_dist = 1 - F.cosine_similarity(anchor, negative, dim=-1)
        else:
            pos_dist = F.pairwise_distance(anchor, positive)
            neg_dist = F.pairwise_distance(anchor, negative)

        losses = F.relu(pos_dist - neg_dist + self.margin)
        return losses.mean()


class TemporalContrastiveLoss(nn.Module):
    """Temporal Contrastive Loss - 对比同一身份不同时期的样本"""

    def __init__(self, temperature: float = 0.07, queue_size: int = 16384):
        super().__init__()
        self.temperature = temperature
        self.queue_size = queue_size
        self.register_buffer("queue", torch.randn(queue_size, 128))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("queue_year", torch.zeros(queue_size, dtype=torch.long))

    def forward(
        self,
        features: torch.Tensor,
        years: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            features: 特征向量 (B, D)
            years: 年份标签 (B,)
            labels: 身份标签 (B,)
        Returns:
            loss: 时间对比损失
        """
        batch_size = features.size(0)
        device = features.device

        # Normalize features
        features = F.normalize(features, dim=1)

        # 计算相似度矩阵
        sim_matrix = torch.matmul(features, features.T) / self.temperature

        # 创建掩码：只对同一年份的负样本计算损失
        # 正样本：同一身份、不同年份
        mask_positive = (labels.unsqueeze(1) == labels.unsqueeze(0)) & (years.unsqueeze(1) != years.unsqueeze(0))
        mask_negative = ~(labels.unsqueeze(1) == labels.unsqueeze(0))

        # 对角线掩码（去除自身）
        identity_mask = torch.eye(batch_size, device=device).bool()
        mask_positive = mask_positive & ~identity_mask

        if mask_positive.sum() == 0:
            return torch.tensor(0.0, device=device)

        # InfoNCE loss
        exp_sim = torch.exp(sim_matrix)
        log_prob = sim_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True))

        # 只对正样本计算损失
        mask_valid = mask_positive.float()
        loss = -(mask_valid * log_prob).sum() / mask_valid.sum()

        return loss


class CenterLoss(nn.Module):
    """Center loss for face recognition - centers for each identity"""

    def __init__(self, num_identities: int, feature_dim: int, size_average=True):
        super().__init__()
        self.centers = nn.Parameter(torch.randn(num_identities, feature_dim))
        self.size_average = size_average

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (batch_size, feature_dim)
            labels: (batch_size,)
        Returns:
            loss: scalar
        """
        batch_size = features.size(0)
        features_dim = features.size(1)

        # Expand centers to match batch size
        centers_batch = self.centers.index_select(0, labels.long())

        # Calculate center loss
        loss = (features - centers_batch).pow(2).sum() / batch_size

        if self.size_average:
            loss = loss / batch_size

        return loss


class TotalLoss(nn.Module):
    """组合损失函数"""

    def __init__(
        self,
        ce_weight: float = 1.0,
        triplet_weight: float = 0.5,
        tcl_weight: float = 1.0,
        center_weight: float = 0.5,
        use_mixup: bool = True,
        label_smoothing: float = 0.1
    ):
        super().__init__()
        self.ce_weight = ce_weight
        self.triplet_weight = triplet_weight
        self.tcl_weight = tcl_weight
        self.center_weight = center_weight

        # Cross entropy loss
        if label_smoothing > 0:
            self.ce_loss = LabelSmoothingCrossEntropy(smoothing=label_smoothing)
        else:
            self.ce_loss = nn.CrossEntropyLoss()

        # Triplet loss
        self.triplet_loss = TripletLoss()

        # Temporal contrastive loss
        self.tcl_loss = TemporalContrastiveLoss()

    def forward(
        self,
        logits: torch.Tensor,
        features: torch.Tensor,
        labels: torch.Tensor,
        years: Optional[torch.Tensor] = None,
        tcl_features: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            logits: 分类 logits
            features: 特征向量
            labels: 身份标签
            years: 年份标签（用于 TCL）
            tcl_features: 时间对比特征
        Returns:
            loss: 总损失
            loss_dict: 各损失分量
        """
        loss_dict = {}

        # Cross entropy loss
        ce_loss = self.ce_loss(logits, labels)
        loss_dict['ce_loss'] = ce_loss.item()
        total_loss = self.ce_weight * ce_loss

        # Triplet loss (需要构建正负样本对)
        if self.triplet_weight > 0 and features.size(0) >= 3:
            # 简化版 triplet loss
            # 随机选择 anchor, positive, negative
            batch_size = features.size(0)
            indices = torch.randperm(batch_size)[:3]
            anchor = features[indices[0]:indices[0]+1]
            positive = features[indices[1]:indices[1]+1]
            negative = features[indices[2]:indices[2]+1]

            if labels[indices[0]] == labels[indices[1]]:  # 确保 anchor 和 positive 是同一身份
                triplet_loss = self.triplet_loss(anchor, positive, negative)
                loss_dict['triplet_loss'] = triplet_loss.item()
                total_loss = total_loss + self.triplet_weight * triplet_loss

        # Temporal contrastive loss
        if self.tcl_weight > 0 and tcl_features is not None and years is not None:
            tcl_loss = self.tcl_loss(tcl_features, years, labels)
            loss_dict['tcl_loss'] = tcl_loss.item()
            total_loss = total_loss + self.tcl_weight * tcl_loss

        loss_dict['total_loss'] = total_loss.item()

        return total_loss, loss_dict


def build_loss(config) -> TotalLoss:
    """构建损失函数"""
    return TotalLoss(
        ce_weight=config.loss.ce_weight,
        triplet_weight=config.loss.triplet_weight,
        tcl_weight=config.loss.tcl_weight,
        center_weight=config.loss.center_weight,
        label_smoothing=config.train.label_smoothing
    )
