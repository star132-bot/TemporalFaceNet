# Cross Temporal Face Recognition
# Data Augmentation Module

import torch
import torchvision.transforms as T
import numpy as np
from typing import Optional, List, Tuple


class TemporalAugmentation:
    """时间感知的数据增强 - 模拟时间变化"""

    def __init__(self, image_size: int = 224):
        self.image_size = image_size

    def get_train_transform(self) -> T.Compose:
        """训练时数据增强"""
        return T.Compose([
            T.Resize((self.image_size, self.image_size)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=15),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            T.RandomGrayscale(p=0.1),
            T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def get_test_transform(self) -> T.Compose:
        """测试时数据增强"""
        return T.Compose([
            T.Resize((self.image_size, self.image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def get_augmentations_transform(self):
        """更多样化的增强（使用 torchvision）"""
        return T.Compose([
            T.Resize((self.image_size, self.image_size)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=15),
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            T.RandomGrayscale(p=0.15),
            T.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
            T.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])


class MixupCutmix:
    """Mixup 和 Cutmix 增强"""

    def __init__(self, mixup_alpha: float = 0.2, cutmix_alpha: float = 1.0, prob: float = 0.5):
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.prob = prob

    def __call__(
        self,
        batch_images: torch.Tensor,
        batch_labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """
        Returns: mixed_images, labels_a, labels_b, lambda
        """
        if np.random.rand() > self.prob:
            return batch_images, batch_labels, batch_labels, 1.0

        # Randomly choose mixup or cutmix
        if np.random.rand() < 0.5:
            return self.mixup(batch_images, batch_labels)
        else:
            return self.cutmix(batch_images, batch_labels)

    def mixup(self, x: torch.Tensor, y: torch.Tensor):
        """Mixup augmentation"""
        lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(x.device)

        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam

    def cutmix(self, x: torch.Tensor, y: torch.Tensor):
        """Cutmix augmentation"""
        lam = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(x.device)

        # Generate random bounding box
        W, H = x.size(2), x.size(3)
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)

        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        mixed_x = x.clone()
        mixed_x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]

        # Adjust lambda to exactly match pixel ratio
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))

        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam


class TimeSimulationAugmentation:
    """时间模拟增强 - 模拟老化/年轻化"""

    def __init__(self, image_size: int = 224):
        self.image_size = image_size

    def simulate_aging(self, image: torch.Tensor, age_delta: int) -> torch.Tensor:
        """
        模拟老化效果（简化版）
        age_delta: 正值表示老化，负值表示年轻化
        """
        # 简化的老化模拟：通过调整对比度和添加噪点
        factor = 1.0 + age_delta * 0.01

        # Adjust brightness and contrast
        image = image * factor
        image = torch.clamp(image, 0, 1)

        # Add slight noise to simulate skin texture changes
        if abs(age_delta) > 5:
            noise = torch.randn_like(image) * (abs(age_delta) * 0.005)
            image = image + noise
            image = torch.clamp(image, 0, 1)

        return image

    def get_time_transform(self, year_diff: int) -> T.Compose:
        """根据年份差异应用时间变换"""
        return T.Compose([
            T.Resize((self.image_size, self.image_size)),
            T.Lambda(lambda x: self.simulate_aging(x, year_diff)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])


def build_transforms(
    image_size: int = 224,
    is_training: bool = True,
    use_albumentations: bool = False
) -> T.Compose:
    """构建数据增强管道"""
    if use_albumentations:
        aug = TemporalAugmentation(image_size)
        return aug.get_augmentations_transform() if is_training else aug.get_test_transform()
    else:
        aug = TemporalAugmentation(image_size)
        return aug.get_train_transform() if is_training else aug.get_test_transform()
