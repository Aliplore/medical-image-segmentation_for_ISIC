import torch
from torch.utils.data import Dataset, DataLoader, random_split
import cv2
import numpy as np
from pathlib import Path
import albumentations as A
from config import *


class ISICDataset(Dataset):
    """ISIC皮肤病变数据集"""

    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.transform = transform

        # 获取所有图像文件（确保图像和掩码匹配）
        self.image_paths = sorted(list(self.images_dir.glob("*.jpg")))
        self.mask_paths = []
        for img_path in self.image_paths:
            mask_name = img_path.stem + "_segmentation.png"
            mask_path = self.masks_dir / mask_name
            if mask_path.exists():
                self.mask_paths.append(mask_path)
            else:
                print(f"Warning: Mask not found for {img_path.name}")

        print(f"Loaded {len(self.image_paths)} image-mask pairs")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 读取图像
        image = cv2.imread(str(self.image_paths[idx]))
        if image is None:
            raise ValueError(f"Could not load image: {self.image_paths[idx]}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 读取掩码（ISIC的掩码是RGB彩色图，需要转换为单通道二值图）
        mask = cv2.imread(str(self.mask_paths[idx]), 0)  # 灰度读取
        if mask is None:
            raise ValueError(f"Could not load mask: {self.mask_paths[idx]}")
        mask = (mask > 128).astype(np.float32)  # 二值化

        # 应用数据增强
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']

        # 转换为tensor
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        mask = torch.from_numpy(mask).unsqueeze(0).float()

        return image, mask


def get_transforms(phase='train'):
    """获取数据增强策略"""
    if phase == 'train':
        return A.Compose([
            A.Resize(*IMG_SIZE),
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),  # 修复：Flip改为HorizontalFlip
            A.VerticalFlip(p=0.5),  # 添加垂直翻转
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5),
            A.OneOf([
                A.CLAHE(clip_limit=2, p=0.5),
                A.RandomBrightnessContrast(p=0.5),
            ], p=0.5),
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),  # 修复：指定参数
                A.MultiplicativeNoise(multiplier=(0.9, 1.1), p=0.3),  # 修复：指定参数
            ], p=0.3),
        ])
    else:  # validation/test
        return A.Compose([
            A.Resize(*IMG_SIZE),
        ])


def create_dataloaders():
    """创建训练、验证、测试数据加载器"""

    # 创建完整训练集
    full_train_dataset = ISICDataset(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        transform=get_transforms('train')
    )

    # 划分训练集和验证集
    val_size = int(len(full_train_dataset) * VAL_RATIO)
    train_size = len(full_train_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED)
    )

    # 为验证集设置不同的transform（不做数据增强）
    # 注意：需要重新设置验证集的transform
    val_dataset.dataset.transform = get_transforms('val')

    # 创建测试集
    test_dataset = ISICDataset(
        TEST_IMG_DIR,
        TEST_MASK_DIR,
        transform=get_transforms('val')
    )

    # 创建DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,  # Windows下建议设为0，避免多进程问题
        pin_memory=True if torch.cuda.is_available() else False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,  # 测试时batch_size=1方便可视化
        shuffle=False,
        num_workers=0
    )

    print(f"Train size: {len(train_dataset)}")
    print(f"Val size: {len(val_dataset)}")
    print(f"Test size: {len(test_dataset)}")

    return train_loader, val_loader, test_loader