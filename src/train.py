import torch
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))
from config import *
from dataset import create_dataloaders
from models import UNet, AttentionUNet, CombinedLoss


def calculate_metrics(pred, target, threshold=0.5):
    """计算各种评估指标"""
    pred_binary = (torch.sigmoid(pred) > threshold).float()

    # 展平
    pred_flat = pred_binary.view(-1)
    target_flat = target.view(-1)

    # 计算TP, FP, TN, FN
    tp = (pred_flat * target_flat).sum().float()
    fp = (pred_flat * (1 - target_flat)).sum().float()
    tn = ((1 - pred_flat) * (1 - target_flat)).sum().float()
    fn = ((1 - pred_flat) * target_flat).sum().float()

    # 计算指标
    iou = tp / (tp + fp + fn + 1e-6)
    dice = 2 * tp / (2 * tp + fp + fn + 1e-6)
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-6)
    sensitivity = tp / (tp + fn + 1e-6)  # Recall
    specificity = tn / (tn + fp + 1e-6)

    return {
        'dice': dice.item(),
        'iou': iou.item(),
        'accuracy': accuracy.item(),
        'sensitivity': sensitivity.item(),
        'specificity': specificity.item()
    }


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    metrics = {'dice': 0, 'iou': 0, 'accuracy': 0, 'sensitivity': 0, 'specificity': 0}

    pbar = tqdm(loader, desc='Training')
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # 计算指标
        with torch.no_grad():
            batch_metrics = calculate_metrics(outputs, masks)
            for k, v in batch_metrics.items():
                metrics[k] += v

        pbar.set_postfix({'loss': loss.item()})

    # 平均指标
    n_batches = len(loader)
    for k in metrics:
        metrics[k] /= n_batches

    return total_loss / n_batches, metrics


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    metrics = {'dice': 0, 'iou': 0, 'accuracy': 0, 'sensitivity': 0, 'specificity': 0}

    with torch.no_grad():
        pbar = tqdm(loader, desc='Validation')
        for images, masks in pbar:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)
            total_loss += loss.item()

            batch_metrics = calculate_metrics(outputs, masks)
            for k, v in batch_metrics.items():
                metrics[k] += v

    n_batches = len(loader)
    for k in metrics:
        metrics[k] /= n_batches

    return total_loss / n_batches, metrics


def train():
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 创建数据加载器
    train_loader, val_loader, _ = create_dataloaders()

    # 创建模型（可以选择UNet或AttentionUNet）
    model = AttentionUNet(n_channels=N_CHANNELS, n_classes=N_CLASSES).to(device)
    # model = UNet(n_channels=N_CHANNELS, n_classes=N_CLASSES).to(device)

    # 损失函数和优化器
    criterion = CombinedLoss(weight_bce=1.0, weight_dice=1.0)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

    # 训练记录
    history = {
        'train_loss': [], 'val_loss': [],
        'train_dice': [], 'val_dice': [],
        'val_iou': [], 'val_acc': []
    }

    best_val_dice = 0
    patience_counter = 0

    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")

        # 训练
        train_loss, train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_metrics = validate(model, val_loader, criterion, device)

        # 更新学习率
        scheduler.step(val_loss)

        # 记录
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_dice'].append(train_metrics['dice'])
        history['val_dice'].append(val_metrics['dice'])
        history['val_iou'].append(val_metrics['iou'])
        history['val_acc'].append(val_metrics['accuracy'])

        print(f"Train Loss: {train_loss:.4f}, Train Dice: {train_metrics['dice']:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Dice: {val_metrics['dice']:.4f}, Val IoU: {val_metrics['iou']:.4f}")

        # 保存最佳模型
        if val_metrics['dice'] > best_val_dice:
            best_val_dice = val_metrics['dice']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_dice': val_metrics['dice'],
            }, CHECKPOINT_DIR / 'best_model.pth')
            print(f"Saved best model with Dice: {best_val_dice:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1

        # 早停
        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"Early stopping after {epoch + 1} epochs")
            break

    # 保存训练历史
    history_df = pd.DataFrame(history)
    history_df.to_csv(CHECKPOINT_DIR / 'training_history.csv')

    return model, history


if __name__ == "__main__":
    train()