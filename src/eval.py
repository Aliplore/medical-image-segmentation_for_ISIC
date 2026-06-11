# eval.py (完整替换)
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import sys
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score

sys.path.append(str(Path(__file__).parent))
from config import *
from dataset import create_dataloaders
from models import UNet, AttentionUNet
from train import calculate_metrics


def calculate_auc(model, loader, device):
    """计算AUC (像素级)"""
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, masks in tqdm(loader, desc="Computing AUC"):
            images = images.to(device)
            outputs = model(images)
            preds = torch.sigmoid(outputs).cpu().numpy().flatten()
            targets = masks.cpu().numpy().flatten()

            all_preds.extend(preds)
            all_targets.extend(targets)

    # 计算AUC
    auc_score = roc_auc_score(all_targets, all_preds)
    fpr, tpr, _ = roc_curve(all_targets, all_preds)

    return auc_score, fpr, tpr


def plot_roc_curve(fpr, tpr, auc_score, save_path):
    """绘制ROC曲线"""
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'Attention U-Net (AUC = {auc_score:.4f})')
    plt.plot([0, 1], [0, 1], 'r--', linewidth=1, label='Random (AUC = 0.5)')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve - Skin Lesion Segmentation', fontsize=14)
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def evaluate_model(model_path, model_type='attention'):
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 加载测试数据
    _, _, test_loader = create_dataloaders()

    # 创建模型
    if model_type == 'attention':
        model = AttentionUNet(n_channels=N_CHANNELS, n_classes=N_CLASSES)
    else:
        model = UNet(n_channels=N_CHANNELS, n_classes=N_CLASSES)

    # 加载权重
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    # 评估指标
    all_metrics = []

    with torch.no_grad():
        for idx, (images, masks) in enumerate(tqdm(test_loader, desc="Evaluating")):
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            metrics = calculate_metrics(outputs, masks)
            metrics['image_id'] = idx
            all_metrics.append(metrics)

    # 计算AUC
    print("\nComputing AUC...")
    auc_score, fpr, tpr = calculate_auc(model, test_loader, device)

    # 汇总统计
    metrics_df = pd.DataFrame(all_metrics)
    summary = metrics_df.drop('image_id', axis=1).agg(['mean', 'std']).round(4)

    # 添加AUC到汇总
    summary['auc'] = [auc_score, 0]  # mean, std (std设为0因为只有一个AUC值)

    print("\n" + "=" * 60)
    print("Test Set Results (mean ± std):")
    print("=" * 60)
    for metric in ['dice', 'iou', 'accuracy', 'sensitivity', 'specificity']:
        print(f"{metric.capitalize():12s}: {summary[metric]['mean']:.4f} ± {summary[metric]['std']:.4f}")
    print(f"{'AUC':12s}: {auc_score:.4f}")

    # 绘制ROC曲线
    roc_path = PRED_DIR.parent / 'roc_curve.png'
    plot_roc_curve(fpr, tpr, auc_score, roc_path)
    print(f"\nROC curve saved to: {roc_path}")

    # 保存结果
    metrics_df.to_csv(METRICS_PATH, index=False)

    # 保存汇总结果
    summary.to_csv(METRICS_PATH.parent / 'summary_metrics.csv')

    # 保存详细结果JSON
    import json
    results = {
        'model_type': model_type,
        'test_size': len(test_loader.dataset),
        'metrics': {
            'dice': {'mean': float(summary['dice']['mean']), 'std': float(summary['dice']['std'])},
            'iou': {'mean': float(summary['iou']['mean']), 'std': float(summary['iou']['std'])},
            'accuracy': {'mean': float(summary['accuracy']['mean']), 'std': float(summary['accuracy']['std'])},
            'sensitivity': {'mean': float(summary['sensitivity']['mean']), 'std': float(summary['sensitivity']['std'])},
            'specificity': {'mean': float(summary['specificity']['mean']), 'std': float(summary['specificity']['std'])},
            'auc': auc_score
        }
    }
    with open(CHECKPOINT_DIR / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    return metrics_df, summary, auc_score


if __name__ == "__main__":
    # 评估最佳模型
    model_path = CHECKPOINT_DIR / 'best_model.pth'
    if model_path.exists():
        evaluate_model(model_path, model_type='attention')
    else:
        print("No trained model found. Please run train.py first.")