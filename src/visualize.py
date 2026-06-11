import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys
import random

sys.path.append(str(Path(__file__).parent))
from config import *
from dataset import create_dataloaders
from models import AttentionUNet
from train import calculate_metrics


def visualize_predictions(model_path, num_samples=5, save_dir=None):
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")

    # 加载测试数据
    _, _, test_loader = create_dataloaders()

    # 加载模型
    model = AttentionUNet(n_channels=N_CHANNELS, n_classes=N_CLASSES)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    # 随机选择样本
    all_indices = list(range(len(test_loader.dataset)))
    selected_indices = random.sample(all_indices, min(num_samples, len(all_indices)))

    if save_dir is None:
        save_dir = PRED_DIR / 'visualizations'
    save_dir.mkdir(parents=True, exist_ok=True)

    metrics_list = []

    with torch.no_grad():
        for img_idx in selected_indices:
            # 获取单张图像
            image, mask = test_loader.dataset[img_idx]
            image_tensor = image.unsqueeze(0).to(device)

            # 预测
            output = model(image_tensor)
            pred = torch.sigmoid(output).cpu().numpy()[0, 0]
            pred_binary = (pred > 0.5).astype(np.float32)

            # 计算指标
            metrics = calculate_metrics(output.cpu(), mask.unsqueeze(0))
            metrics_list.append(metrics)

            # 可视化
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))

            # 原始图像
            img_np = image.cpu().permute(1, 2, 0).numpy()
            axes[0, 0].imshow(img_np)
            axes[0, 0].set_title('Original Image')
            axes[0, 0].axis('off')

            # Ground Truth
            mask_np = mask.cpu().squeeze().numpy()
            axes[0, 1].imshow(mask_np, cmap='gray')
            axes[0, 1].set_title('Ground Truth')
            axes[0, 1].axis('off')

            # 预测概率图
            axes[0, 2].imshow(pred, cmap='hot')
            axes[0, 2].set_title('Prediction (Probability)')
            axes[0, 2].axis('off')

            # 二值预测
            axes[1, 0].imshow(pred_binary, cmap='gray')
            axes[1, 0].set_title(f'Binary Prediction (Dice: {metrics["dice"]:.3f})')
            axes[1, 0].axis('off')

            # 误差图
            error = np.abs(pred_binary - mask_np)
            axes[1, 1].imshow(error, cmap='Reds', vmin=0, vmax=1)
            axes[1, 1].set_title('Error Map (Red=Wrong)')
            axes[1, 1].axis('off')

            # 叠加显示
            overlay = img_np.copy()
            overlay[pred_binary > 0] = [255, 0, 0]  # 预测区域标红
            axes[1, 2].imshow(overlay)
            axes[1, 2].set_title('Prediction Overlay (Red=Predicted)')
            axes[1, 2].axis('off')

            plt.tight_layout()
            plt.savefig(save_dir / f'sample_{img_idx}_dice_{metrics["dice"]:.3f}.png', dpi=150, bbox_inches='tight')
            plt.close()

            print(f"Sample {img_idx}: Dice={metrics['dice']:.4f}, IoU={metrics['iou']:.4f}")

    # 统计
    print("\n" + "=" * 50)
    print("Visualized Samples Statistics:")
    print("=" * 50)
    for metric in ['dice', 'iou', 'accuracy']:
        values = [m[metric] for m in metrics_list]
        print(f"{metric.capitalize():12s}: {np.mean(values):.4f} ± {np.std(values):.4f}")


def compare_methods(model_path, otsu_path=None, num_samples=3):
    """对比不同方法"""
    # 实现对比Otsu方法
    from skimage.filters import threshold_otsu

    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    _, _, test_loader = create_dataloaders()

    model = AttentionUNet(n_channels=N_CHANNELS, n_classes=N_CLASSES)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    # 选择样本
    selected_indices = list(range(min(num_samples, len(test_loader.dataset))))

    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4 * num_samples))

    with torch.no_grad():
        for row, img_idx in enumerate(selected_indices):
            image, mask = test_loader.dataset[img_idx]

            # Otsu方法
            img_np = image.cpu().permute(1, 2, 0).numpy()
            gray = np.mean(img_np, axis=2)
            thresh = threshold_otsu(gray)
            otsu_pred = (gray > thresh).astype(np.float32)

            # 深度学习预测
            image_tensor = image.unsqueeze(0).to(device)
            output = model(image_tensor)
            dl_pred = (torch.sigmoid(output).cpu().numpy()[0, 0] > 0.5).astype(np.float32)

            # 显示
            axes[row, 0].imshow(img_np)
            axes[row, 0].set_title('Original')
            axes[row, 0].axis('off')

            axes[row, 1].imshow(mask.cpu().squeeze().numpy(), cmap='gray')
            axes[row, 1].set_title('Ground Truth')
            axes[row, 1].axis('off')

            axes[row, 2].imshow(otsu_pred, cmap='gray')
            axes[row, 2].set_title('Otsu Threshold')
            axes[row, 2].axis('off')

            axes[row, 3].imshow(dl_pred, cmap='gray')
            axes[row, 3].set_title('Deep Learning (U-Net+Attention)')
            axes[row, 3].axis('off')

    plt.tight_layout()
    plt.savefig(PRED_DIR / 'method_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    model_path = CHECKPOINT_DIR / 'best_model.pth'
    if model_path.exists():
        # 可视化预测结果
        visualize_predictions(model_path, num_samples=5)
        # 对比不同方法
        compare_methods(model_path, num_samples=3)
    else:
        print("No trained model found. Please run train.py first.")