# compare_methods.py
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from tqdm import tqdm
import cv2

sys.path.append(str(Path(__file__).parent))
from config import *
from dataset import create_dataloaders
from models import AttentionUNet
from classic_methods import CLASSIC_METHODS, evaluate_classic_method
from train import calculate_metrics


def compare_all_methods(model_path, num_test_samples=None):
    """
    对比所有方法的性能

    Args:
        model_path: 深度学习模型路径
        num_test_samples: 使用多少个测试样本，None表示全部
    """
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")

    # 加载数据
    _, _, test_loader = create_dataloaders()
    test_dataset = test_loader.dataset

    if num_test_samples is None:
        num_test_samples = len(test_dataset)

    print(f"\n{'=' * 60}")
    print(f"Comparing methods on {num_test_samples} test samples")
    print(f"{'=' * 60}\n")

    # 加载深度学习模型
    model = AttentionUNet(n_channels=N_CHANNELS, n_classes=N_CLASSES)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    # 存储结果
    results = {method: [] for method in CLASSIC_METHODS.keys()}
    results['Deep Learning (Attention U-Net)'] = []

    # 逐样本评估
    for idx in tqdm(range(min(num_test_samples, len(test_dataset))), desc="Comparing methods"):
        image, mask = test_dataset[idx]
        img_np = image.cpu().permute(1, 2, 0).numpy()
        mask_np = mask.cpu().squeeze().numpy()

        # 评估经典方法
        for method_name, method_func in CLASSIC_METHODS.items():
            metrics = evaluate_classic_method(img_np, mask_np, method_name, method_func)
            results[method_name].append(metrics)

        # 评估深度学习方法
        with torch.no_grad():
            image_tensor = image.unsqueeze(0).to(device)
            output = model(image_tensor)
            dl_metrics = calculate_metrics(output.cpu(), mask.unsqueeze(0))
            dl_metrics['method'] = 'Deep Learning'
            results['Deep Learning (Attention U-Net)'].append(dl_metrics)

    # 汇总统计
    summary_data = []
    for method_name, method_results in results.items():
        if not method_results:
            continue

        df = pd.DataFrame(method_results)
        summary_data.append({
            'Method': method_name,
            'Dice': f"{df['dice'].mean():.4f} ± {df['dice'].std():.4f}",
            'IoU': f"{df['iou'].mean():.4f} ± {df['iou'].std():.4f}",
            'Accuracy': f"{df['accuracy'].mean():.4f} ± {df['accuracy'].std():.4f}",
            'Sensitivity': f"{df['sensitivity'].mean():.4f} ± {df['sensitivity'].std():.4f}",
            'Specificity': f"{df['specificity'].mean():.4f} ± {df['specificity'].std():.4f}",
            'Dice_mean': df['dice'].mean(),
            'Dice_std': df['dice'].std()
        })

    # 按Dice排序
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values('Dice_mean', ascending=False)
    summary_df = summary_df.drop('Dice_mean', axis=1)

    print("\n" + "=" * 80)
    print("COMPARISON RESULTS (mean ± std)")
    print("=" * 80)
    print(summary_df.to_string(index=False))

    # 保存结果
    summary_df.to_csv(PRED_DIR.parent / 'method_comparison.csv', index=False)

    # 绘制对比柱状图
    plot_comparison_bar(summary_df)

    return summary_df, results


def plot_comparison_bar(summary_df):
    """绘制方法对比柱状图"""
    methods = summary_df['Method'].tolist()
    dice_values = [float(x.split(' ± ')[0]) for x in summary_df['Dice'].tolist()]
    dice_stds = [float(x.split(' ± ')[1]) for x in summary_df['Dice'].tolist()]

    fig, ax = plt.subplots(figsize=(12, 6))

    colors = ['#ff7f7f' if 'Deep' in m else '#7f7fff' for m in methods]
    bars = ax.bar(range(len(methods)), dice_values, yerr=dice_stds,
                  capsize=5, color=colors, alpha=0.7)

    ax.set_xlabel('Method', fontsize=12)
    ax.set_ylabel('Dice Score', fontsize=12)
    ax.set_title('Comparison of Different Segmentation Methods', fontsize=14)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=45, ha='right', fontsize=10)
    ax.set_ylim(0, 1)
    ax.axhline(y=0.8, color='g', linestyle='--', alpha=0.5, label='Good threshold (0.8)')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # 添加数值标签
    for i, (v, err) in enumerate(zip(dice_values, dice_stds)):
        ax.text(i, v + err + 0.02, f'{v:.3f}', ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(PRED_DIR.parent / 'method_comparison_bar.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nComparison bar chart saved to: {PRED_DIR.parent / 'method_comparison_bar.png'}")


def plot_side_by_side_comparison(model_path, num_samples=3):
    """并排展示多种方法的预测结果"""
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")

    _, _, test_loader = create_dataloaders()
    test_dataset = test_loader.dataset

    # 加载深度学习模型
    model = AttentionUNet(n_channels=N_CHANNELS, n_classes=N_CLASSES)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    # 选择前num_samples个样本
    selected_indices = list(range(min(num_samples, len(test_dataset))))

    # 选择要对比的方法
    methods_to_show = ['Otsu', 'HSV', 'K-means', 'Combined']

    fig, axes = plt.subplots(num_samples, len(methods_to_show) + 2,
                             figsize=(18, 4 * num_samples))

    for row, idx in enumerate(selected_indices):
        image, mask = test_dataset[idx]
        img_np = image.cpu().permute(1, 2, 0).numpy()
        mask_np = mask.cpu().squeeze().numpy()

        # 显示原图
        axes[row, 0].imshow(img_np)
        axes[row, 0].set_title('Original', fontsize=10)
        axes[row, 0].axis('off')

        # 显示Ground Truth
        axes[row, 1].imshow(mask_np, cmap='gray')
        axes[row, 1].set_title('Ground Truth', fontsize=10)
        axes[row, 1].axis('off')

        # 显示各个经典方法的结果
        for col, method_name in enumerate(methods_to_show):
            method_func = CLASSIC_METHODS[method_name]
            pred = method_func(img_np)
            axes[row, col + 2].imshow(pred, cmap='gray')

            # 计算Dice显示在标题上
            pred_flat = pred.flatten()
            mask_flat = mask_np.flatten()
            tp = np.sum((pred_flat == 1) & (mask_flat == 1))
            fp = np.sum((pred_flat == 1) & (mask_flat == 0))
            fn = np.sum((pred_flat == 0) & (mask_flat == 1))
            dice = 2 * tp / (2 * tp + fp + fn + 1e-6)

            axes[row, col + 2].set_title(f'{method_name}\nDice: {dice:.3f}', fontsize=9)
            axes[row, col + 2].axis('off')

    plt.tight_layout()
    plt.savefig(PRED_DIR.parent / 'side_by_side_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nSide-by-side comparison saved to: {PRED_DIR.parent / 'side_by_side_comparison.png'}")


def analyze_failure_cases(model_path, num_failures=3):
    """分析失败案例"""
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")

    _, _, test_loader = create_dataloaders()
    test_dataset = test_loader.dataset

    # 加载模型
    model = AttentionUNet(n_channels=N_CHANNELS, n_classes=N_CLASSES)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    # 计算所有样本的Dice分数
    results = []
    with torch.no_grad():
        for idx in range(len(test_dataset)):
            image, mask = test_dataset[idx]
            image_tensor = image.unsqueeze(0).to(device)
            output = model(image_tensor)
            metrics = calculate_metrics(output.cpu(), mask.unsqueeze(0))
            results.append((idx, metrics['dice']))

    # 按Dice排序，找出最差的
    results.sort(key=lambda x: x[1])
    worst_samples = results[:num_failures]

    # 可视化失败案例
    fig, axes = plt.subplots(num_failures, 4, figsize=(16, 4 * num_failures))

    for row, (idx, dice_score) in enumerate(worst_samples):
        image, mask = test_dataset[idx]
        img_np = image.cpu().permute(1, 2, 0).numpy()
        mask_np = mask.cpu().squeeze().numpy()

        # 预测
        image_tensor = image.unsqueeze(0).to(device)
        output = model(image_tensor)
        pred = torch.sigmoid(output).detach().cpu().numpy()[0, 0]
        pred_binary = (pred > 0.5).astype(np.float32)

        # 误差图
        error = np.abs(pred_binary - mask_np)

        # 显示
        axes[row, 0].imshow(img_np)
        axes[row, 0].set_title(f'Original (ID: {idx})', fontsize=10)
        axes[row, 0].axis('off')

        axes[row, 1].imshow(mask_np, cmap='gray')
        axes[row, 1].set_title('Ground Truth', fontsize=10)
        axes[row, 1].axis('off')

        axes[row, 2].imshow(pred_binary, cmap='gray')
        axes[row, 2].set_title(f'Prediction (Dice: {dice_score:.3f})', fontsize=10)
        axes[row, 2].axis('off')

        axes[row, 3].imshow(error, cmap='Reds', vmin=0, vmax=1)
        axes[row, 3].set_title('Error Map (Red=Wrong)', fontsize=10)
        axes[row, 3].axis('off')

    plt.suptitle('Failure Case Analysis', fontsize=14)
    plt.tight_layout()
    plt.savefig(PRED_DIR.parent / 'failure_cases.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nFailure cases analysis saved to: {PRED_DIR.parent / 'failure_cases.png'}")
    print("\nWorst performing samples (lowest Dice):")
    for idx, dice in worst_samples:
        print(f"  Sample {idx}: Dice = {dice:.4f}")

    return worst_samples


if __name__ == "__main__":
    model_path = CHECKPOINT_DIR / 'best_model.pth'

    if model_path.exists():
        # 1. 横向对比所有方法
        print("\n>>> Running method comparison...")
        compare_all_methods(model_path)

        # 2. 并排可视化对比
        print("\n>>> Generating side-by-side comparison...")
        plot_side_by_side_comparison(model_path, num_samples=3)

        # 3. 分析失败案例
        print("\n>>> Analyzing failure cases...")
        analyze_failure_cases(model_path, num_failures=3)

        print("\n>>> All comparisons completed!")
    else:
        print("No trained model found. Please run train.py first.")