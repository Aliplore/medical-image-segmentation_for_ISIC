import os
from pathlib import Path

# 路径配置
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = "./ISIC_new"

TRAIN_IMG_DIR = "../ISIC_new/train/images"
TRAIN_MASK_DIR = "../ISIC_new/train/annotations"
TEST_IMG_DIR = "../ISIC_new/test/images"
TEST_MASK_DIR = "../ISIC_new/test/annotations"

# 训练配置
IMG_SIZE = (256, 256)  # 统一图像尺寸
BATCH_SIZE = 16
NUM_EPOCHS = 100
LEARNING_RATE = 1e-4
EARLY_STOPPING_PATIENCE = 15

# 数据划分比例 (从训练集中分出验证集)
VAL_RATIO = 0.15

# 模型配置
N_CLASSES = 1  # 二分类分割
N_CHANNELS = 3  # RGB图像

# 设备配置
DEVICE = "cuda"  # 或 "cpu"

# 随机种子
SEED = 42

# 保存路径
CHECKPOINT_DIR = BASE_DIR / "results" / "checkpoints"
PRED_DIR = BASE_DIR / "results" / "predictions"
METRICS_PATH = BASE_DIR / "results" / "metrics.csv"

# 创建目录
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
PRED_DIR.mkdir(parents=True, exist_ok=True)