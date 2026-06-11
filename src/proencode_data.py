import os
import shutil
from pathlib import Path
import re
import random
from sklearn.model_selection import train_test_split

def organize_isic_dataset(source_dir, target_dir, test_ratio=0.2, random_seed=42):
    """
    从ISIC训练数据中按比例划分训练集和测试集
    
    Args:
        source_dir: 源数据目录路径 (/media/hpc/???/python_project/hzh/Datasets/ISIC)
        target_dir: 目标数据目录路径 (/media/hpc/???/python_project/hzh/Datasets/ISIC_new)
        test_ratio: 测试集比例，默认0.2 (20%)
        random_seed: 随机种子，保证可重复性
    """
    
    # 设置随机种子
    random.seed(random_seed)
    
    # 定义源目录路径
    source_path = Path(source_dir)
    training_images_dir = source_path / "ISIC2018_Task1-2_Training_Input"
    training_annotations_dir = source_path / "ISIC2018_Task1_Training_GroundTruth"
    
    # 检查目录是否存在
    if not training_images_dir.exists():
        raise FileNotFoundError(f"找不到目录: {training_images_dir}")
    if not training_annotations_dir.exists():
        raise FileNotFoundError(f"找不到目录: {training_annotations_dir}")
    
    # 定义目标目录路径
    target_path = Path(target_dir)
    train_images_dir = target_path / "train" / "images"
    train_annotations_dir = target_path / "train" / "annotations"
    test_images_dir = target_path / "test" / "images"
    test_annotations_dir = target_path / "test" / "annotations"
    
    # 创建目标目录
    train_images_dir.mkdir(parents=True, exist_ok=True)
    train_annotations_dir.mkdir(parents=True, exist_ok=True)
    test_images_dir.mkdir(parents=True, exist_ok=True)
    test_annotations_dir.mkdir(parents=True, exist_ok=True)
    
    # 获取所有图片和对应的标注文件
    all_images = list(training_images_dir.glob("ISIC_*.jpg"))
    
    # 筛选出有对应标注文件的图片
    valid_pairs = []
    for img_path in all_images:
        # 提取ISIC编号
        match = re.match(r"ISIC_(\d+)\.jpg", img_path.name)
        if match:
            isic_id = match.group(1)
            annotation_name = f"ISIC_{isic_id}_segmentation.png"
            annotation_path = training_annotations_dir / annotation_name
            
            if annotation_path.exists():
                valid_pairs.append((img_path, annotation_path))
            else:
                print(f"警告: 找不到标注文件 {annotation_name}，跳过图片 {img_path.name}")
    
    print(f"找到 {len(valid_pairs)} 对有效的图片-标注文件")
    
    # 按比例划分训练集和测试集
    image_paths = [pair[0] for pair in valid_pairs]
    annotation_paths = [pair[1] for pair in valid_pairs]
    
    train_images, test_images, train_annotations, test_annotations = train_test_split(
        image_paths, annotation_paths, 
        test_size=test_ratio, 
        random_state=random_seed,
        shuffle=True
    )
    
    print(f"训练集: {len(train_images)} 对")
    print(f"测试集: {len(test_images)} 对")
    
    # 复制训练集文件
    print("正在复制训练集...")
    for img_path, ann_path in zip(train_images, train_annotations):
        # 复制图片
        dest_img = train_images_dir / img_path.name
        shutil.copy2(img_path, dest_img)
        
        # 复制标注
        dest_ann = train_annotations_dir / ann_path.name
        shutil.copy2(ann_path, dest_ann)
    
    # 复制测试集文件
    print("正在复制测试集...")
    for img_path, ann_path in zip(test_images, test_annotations):
        # 复制图片
        dest_img = test_images_dir / img_path.name
        shutil.copy2(img_path, dest_img)
        
        # 复制标注
        dest_ann = test_annotations_dir / ann_path.name
        shutil.copy2(ann_path, dest_ann)
    
    # 验证结果
    train_img_count = len(list(train_images_dir.glob("*.jpg")))
    train_ann_count = len(list(train_annotations_dir.glob("*.png")))
    test_img_count = len(list(test_images_dir.glob("*.jpg")))
    test_ann_count = len(list(test_annotations_dir.glob("*.png")))
    
    print("\n" + "="*50)
    print("数据集划分完成！")
    print(f"训练集图片: {train_img_count} 张")
    print(f"训练集标注: {train_ann_count} 个")
    print(f"测试集图片: {test_img_count} 张")
    print(f"测试集标注: {test_ann_count} 个")
    print(f"目标目录: {target_path}")
    print("="*50)
    
    # 可选：打印几个示例文件名
    print("\n示例文件:")
    print(f"训练集图片示例: {train_images[0].name if train_images else 'None'}")
    print(f"测试集图片示例: {test_images[0].name if test_images else 'None'}")

if __name__ == "__main__":
    # 配置路径
    source_directory = "./Datasets/ISIC"
    target_directory = "./Datasets/ISIC_new"
    
    # 运行数据整理
    # test_ratio: 测试集比例，可以修改为0.1 (10%)、0.3 (30%)等
    organize_isic_dataset(
        source_dir=source_directory,
        target_dir=target_directory,
        test_ratio=0.2,  # 20%作为测试集
        random_seed=42   # 固定随机种子，确保每次划分一致
    )