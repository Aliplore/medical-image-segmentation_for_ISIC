# classic_methods.py
import numpy as np
from skimage.filters import threshold_otsu, threshold_multiotsu
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from scipy import ndimage as ndi
from skimage.color import rgb2hsv, rgb2gray
from sklearn.cluster import KMeans
import cv2


def otsu_segmentation(image, return_prob=False):
    """
    Otsu大津阈值分割

    Args:
        image: RGB图像 (H, W, 3) 值域[0,1]
        return_prob: 是否返回概率图

    Returns:
        二值分割结果 (H, W)
    """
    # 转为灰度图
    gray = rgb2gray(image)

    # 应用Otsu
    thresh = threshold_otsu(gray)
    binary = (gray > thresh).astype(np.float32)

    if return_prob:
        return binary, gray
    return binary


def multi_otsu_segmentation(image, n_classes=3):
    """
    多阈值Otsu分割（更适合皮肤病变，因为病变区域灰度不均匀）

    Args:
        image: RGB图像 (H, W, 3) 值域[0,1]
        n_classes: 分割类别数

    Returns:
        二值分割结果 (将最亮区域作为病变)
    """
    gray = rgb2gray(image)

    # 多阈值分割
    thresholds = threshold_multiotsu(gray, classes=n_classes)
    regions = np.digitize(gray, bins=thresholds)

    # 将最高灰度区域作为病变（皮肤病变通常颜色较深或较亮，取决于类型）
    # ISIC数据集病变通常颜色较深
    binary = (regions == n_classes - 1).astype(np.float32)

    # 如果效果不好，尝试取中间区域
    if binary.sum() < gray.size * 0.05:  # 太少了
        binary = (regions >= n_classes - 2).astype(np.float32)

    return binary


def hsv_segmentation(image):
    """
    基于HSV颜色空间的分割（皮肤病变通常有独特的颜色特征）

    Args:
        image: RGB图像 (H, W, 3) 值域[0,1]

    Returns:
        二值分割结果
    """
    hsv = rgb2hsv(image)

    # 皮肤病变通常色调在红褐色范围
    # H: 0-0.1 (红色调), S: >0.3, V: 可变的
    hue = hsv[:, :, 0]
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    # 定义病变区域的颜色范围
    lesion_mask = (hue < 0.1) | (hue > 0.9)  # 红色区域
    lesion_mask = lesion_mask & (saturation > 0.3)

    # 形态学后处理
    lesion_mask = lesion_mask.astype(np.uint8)
    kernel = np.ones((5, 5), np.uint8)
    lesion_mask = cv2.morphologyEx(lesion_mask, cv2.MORPH_CLOSE, kernel)
    lesion_mask = cv2.morphologyEx(lesion_mask, cv2.MORPH_OPEN, kernel)

    return lesion_mask.astype(np.float32)


def kmeans_segmentation(image, n_clusters=2):
    """
    K-means聚类分割

    Args:
        image: RGB图像 (H, W, 3) 值域[0,1]
        n_clusters: 聚类数

    Returns:
        二值分割结果
    """
    h, w, c = image.shape
    pixels = image.reshape(-1, 3)

    # K-means聚类
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(pixels)
    labels = labels.reshape(h, w)

    # 确定哪个聚类对应病变区域
    # 通常病变区域颜色较深，取平均亮度较低的聚类
    cluster_means = []
    for i in range(n_clusters):
        cluster_pixels = pixels[labels.flatten() == i]
        if len(cluster_pixels) > 0:
            brightness = np.mean(cluster_pixels.sum(axis=1))
            cluster_means.append(brightness)

    # 取最暗的聚类作为病变（ISIC病变通常颜色深）
    lesion_cluster = np.argmin(cluster_means)
    binary = (labels == lesion_cluster).astype(np.float32)

    return binary


def watershed_segmentation(image):
    """
    分水岭分割算法

    Args:
        image: RGB图像 (H, W, 3) 值域[0,1]

    Returns:
        二值分割结果
    """
    gray = rgb2gray(image)

    # 使用Sobel边缘检测
    gradient = np.abs(np.gradient(gray))[0] + np.abs(np.gradient(gray))[1]

    # 阈值处理得到标记
    thresh = threshold_otsu(gray)
    markers = np.zeros_like(gray, dtype=np.int32)
    markers[gray < thresh * 0.8] = 1  # 背景
    markers[gray > thresh * 1.2] = 2  # 前景

    # 应用分水岭
    from skimage.segmentation import watershed
    labels = watershed(gradient, markers)

    # 取前景区域
    binary = (labels == 2).astype(np.float32)

    return binary


def combined_classic_method(image):
    """
    组合多种经典方法的投票结果

    Args:
        image: RGB图像 (H, W, 3) 值域[0,1]

    Returns:
        二值分割结果
    """
    # 获取各方法的结果
    results = []
    methods = [
        ('otsu', otsu_segmentation(image)),
        ('hsv', hsv_segmentation(image)),
        ('kmeans', kmeans_segmentation(image)),
    ]

    # 投票
    sum_mask = np.zeros_like(image[:, :, 0])
    for name, mask in methods:
        sum_mask += mask

    # 多数投票
    binary = (sum_mask >= 2).astype(np.float32)

    # 形态学后处理
    binary_uint8 = binary.astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary_uint8, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    return binary.astype(np.float32)


def evaluate_classic_method(image, mask, method_name, method_func):
    """
    评估单个经典方法

    Args:
        image: RGB图像
        mask: Ground truth
        method_name: 方法名称
        method_func: 方法函数

    Returns:
        评估指标字典
    """
    pred = method_func(image)

    # 计算指标
    pred_flat = pred.flatten()
    mask_flat = mask.flatten()

    tp = np.sum((pred_flat == 1) & (mask_flat == 1))
    fp = np.sum((pred_flat == 1) & (mask_flat == 0))
    tn = np.sum((pred_flat == 0) & (mask_flat == 0))
    fn = np.sum((pred_flat == 0) & (mask_flat == 1))

    dice = 2 * tp / (2 * tp + fp + fn + 1e-6)
    iou = tp / (tp + fp + fn + 1e-6)
    acc = (tp + tn) / (tp + tn + fp + fn + 1e-6)
    sen = tp / (tp + fn + 1e-6)
    spe = tn / (tn + fp + 1e-6)

    return {
        'method': method_name,
        'dice': dice,
        'iou': iou,
        'accuracy': acc,
        'sensitivity': sen,
        'specificity': spe
    }


# 所有经典方法字典
CLASSIC_METHODS = {
    'Otsu': otsu_segmentation,
    'Multi-Otsu': lambda x: multi_otsu_segmentation(x, n_classes=3),
    'HSV': hsv_segmentation,
    'K-means': lambda x: kmeans_segmentation(x, n_clusters=2),
    'Watershed': watershed_segmentation,
    'Combined': combined_classic_method
}