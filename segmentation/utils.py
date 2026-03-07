import torch
import numpy as np
from skimage.morphology import skeletonize
import matplotlib.pyplot as plt
import os

def dice_score(pred, target, smooth=1e-5):
    """计算Dice系数。"""
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    return (2. * intersection + smooth) / (union + smooth)

def iou_score(pred, target, smooth=1e-5):
    """计算IoU (Intersection over Union)。"""
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return (intersection + smooth) / (union + smooth)

def calculate_metrics(pred, target):
    """一次性计算所有指标。"""
    dice = dice_score(pred, target)
    iou = iou_score(pred, target)
    return {'dice': dice.item(), 'iou': iou.item()}

def calculate_biomarkers(pred_mask_np, fov_mask_np):
    """
    在视野范围内计算生物标志物。
    
    参数:
    - pred_mask_np (np.array): 二值化的预测掩码 (0或1)。
    - fov_mask_np (np.array): 二值化的视野范围掩码 (0或1)。
    
    返回:
    - dict: 包含生物标志物值的字典。
    """
    # 确保只在FOV内计算
    total_pixels_in_fov = np.sum(fov_mask_np)
    if total_pixels_in_fov == 0:
        return {'vessel_density': 0, 'skeleton_length': 0}

    vessel_pixels_in_fov = np.sum(pred_mask_np * fov_mask_np)
    
    # 1. 血管密度
    density = vessel_pixels_in_fov / total_pixels_in_fov
    
    # 2. 骨架长度
    skeleton = skeletonize(pred_mask_np * fov_mask_np)
    length = np.sum(skeleton)
    
    return {'vessel_density': density, 'skeleton_length': int(length)}

def save_comparison_plot(original_img, gt_mask, pred_mask, save_path):
    """
    保存三图对比的可视化结果。
    
    参数:
    - original_img (np.array): 原始输入图像 (H, W, C)。
    - gt_mask (np.array): 真值掩码 (H, W)。
    - pred_mask (np.array): 预测掩码 (H, W)。
    - save_path (str): 图像保存路径。
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    axes[0].imshow(original_img)
    axes[0].set_title('Input Retinal Image')
    axes[0].axis('off')
    
    axes[1].imshow(gt_mask, cmap='gray')
    axes[1].set_title('Ground Truth Mask')
    axes[1].axis('off')
    
    axes[2].imshow(pred_mask, cmap='gray')
    axes[2].set_title('Predicted Mask')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)

def unnormalize_image(tensor):
    """将标准化的Tensor图像转换回用于显示的Numpy数组。"""
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    # 反标准化
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
        
    # 从 (C, H, W) 转换为 (H, W, C)
    np_img = tensor.numpy().transpose(1, 2, 0)
    np_img = np.clip(np_img, 0, 1)
    return np_img
