import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import argparse

# 导入我们自己的模块
from data_loader import DRIVEDataset, get_transforms
from model import build_model
from utils import calculate_metrics, calculate_biomarkers, save_comparison_plot, unnormalize_image

def get_latest_model_path(output_dir):
    """在输出目录中查找最新的模型文件路径。"""
    try:
        subdirs = [d for d in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, d))]
        if not subdirs: return None
        latest_dir = sorted(subdirs)[-1]
        model_path = os.path.join(output_dir, latest_dir, 'best_model.pth')
        return model_path if os.path.exists(model_path) else None
    except FileNotFoundError:
        return None

def evaluate(config):
    """主评估函数。"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # --- 确定模型路径 ---
    model_path = config['model_path']
    if not model_path:
        print("未指定模型路径，正在查找最新模型...")
        model_path = get_latest_model_path(config['output_dir'])
        if not model_path:
            print(f"错误: 在 '{config['output_dir']}' 中找不到任何模型。请先运行训练。")
            return
    print(f"将使用模型: {model_path}")

    # --- 加载模型 ---
    model = build_model().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # --- 加载测试数据 ---
    transforms = get_transforms(config['image_size'])
    test_dataset = DRIVEDataset(root_dir=config['data_dir'], transform=transforms, subset='test')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    # --- 创建结果保存目录 ---
    results_dir = os.path.join(os.path.dirname(model_path), 'test_results')
    os.makedirs(results_dir, exist_ok=True)
    print(f"评估结果将保存在: {results_dir}")

    # --- 评估循环 ---
    total_metrics = {'dice': 0.0, 'iou': 0.0}
    total_biomarkers = {'vessel_density': 0.0, 'skeleton_length': 0.0}

    for i, batch in enumerate(tqdm(test_loader, desc="正在评估测试集")):
        image = batch['image'].to(device)
        gt_mask = batch['mask'].to(device)
        fov_mask = batch['fov_mask'] # 在CPU上处理

        with torch.no_grad():
            pred_mask = model(image)

        # 计算指标
        metrics = calculate_metrics(pred_mask, gt_mask)
        total_metrics['dice'] += metrics['dice']
        total_metrics['iou'] += metrics['iou']

        # --- CPU端处理，用于可视化和生物标志物计算 ---
        # 将Tensor转为Numpy数组
        original_img_np = unnormalize_image(batch['image'].squeeze(0).cpu())
        gt_mask_np = gt_mask.squeeze().cpu().numpy()
        pred_mask_np = (pred_mask.squeeze().cpu().numpy() > 0.5).astype(np.uint8)
        fov_mask_np = (fov_mask.squeeze().cpu().numpy() > 0.5).astype(np.uint8)

        # 计算生物标志物
        biomarkers = calculate_biomarkers(pred_mask_np, fov_mask_np)
        total_biomarkers['vessel_density'] += biomarkers['vessel_density']
        total_biomarkers['skeleton_length'] += biomarkers['skeleton_length']
        
        print(f"\n图片 {i+1}: Dice={metrics['dice']:.4f}, IoU={metrics['iou']:.4f}, "
              f"密度={biomarkers['vessel_density']:.4f}, 长度={biomarkers['skeleton_length']}")

        # 保存三图对比
        plot_path = os.path.join(results_dir, f'comparison_{i+1:02d}.png')
        save_comparison_plot(original_img_np, gt_mask_np, pred_mask_np, plot_path)

    # --- 打印最终平均结果 ---
    num_samples = len(test_loader)
    avg_dice = total_metrics['dice'] / num_samples
    avg_iou = total_metrics['iou'] / num_samples
    avg_density = total_biomarkers['vessel_density'] / num_samples
    avg_length = total_biomarkers['skeleton_length'] / num_samples

    print("\n" + "="*30)
    print("测试集平均结果:")
    print(f"  - 平均 Dice Score: {avg_dice:.4f}")
    print(f"  - 平均 IoU Score: {avg_iou:.4f}")
    print(f"  - 平均血管密度: {avg_density:.4f}")
    print(f"  - 平均骨架长度: {avg_length:.2f} pixels")
    print("="*30)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='评估视网膜血管分割模型。')
    parser.add_argument('--model_path', type=str, default=None, help='要评估的.pth模型文件路径。如果未提供，将自动使用最新模型。')
    
    args = parser.parse_args()

    config = {
        'data_dir': r'segmentation/datasets/drive-retina-dataset-master',
        'output_dir': r'segmentation/outputs',
        'image_size': 480,
        'model_path': args.model_path
    }
    
    evaluate(config)
