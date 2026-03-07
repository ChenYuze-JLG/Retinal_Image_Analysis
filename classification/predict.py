import torch
from torchvision import models
import torch.nn as nn
from PIL import Image
import json
import os
import argparse

# 导入我们自己编写的辅助工具
from utils import get_data_transforms


def get_latest_model_path(output_dir):
    """在输出目录中查找最新的模型文件路径。"""
    try:
        # 获取所有带时间戳的子目录
        subdirs = [d for d in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, d))]
        if not subdirs:
            return None

        # 按名称排序（时间戳格式保证了字母顺序就是时间顺序）
        latest_dir = sorted(subdirs)[-1]
        model_path = os.path.join(output_dir, latest_dir, 'best_model.pth')

        return model_path if os.path.exists(model_path) else None
    except FileNotFoundError:
        return None


def predict(model_path, image_path, top_k=3):
    """
    使用训练好的模型对单张图片进行预测。

    参数:
    - model_path (str): .pth模型文件的路径。
    - image_path (str): 要预测的图片路径。
    - top_k (int): 返回最可能的k个类别。
    """
    # 检查设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 1. 加载类别名称
    model_dir = os.path.dirname(model_path)
    class_names_path = os.path.join(model_dir, 'class_names.json')
    if not os.path.exists(class_names_path):
        raise FileNotFoundError(f"找不到类别文件: {class_names_path}。请确保它与模型在同一目录下。")

    with open(class_names_path, 'r') as f:
        class_names = json.load(f)
    num_classes = len(class_names)

    # 2. 加载模型
    # 重新构建模型结构，与训练时保持一致
    model = models.squeezenet1_1(pretrained=False)  # 这里不加载预训练权重，因为我们要加载自己的权重
    model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
    model.num_classes = num_classes

    # 加载我们训练好的权重
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    # 3. 图像预处理
    data_transforms = get_data_transforms()
    transform = data_transforms['val']  # 使用验证集的转换流程

    try:
        image = Image.open(image_path).convert('RGB')
    except FileNotFoundError:
        print(f"错误：找不到图片文件 '{image_path}'")
        return

    image_tensor = transform(image).unsqueeze(0).to(device)

    # 4. 进行预测
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        top_prob, top_indices = torch.topk(probabilities, top_k)

    # 5. 打印结果
    print(f"\n对图片 '{os.path.basename(image_path)}' 的预测结果:")
    for i in range(top_k):
        prob = top_prob[i].item()
        class_name = class_names[top_indices[i].item()]
        print(f"  - {i + 1}. 类别: {class_name:<30} | 概率: {prob:.4f}")


if __name__ == '__main__':
    # --- 默认配置 ---
    OUTPUT_DIR = r'classification/outputs'
    DEFAULT_IMAGE = r'classification/datasets/sampled_images/0.0.Normal/1ffa9630-8d87-11e8-9daf-6045cb817f5b..JPG'

    parser = argparse.ArgumentParser(description='使用训练好的模型进行图像分类。')
    parser.add_argument('--model', type=str, help='已训练模型的.pth文件路径。如果未提供，将自动使用最新模型。')
    parser.add_argument('--image', type=str,
                        help=f'要分类的单张图片路径。如果未提供，将使用默认图片: {os.path.basename(DEFAULT_IMAGE)}')
    parser.add_argument('--top_k', type=int, default=3, help='显示最可能的K个类别。')

    args = parser.parse_args()

    # 确定模型路径
    model_to_use = args.model
    if not model_to_use:
        print("未指定模型，正在查找最新模型...")
        model_to_use = get_latest_model_path(OUTPUT_DIR)
        if model_to_use:
            print(f"将使用最新模型: {model_to_use}")
        else:
            print(f"错误: 在 '{OUTPUT_DIR}' 中找不到任何模型。请先运行 train_model.py 进行训练。")
            exit()  # 退出脚本

    # 确定图片路径
    image_to_use = args.image if args.image else DEFAULT_IMAGE

    # 检查文件是否存在
    if not os.path.exists(model_to_use):
        print(f"错误：找不到模型文件 '{model_to_use}'")
    elif not os.path.exists(image_to_use):
        print(f"错误：找不到图片文件 '{image_to_use}'")
    else:
        predict(model_to_use, image_to_use, args.top_k)