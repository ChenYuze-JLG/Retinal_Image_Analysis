import os
import torch
from torchvision import transforms

def get_class_names(data_dir):
    """从数据集目录中获取所有类别的名称。"""
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"指定的目录不存在: {data_dir}")
    
    # ImageFolder会自动按字母顺序排序文件夹，所以我们直接用os.listdir并排序来保持一致
    class_names = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    return class_names

def get_data_transforms():
    """
    获取用于训练集和验证集的数据转换流程。
    - 训练集：随机裁剪、随机水平翻转（数据增强），然后转换为Tensor并标准化。
    - 验证集/测试集：仅进行缩放、中心裁剪、转换为Tensor和标准化。
    """
    # ImageNet的均值和标准差，这是一个常用的基准
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
    }
    return data_transforms

if __name__ == '__main__':
    # --- 用于测试辅助功能 ---
    dataset_path = r'classification/datasets/sampled_images'
    
    print("测试获取类别名称:")
    try:
        classes = get_class_names(dataset_path)
        print(f"共找到 {len(classes)} 个类别。")
        print("类别列表:", classes)
    except FileNotFoundError as e:
        print(e)

    print("\n测试获取数据转换流程:")
    transforms_dict = get_data_transforms()
    print("训练集转换流程:\n", transforms_dict['train'])
    print("\n验证集转换流程:\n", transforms_dict['val'])
