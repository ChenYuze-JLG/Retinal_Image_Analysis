import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import time
import json

# 导入我们自己编写的辅助工具
from utils import get_data_transforms, get_class_names

def train_model(data_dir, output_dir, num_epochs=25, batch_size=16, learning_rate=0.001):
    """
    训练图像分类模型。

    参数:
    - data_dir (str): 采样后数据集的根目录。
    - output_dir (str): 保存模型和日志的输出根目录。
    - num_epochs (int): 训练的总轮数。
    - batch_size (int): 每个批次的图片数量。
    - learning_rate (float): 学习率。
    """
    print("开始模型训练...")
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"CUDA 是否可用: {torch.cuda.is_available()}")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"将使用设备: {device}")

    # 1. 准备数据
    data_transforms = get_data_transforms()
    full_dataset = datasets.ImageFolder(data_dir, data_transforms['train'])

    # 划分训练集和验证集 (2:1)
    train_size = int(0.67 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # 为验证集设置正确的数据转换
    val_dataset.dataset.transform = data_transforms['val']

    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4),
        'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    }
    dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset)}
    class_names = get_class_names(data_dir)
    num_classes = len(class_names)

    print(f"数据集信息: 总数={len(full_dataset)}, 训练集={dataset_sizes['train']}, 验证集={dataset_sizes['val']}")
    print(f"共 {num_classes} 个类别。")

    # 2. 定义模型
    # 使用SqueezeNet 1.1，它是一个非常小且高效的模型
    model = models.squeezenet1_1(pretrained=True)

    # 冻结预训练模型的参数
    for param in model.parameters():
        param.requires_grad = False

    # 替换分类器层以匹配我们的类别数量
    # SqueezeNet的分类器是一个Conv2d层
    model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
    model.num_classes = num_classes

    model = model.to(device)

    # 3. 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    # 只优化我们新添加的分类器层的参数
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    # 4. 创建输出目录和TensorBoard writer
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    model_output_dir = os.path.join(output_dir, timestamp)
    os.makedirs(model_output_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(model_output_dir, 'tensorboard_logs'))

    # 保存类别名称到文件，方便预测时使用
    with open(os.path.join(model_output_dir, 'class_names.json'), 'w') as f:
        json.dump(class_names, f)

    print(f"模型和日志将保存在: {model_output_dir}")

    # 5. 训练循环
    best_model_wts = model.state_dict()
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            # 使用tqdm显示进度条
            progress_bar = tqdm(dataloaders[phase], desc=f"{phase.capitalize()} Epoch {epoch+1}")
            for inputs, labels in progress_bar:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                progress_bar.set_postfix(loss=loss.item(), acc=torch.sum(preds == labels.data).item() / len(inputs))

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # 记录到TensorBoard
            writer.add_scalar(f'Loss/{phase}', epoch_loss, epoch)
            writer.add_scalar(f'Accuracy/{phase}', epoch_acc, epoch)

            # 保存最佳模型
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()
                torch.save(model.state_dict(), os.path.join(model_output_dir, 'best_model.pth'))
                print("已保存新的最佳模型！")

    writer.close()
    print(f'\n训练完成！最佳验证集准确率: {best_acc:4f}')
    print(f"最佳模型已保存在: {os.path.join(model_output_dir, 'best_model.pth')}")


if __name__ == '__main__':
    # --- 配置区域 ---
    DATASET_DIR = r'classification/datasets/sampled_images'
    OUTPUT_DIR = r'classification/outputs'

    # 训练参数
    NUM_EPOCHS = 20  # 可以根据需要调整
    BATCH_SIZE = 16  # 如果显存不足，可以减小这个值
    LEARNING_RATE = 0.001
    # --- 配置区域结束 ---

    train_model(DATASET_DIR, OUTPUT_DIR, num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE)
