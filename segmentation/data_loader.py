import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class DRIVEDataset(Dataset):
    """
    用于DRIVE视网膜血管分割数据集的自定义PyTorch Dataset。
    """

    def __init__(self, root_dir, transform=None, subset='train'):
        """
        参数:
        - root_dir (str): DRIVE数据集的根目录 (e.g., '.../segmentation/DRIVE')。
        - transform (callable, optional): 应用于样本的转换。
        - subset (str): 'train' 或 'test'，指定加载哪个子集。
        """
        self.root_dir = root_dir
        self.transform = transform
        self.subset = subset.lower()

        if self.subset == 'train':
            self.images_dir = os.path.join(root_dir, 'training', 'images')
            self.masks_dir = os.path.join(root_dir, 'training', '1st_manual')
            self.fov_masks_dir = os.path.join(root_dir, 'training', 'mask')
        elif self.subset == 'test':
            self.images_dir = os.path.join(root_dir, 'test', 'images')
            self.masks_dir = os.path.join(root_dir, 'test', '1st_manual')
            self.fov_masks_dir = os.path.join(root_dir, 'test', 'mask')
        else:
            raise ValueError("subset 必须是 'train' 或 'test'")

        # 获取所有图片文件名（不含扩展名）
        self.image_files = sorted([f for f in os.listdir(self.images_dir) if f.endswith('.tif')])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # 构建文件名
        img_name = self.image_files[idx]
        base_name = img_name.split('_')[0]  # e.g., '21' from '21_training.tif'

        if self.subset == 'train':
            mask_name = f"{base_name}_manual1.gif"
            fov_mask_name = f"{base_name}_training_mask.gif"
        else:  # test
            mask_name = f"{base_name}_manual1.gif"
            fov_mask_name = f"{base_name}_test_mask.gif"

        # 加载图像、真值和FOV掩码
        img_path = os.path.join(self.images_dir, img_name)
        mask_path = os.path.join(self.masks_dir, mask_name)
        fov_mask_path = os.path.join(self.fov_masks_dir, fov_mask_name)

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # 'L' for grayscale
        fov_mask = Image.open(fov_mask_path).convert("L")

        # 将掩码转换为二进制 (0 或 1)
        mask = np.array(mask) > 0
        mask = Image.fromarray(mask.astype(np.uint8) * 255)

        sample = {'image': image, 'mask': mask, 'fov_mask': fov_mask}

        if self.transform:
            sample = self.transform(sample)

        return sample


# 定义数据预处理和增强
class ResizeAndPad:
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, mask, fov_mask = sample['image'], sample['mask'], sample['fov_mask']

        # 保持长宽比进行缩放
        w, h = image.size
        if w > h:
            new_w, new_h = self.output_size, int(h * self.output_size / w)
        else:
            new_w, new_h = int(w * self.output_size / h), self.output_size

        image = image.resize((new_w, new_h), Image.BILINEAR)
        mask = mask.resize((new_w, new_h), Image.NEAREST)
        fov_mask = fov_mask.resize((new_w, new_h), Image.NEAREST)

        # 填充到正方形
        new_image = Image.new('RGB', (self.output_size, self.output_size))
        new_mask = Image.new('L', (self.output_size, self.output_size))
        new_fov_mask = Image.new('L', (self.output_size, self.output_size))

        new_image.paste(image, ((self.output_size - new_w) // 2, (self.output_size - new_h) // 2))
        new_mask.paste(mask, ((self.output_size - new_w) // 2, (self.output_size - new_h) // 2))
        new_fov_mask.paste(fov_mask, ((self.output_size - new_w) // 2, (self.output_size - new_h) // 2))

        sample['image'], sample['mask'], sample['fov_mask'] = new_image, new_mask, new_fov_mask
        return sample


class ToTensor:
    """将样本中的PIL图像转换为Tensor。"""

    def __call__(self, sample):
        image, mask, fov_mask = sample['image'], sample['mask'], sample['fov_mask']

        # 标准化图像
        image = transforms.functional.to_tensor(image)
        image = transforms.functional.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # 将掩码转换为 [0, 1] 的Tensor
        mask = torch.from_numpy(np.asarray(mask, dtype=np.int64)).float().div(255.0).unsqueeze(0)
        fov_mask = torch.from_numpy(np.asarray(fov_mask, dtype=np.int64)).float().div(255.0).unsqueeze(0)
        return {'image': image, 'mask': mask, 'fov_mask': fov_mask}

def get_transforms(image_size):
    return transforms.Compose([
        ResizeAndPad(image_size),
        ToTensor()
    ])