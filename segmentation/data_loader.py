import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class DRIVEDataset(Dataset):
    """
    Custom PyTorch Dataset for DRIVE retinal vessel segmentation.
    """

    def __init__(self, root_dir, transform=None, subset='train'):
        """
        Args:
        - root_dir (str): root directory of DRIVE dataset (e.g., '.../segmentation/DRIVE').
        - transform (callable, optional): transformation applied to samples.
        - subset (str): 'train' or 'test' to specify which set to load.
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
            raise ValueError("subset must be 'train' or 'test'")

        # get all image filenames
        self.image_files = sorted([f for f in os.listdir(self.images_dir) if f.endswith('.tif')])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        base_name = img_name.split('_')[0]  # e.g., '21' from '21_training.tif'

        if self.subset == 'train':
            mask_name = f"{base_name}_manual1.gif"
            fov_mask_name = f"{base_name}_training_mask.gif"
        else:
            mask_name = f"{base_name}_manual1.gif"
            fov_mask_name = f"{base_name}_test_mask.gif"

        # load image, mask, and field-of-view mask
        img_path = os.path.join(self.images_dir, img_name)
        mask_path = os.path.join(self.masks_dir, mask_name)
        fov_mask_path = os.path.join(self.fov_masks_dir, fov_mask_name)

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        fov_mask = Image.open(fov_mask_path).convert("L")

        # convert mask to binary
        mask = np.array(mask) > 0
        mask = Image.fromarray(mask.astype(np.uint8) * 255)

        sample = {'image': image, 'mask': mask, 'fov_mask': fov_mask}

        if self.transform:
            sample = self.transform(sample)

        return sample


class ResizeAndPad:
    """Resize image and pad to make it square."""

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, mask, fov_mask = sample['image'], sample['mask'], sample['fov_mask']

        w, h = image.size
        if w > h:
            new_w, new_h = self.output_size, int(h * self.output_size / w)
        else:
            new_w, new_h = int(w * self.output_size / h), self.output_size

        image = image.resize((new_w, new_h), Image.BILINEAR)
        mask = mask.resize((new_w, new_h), Image.NEAREST)
        fov_mask = fov_mask.resize((new_w, new_h), Image.NEAREST)

        # pad to square
        new_image = Image.new('RGB', (self.output_size, self.output_size))
        new_mask = Image.new('L', (self.output_size, self.output_size))
        new_fov_mask = Image.new('L', (self.output_size, self.output_size))

        new_image.paste(image, ((self.output_size - new_w) // 2, (self.output_size - new_h) // 2))
        new_mask.paste(mask, ((self.output_size - new_w) // 2, (self.output_size - new_h) // 2))
        new_fov_mask.paste(fov_mask, ((self.output_size - new_w) // 2, (self.output_size - new_h) // 2))

        sample['image'], sample['mask'], sample['fov_mask'] = new_image, new_mask, new_fov_mask
        return sample


class ToTensor:
    """Convert PIL images in a sample to Tensors."""

    def __call__(self, sample):
        image, mask, fov_mask = sample['image'], sample['mask'], sample['fov_mask']

        image = transforms.functional.to_tensor(image)
        image = transforms.functional.normalize(image, mean=[0.485, 0.456, 0.406],
                                                  std=[0.229, 0.224, 0.225])

        mask = torch.from_numpy(np.asarray(mask, dtype=np.int64)).float().div(255.0).unsqueeze(0)
        fov_mask = torch.from_numpy(np.asarray(fov_mask, dtype=np.int64)).float().div(255.0).unsqueeze(0)

        return {'image': image, 'mask': mask, 'fov_mask': fov_mask}


def get_transforms(image_size):
    return transforms.Compose([
        ResizeAndPad(image_size),
        ToTensor()
    ])