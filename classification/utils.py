import os
import torch
from torchvision import transforms


def get_class_names(data_dir):
    """Get all class names from the dataset folder."""
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Directory not found: {data_dir}")

    # List subfolders and sort them to get class names
    class_names = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    return class_names


def get_data_transforms():
    """
    Get data transforms for training and validation.
    - Train: random crop and horizontal flip for augmentation, then convert to tensor and normalize.
    - Val/Test: resize and center crop, then convert to tensor and normalize.
    """
    mean = [0.485, 0.456, 0.406]  # ImageNet mean
    std = [0.229, 0.224, 0.225]  # ImageNet std

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
    # --- test helper functions ---
    dataset_path = r'./datasets/sampled_images'

    print("Test: get class names")
    try:
        classes = get_class_names(dataset_path)
        print(f"Found {len(classes)} classes.")
        print("Class list:", classes)
    except FileNotFoundError as e:
        print(e)

    print("\nTest: get data transforms")
    transforms_dict = get_data_transforms()
    print("Train transforms:\n", transforms_dict['train'])
    print("\nValidation transforms:\n", transforms_dict['val'])