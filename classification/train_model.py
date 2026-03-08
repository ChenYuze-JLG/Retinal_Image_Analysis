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

from utils import get_data_transforms, get_class_names

def train_model(data_dir, output_dir, num_epochs=25, batch_size=16, learning_rate=0.001):
    """
    Train an image classification model.

    Args:
    - data_dir (str): root folder of sampled dataset.
    - output_dir (str): folder to save model and logs.
    - num_epochs (int): total number of training epochs.
    - batch_size (int): number of images per batch.
    - learning_rate (float): learning rate.
    """
    print("Starting training...")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Prepare data
    data_transforms = get_data_transforms()
    full_dataset = datasets.ImageFolder(data_dir, data_transforms['train'])

    # Split into train and validation sets (2:1)
    train_size = int(0.67 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Use validation transforms for val set
    val_dataset.dataset.transform = data_transforms['val']

    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4),
        'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    }
    dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset)}
    class_names = get_class_names(data_dir)
    num_classes = len(class_names)

    print(f"Dataset info: total={len(full_dataset)}, train={dataset_sizes['train']}, val={dataset_sizes['val']}")
    print(f"Number of classes: {num_classes}")

    # 2. Define model
    # Use SqueezeNet 1.1 (small and efficient)
    model = models.squeezenet1_1(pretrained=True)

    # Freeze all pretrained parameters
    for param in model.parameters():
        param.requires_grad = False

    # Replace classifier to match our number of classes
    model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
    model.num_classes = num_classes

    model = model.to(device)

    # 3. Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    # Only train classifier layer
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    # 4. Prepare output folder and TensorBoard writer
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    model_output_dir = os.path.join(output_dir, timestamp)
    os.makedirs(model_output_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(model_output_dir, 'tensorboard_logs'))

    # Save class names for later use
    with open(os.path.join(model_output_dir, 'class_names.json'), 'w') as f:
        json.dump(class_names, f)

    print(f"Model and logs will be saved at: {model_output_dir}")

    # 5. Training loop
    best_model_wts = model.state_dict()
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        for phase in ['train', 'val']:
            model.train() if phase == 'train' else model.eval()

            running_loss = 0.0
            running_corrects = 0

            # Progress bar
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

            # Log metrics to TensorBoard
            writer.add_scalar(f'Loss/{phase}', epoch_loss, epoch)
            writer.add_scalar(f'Accuracy/{phase}', epoch_acc, epoch)

            # Save best model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()
                torch.save(model.state_dict(), os.path.join(model_output_dir, 'best_model.pth'))
                print("Saved new best model!")

    writer.close()
    print(f'\nTraining complete! Best val accuracy: {best_acc:.4f}')
    print(f"Best model saved at: {os.path.join(model_output_dir, 'best_model.pth')}")


if __name__ == '__main__':
    # --- Config ---
    DATASET_DIR = r'./datasets/sampled_images'
    OUTPUT_DIR = r'./outputs'

    NUM_EPOCHS = 20  # adjust if needed
    BATCH_SIZE = 16  # reduce if GPU memory is limited
    LEARNING_RATE = 0.001
    # --- End config ---

    train_model(DATASET_DIR, OUTPUT_DIR, num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE)