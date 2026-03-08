import os
import time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import segmentation_models_pytorch as smp

from data_loader import DRIVEDataset, get_transforms
from model import build_model
from utils import calculate_metrics


def train_model(config):
    """Main training function."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # dataset
    transforms = get_transforms(config['image_size'])
    full_dataset = DRIVEDataset(
        root_dir=config['data_dir'],
        transform=transforms,
        subset='train'
    )

    # split train / validation
    val_percent = 0.2
    n_val = int(len(full_dataset) * val_percent)
    n_train = len(full_dataset) - n_val
    train_dataset, val_dataset = random_split(full_dataset, [n_train, n_val])

    dataloaders = {
        'train': DataLoader(
            train_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=4,
            pin_memory=True
        ),
        'val': DataLoader(
            val_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
    }

    print(f"Dataset loaded: {n_train} training samples, {n_val} validation samples")

    # model
    model = build_model().to(device)

    # segmentation loss
    loss_fn = smp.losses.DiceLoss(
        smp.losses.BINARY_MODE,
        from_logits=False
    )
    bce_loss = torch.nn.BCELoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=config['learning_rate']
    )

    # output directory and TensorBoard
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_dir = os.path.join(config['output_dir'], timestamp)
    os.makedirs(output_dir, exist_ok=True)

    writer = SummaryWriter(
        log_dir=os.path.join(output_dir, 'tensorboard_logs')
    )

    print(f"Logs and models saved to: {output_dir}")

    best_val_dice = 0.0

    for epoch in range(config['num_epochs']):

        print(f"\nEpoch {epoch + 1}/{config['num_epochs']}")

        for phase in ['train', 'val']:

            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            epoch_metrics = {'dice': 0.0, 'iou': 0.0}

            progress_bar = tqdm(
                dataloaders[phase],
                desc=f"{phase.capitalize()} Epoch {epoch + 1}"
            )

            for batch in progress_bar:

                images = batch['image'].to(device)
                true_masks = batch['mask'].to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):

                    pred_masks = model(images)

                    # combined loss
                    dice_loss_val = loss_fn(pred_masks, true_masks)
                    bce_loss_val = bce_loss(pred_masks, true_masks)
                    loss = dice_loss_val + bce_loss_val

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * images.size(0)

                metrics = calculate_metrics(
                    pred_masks.detach(),
                    true_masks
                )

                epoch_metrics['dice'] += metrics['dice'] * images.size(0)
                epoch_metrics['iou'] += metrics['iou'] * images.size(0)

                progress_bar.set_postfix(
                    loss=loss.item(),
                    dice=metrics['dice'],
                    iou=metrics['iou']
                )

            dataset_size = len(dataloaders[phase].dataset)

            epoch_loss = running_loss / dataset_size
            epoch_dice = epoch_metrics['dice'] / dataset_size
            epoch_iou = epoch_metrics['iou'] / dataset_size

            writer.add_scalar(f'Loss/{phase}', epoch_loss, epoch)
            writer.add_scalar(f'Dice/{phase}', epoch_dice, epoch)
            writer.add_scalar(f'IoU/{phase}', epoch_iou, epoch)

            print(
                f"{phase.capitalize()} - "
                f"Loss: {epoch_loss:.4f}, "
                f"Dice: {epoch_dice:.4f}, "
                f"IoU: {epoch_iou:.4f}"
            )

            # save best model
            if phase == 'val' and epoch_dice > best_val_dice:
                best_val_dice = epoch_dice
                torch.save(
                    model.state_dict(),
                    os.path.join(output_dir, 'best_model.pth')
                )
                print("New best model saved")

    writer.close()

    print(
        f"\nTraining finished. "
        f"Best validation Dice: {best_val_dice:.4f}"
    )


if __name__ == '__main__':

    config = {
        'data_dir': r'./datasets/drive-retina-dataset-master',
        'output_dir': r'./outputs',
        'image_size': 480,
        'num_epochs': 30,
        'batch_size': 2,
        'learning_rate': 1e-4,
    }

    train_model(config)