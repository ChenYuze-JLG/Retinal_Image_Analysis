import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import argparse

from data_loader import DRIVEDataset, get_transforms
from model import build_model
from utils import (
    calculate_metrics,
    calculate_biomarkers,
    save_comparison_plot,
    unnormalize_image
)


def get_latest_model_path(output_dir):
    """Return the newest trained model in the output directory."""
    try:
        subdirs = [
            d for d in os.listdir(output_dir)
            if os.path.isdir(os.path.join(output_dir, d))
        ]

        if not subdirs:
            return None

        latest_dir = sorted(subdirs)[-1]
        model_path = os.path.join(latest_dir, 'best_model.pth')
        model_path = os.path.join(output_dir, model_path)

        return model_path if os.path.exists(model_path) else None

    except FileNotFoundError:
        return None


def evaluate(config):
    """Run evaluation on the test set."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # select model
    model_path = config['model_path']

    if not model_path:
        print("No model path provided. Searching for latest model...")
        model_path = get_latest_model_path(config['output_dir'])

        if not model_path:
            print(
                f"Error: no model found in '{config['output_dir']}'. "
                "Train the model first."
            )
            return

    print(f"Using model: {model_path}")

    # load model
    model = build_model().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # load test data
    transforms = get_transforms(config['image_size'])

    test_dataset = DRIVEDataset(
        root_dir=config['data_dir'],
        transform=transforms,
        subset='train'  # using training sets for visualization
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False
    )

    # result directory
    results_dir = os.path.join(
        os.path.dirname(model_path),
        'test_results'
    )

    os.makedirs(results_dir, exist_ok=True)

    print(f"Results saved to: {results_dir}")

    total_metrics = {'dice': 0.0, 'iou': 0.0}
    total_biomarkers = {'vessel_density': 0.0, 'skeleton_length': 0.0}

    for i, batch in enumerate(
        tqdm(test_loader, desc="Evaluating test set")
    ):

        image = batch['image'].to(device)
        gt_mask = batch['mask'].to(device)
        fov_mask = batch['fov_mask']

        with torch.no_grad():
            pred_mask = model(image)

        # segmentation metrics
        metrics = calculate_metrics(pred_mask, gt_mask)

        total_metrics['dice'] += metrics['dice']
        total_metrics['iou'] += metrics['iou']

        # convert tensors to numpy for visualization / biomarkers
        original_img_np = unnormalize_image(
            batch['image'].squeeze(0).cpu()
        )

        gt_mask_np = gt_mask.squeeze().cpu().numpy()
        pred_mask_np = (
            pred_mask.squeeze().cpu().numpy() > 0.5
        ).astype(np.uint8)

        fov_mask_np = (
            fov_mask.squeeze().cpu().numpy() > 0.5
        ).astype(np.uint8)

        biomarkers = calculate_biomarkers(
            pred_mask_np,
            fov_mask_np
        )

        total_biomarkers['vessel_density'] += biomarkers['vessel_density']
        total_biomarkers['skeleton_length'] += biomarkers['skeleton_length']

        print(
            f"\nImage {i+1}: "
            f"Dice={metrics['dice']:.4f}, "
            f"IoU={metrics['iou']:.4f}, "
            f"Density={biomarkers['vessel_density']:.4f}, "
            f"Length={biomarkers['skeleton_length']}"
        )

        # save comparison figure
        plot_path = os.path.join(
            results_dir,
            f'comparison_{i+1:02d}.png'
        )

        save_comparison_plot(
            original_img_np,
            gt_mask_np,
            pred_mask_np,
            plot_path
        )

    # average results
    num_samples = len(test_loader)

    avg_dice = total_metrics['dice'] / num_samples
    avg_iou = total_metrics['iou'] / num_samples
    avg_density = total_biomarkers['vessel_density'] / num_samples
    avg_length = total_biomarkers['skeleton_length'] / num_samples

    print("\n" + "=" * 30)
    print("Average results on test set:")
    print(f"  Dice Score: {avg_dice:.4f}")
    print(f"  IoU Score: {avg_iou:.4f}")
    print(f"  Vessel Density: {avg_density:.4f}")
    print(f"  Skeleton Length: {avg_length:.2f} pixels")
    print("=" * 30)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Evaluate retinal vessel segmentation model.'
    )

    parser.add_argument(
        '--model_path',
        type=str,
        default=None,
        help='Path to the .pth model. If not provided, the latest model is used.'
    )

    args = parser.parse_args()

    config = {
        'data_dir': r'./datasets/drive-retina-dataset-master',
        'output_dir': r'./outputs',
        'image_size': 480,
        'model_path': args.model_path
    }

    evaluate(config)