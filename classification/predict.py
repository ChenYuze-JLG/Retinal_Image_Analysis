import torch
from torchvision import models
import torch.nn as nn
from PIL import Image
import json
import os
import argparse

from utils import get_data_transforms

def get_latest_model_path(output_dir):
    """Find the latest model file in the output directory."""
    try:
        # Get all timestamped subfolders
        subdirs = [d for d in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, d))]
        if not subdirs:
            return None

        # Sort by name (timestamps ensure chronological order)
        latest_dir = sorted(subdirs)[-1]
        model_path = os.path.join(output_dir, latest_dir, 'best_model.pth')

        return model_path if os.path.exists(model_path) else None
    except FileNotFoundError:
        return None

def predict(model_path, image_path, top_k=3):
    """
    Predict a single image using a trained model.

    Args:
    - model_path (str): path to the .pth model file.
    - image_path (str): path to the image to classify.
    - top_k (int): return top K probable classes.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 1. Load class names
    model_dir = os.path.dirname(model_path)
    class_names_path = os.path.join(model_dir, 'class_names.json')
    if not os.path.exists(class_names_path):
        raise FileNotFoundError(f"Class file not found: {class_names_path}")

    with open(class_names_path, 'r') as f:
        class_names = json.load(f)
    num_classes = len(class_names)

    # 2. Load model
    model = models.squeezenet1_1(pretrained=False)
    model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
    model.num_classes = num_classes
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    # 3. Image preprocessing
    data_transforms = get_data_transforms()
    transform = data_transforms['val']

    try:
        image = Image.open(image_path).convert('RGB')
    except FileNotFoundError:
        print(f"Error: image file '{image_path}' not found")
        return

    image_tensor = transform(image).unsqueeze(0).to(device)

    # 4. Prediction
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        top_prob, top_indices = torch.topk(probabilities, top_k)

    # 5. Print results
    print(f"\nPrediction for image '{os.path.basename(image_path)}':")
    for i in range(top_k):
        prob = top_prob[i].item()
        class_name = class_names[top_indices[i].item()]
        print(f"  - {i + 1}. Class: {class_name:<30} | Probability: {prob:.4f}")

if __name__ == '__main__':
    # --- Defaults ---
    OUTPUT_DIR = r'./outputs'
    DEFAULT_IMAGE = r'./datasets/sampled_images/0.0.Normal/1ffa962d-8d87-11e8-9daf-6045cb817f5b..JPG'

    parser = argparse.ArgumentParser(description='Classify a single image using a trained model.')
    parser.add_argument('--model', type=str, help='Path to trained .pth model. If not provided, uses latest model.')
    parser.add_argument('--image', type=str,
                        help=f'Path to image. If not provided, uses default image: {os.path.basename(DEFAULT_IMAGE)}')
    parser.add_argument('--top_k', type=int, default=3, help='Show top K probable classes.')

    args = parser.parse_args()

    # Determine model path
    model_to_use = args.model
    if not model_to_use:
        print("No model specified, searching for latest model...")
        model_to_use = get_latest_model_path(OUTPUT_DIR)
        if model_to_use:
            print(f"Using latest model: {model_to_use}")
        else:
            print(f"Error: no model found in '{OUTPUT_DIR}'. Run training first.")
            exit()

    # Determine image path
    image_to_use = args.image if args.image else DEFAULT_IMAGE

    # Check files
    if not os.path.exists(model_to_use):
        print(f"Error: model file '{model_to_use}' not found")
    elif not os.path.exists(image_to_use):
        print(f"Error: image file '{image_to_use}' not found")
    else:
        predict(model_to_use, image_to_use, args.top_k)