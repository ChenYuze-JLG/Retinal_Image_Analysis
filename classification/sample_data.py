import os
import shutil
import random

def sample_retinal_dataset(source_dir, dest_dir, num_images_per_category=3):
    """
    Sample a retinal dataset, selecting a set number of images per category.

    Args:
    - source_dir (str): path to original dataset with subfolders per category.
    - dest_dir (str): folder to store sampled images.
    - num_images_per_category (int): number of images to sample per category.
    """
    # Remove existing destination folder to avoid old data interference
    if os.path.exists(dest_dir):
        print(f"Destination folder '{dest_dir}' exists, clearing it...")
        shutil.rmtree(dest_dir)

    os.makedirs(dest_dir)
    print(f"Created new destination folder: '{dest_dir}'")

    # Get all category subfolders
    try:
        category_folders = [f for f in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, f))]
        if not category_folders:
            print(f"Error: No category subfolders found in '{source_dir}'.")
            return
    except FileNotFoundError:
        print(f"Error: Source directory '{source_dir}' not found.")
        return

    print(f"Found {len(category_folders)} categories. Start sampling...")

    # Iterate through each category
    for category_name in category_folders:
        source_category_path = os.path.join(source_dir, category_name)
        dest_category_path = os.path.join(dest_dir, category_name)

        os.makedirs(dest_category_path, exist_ok=True)

        # Get all image files in category
        try:
            images = [f for f in os.listdir(source_category_path)
                      if os.path.isfile(os.path.join(source_category_path, f))]
        except FileNotFoundError:
            print(f"Warning: Cannot access '{source_category_path}', skipping this category.")
            continue

        # Randomly select images
        if len(images) >= num_images_per_category:
            sampled_images = random.sample(images, num_images_per_category)
        else:
            print(f"Warning: Category '{category_name}' has only {len(images)} images, copying all.")
            sampled_images = images

        if not sampled_images:
            print(f"Warning: No images found in category '{category_name}'.")
            continue

        # Copy selected images to destination
        for image_name in sampled_images:
            source_image_path = os.path.join(source_category_path, image_name)
            dest_image_path = os.path.join(dest_category_path, image_name)
            shutil.copy2(source_image_path, dest_image_path)

        print(f"Processed category '{category_name}', copied {len(sampled_images)} images.")

    print("\nSampling completed for all categories!")


if __name__ == '__main__':
    # --- Config ---
    SOURCE_DATASET_DIR = r'./datasets/1000 Fundus images with 39 categories/1000images'
    DESTINATION_DIR = r'./datasets/sampled_images'
    IMAGES_PER_CATEGORY = 3
    # --- End config ---

    sample_retinal_dataset(SOURCE_DATASET_DIR, DESTINATION_DIR, IMAGES_PER_CATEGORY)