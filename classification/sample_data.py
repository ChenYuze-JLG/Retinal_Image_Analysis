import os
import shutil
import random


def sample_retinal_dataset(source_dir, dest_dir, num_images_per_category=3):
    """
    从视网膜数据集中采样，每个类别选择指定数量的图片。

    参数:
    - source_dir (str): 原始数据集的路径，包含各个类别的子文件夹。
    - dest_dir (str): 存放采样后图片的目标路径。
    - num_images_per_category (int): 每个类别要采样的图片数量。
    """
    # 如果目标目录已存在，先清空，避免旧数据干扰
    if os.path.exists(dest_dir):
        print(f"目标目录 '{dest_dir}' 已存在，正在清空...")
        shutil.rmtree(dest_dir)

    os.makedirs(dest_dir)
    print(f"已创建新的目标目录: '{dest_dir}'")

    # 获取所有类别的文件夹名称
    try:
        category_folders = [f for f in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, f))]
        if not category_folders:
            print(f"错误：在 '{source_dir}' 中没有找到任何类别子文件夹。")
            return
    except FileNotFoundError:
        print(f"错误：源目录 '{source_dir}' 不存在。")
        return

    print(f"共找到 {len(category_folders)} 个类别。开始进行采样...")

    # 遍历每个类别文件夹
    for category_name in category_folders:
        source_category_path = os.path.join(source_dir, category_name)
        dest_category_path = os.path.join(dest_dir, category_name)

        # 在目标目录中创建对应的类别文件夹
        os.makedirs(dest_category_path, exist_ok=True)

        # 获取类别下的所有图片文件
        try:
            images = [f for f in os.listdir(source_category_path) if
                      os.path.isfile(os.path.join(source_category_path, f))]
        except FileNotFoundError:
            print(f"警告：无法访问 '{source_category_path}'，跳过该类别。")
            continue

        # 随机选择指定数量的图片
        if len(images) >= num_images_per_category:
            sampled_images = random.sample(images, num_images_per_category)
        else:
            print(
                f"警告：类别 '{category_name}' 的图片数量（{len(images)}）不足 {num_images_per_category} 张，将复制所有图片。")
            sampled_images = images

        if not sampled_images:
            print(f"警告：类别 '{category_name}' 中没有找到图片文件。")
            continue

        # 复制选中的图片到目标文件夹
        for image_name in sampled_images:
            source_image_path = os.path.join(source_category_path, image_name)
            dest_image_path = os.path.join(dest_category_path, image_name)
            shutil.copy2(source_image_path, dest_image_path)

        print(f"已处理类别 '{category_name}'，复制了 {len(sampled_images)} 张图片。")

    print("\n所有类别采样完成！")


if __name__ == '__main__':
    # --- 配置区域 ---
    # 原始数据集的根目录
    SOURCE_DATASET_DIR = r'classification/datasets/1000 Fundus images with 39 categories/1000images'

    # 采样后新数据集的存放目录
    DESTINATION_DIR = r'classification/datasets/sampled_images'

    # 每个类别要保留的图片数量
    IMAGES_PER_CATEGORY = 3
    # --- 配置区域结束 ---

    sample_retinal_dataset(SOURCE_DATASET_DIR, DESTINATION_DIR, IMAGES_PER_CATEGORY)
