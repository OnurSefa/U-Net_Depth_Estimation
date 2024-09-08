import random
import json
import os
import numpy as np
import h5py
from PIL import Image


def find_original_datapoints(in_dir, out_path):

    base_names = [in_dir]
    data_points = []
    while len(base_names) > 0:
        base_name = base_names.pop(0)
        print(base_name)
        if os.path.exists(os.path.join(base_name, 'INDEX.txt')):
            with open(os.path.join(base_name, 'INDEX.txt'), 'r') as f:
                lines = f.readlines()
            data_point = [None, None]
            for line in lines:
                if 'd-' in line and '.pgm' in line:
                    current_path = os.path.join(base_name, line[:-1])
                    if os.path.exists(current_path):
                        data_point[1] = current_path
                if 'r-' in line and '.ppm' in line:
                    current_path = os.path.join(base_name, line[:-1])
                    if os.path.exists(current_path):
                        data_point[0] = current_path
                if None not in data_point:
                    data_points.append(data_point)
                    data_point = [None, None]
        for name in os.listdir(base_name):
            if os.path.isdir(os.path.join(base_name, name)):
                base_names.append(os.path.join(base_name, name))
    with open(out_path, 'w') as f:
        json.dump(data_points, f, indent=6)


def select_original_portion(in_path, out_path, ratio, test_out_path, test_ratio, random_seed=42):
    with open(in_path, 'r') as f:
        data_points = json.load(f)

    selected_count = int(round((len(data_points) * ratio)))
    if random_seed:
        random.seed(random_seed)
    random.shuffle(data_points)
    with open(out_path, 'w') as f:
        json.dump(data_points[:selected_count], f, indent=6)
    test_selected_count = selected_count + int(round((len(data_points) * test_ratio)))
    with open(test_out_path, 'w') as f:
        json.dump(data_points[selected_count:test_selected_count], f, indent=6)


def extract_nyu_labeled_dataset(mat_file_path, output_dir):
    # Create output directories
    rgb_dir = os.path.join(output_dir, 'rgb')
    depth_dir = os.path.join(output_dir, 'depth')
    os.makedirs(rgb_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)

    # Read the .mat file
    print(f"Reading {mat_file_path}...")
    with h5py.File(mat_file_path, 'r') as f:
        # Extract the images and depths
        images = f['images']
        depths = f['depths']

        # Get the number of samples
        num_samples = images.shape[0]

        print(f"Extracting {num_samples} samples...")
        for i in range(num_samples):
            # Extract and save RGB image
            rgb_image = np.transpose(images[i], (2, 1, 0))
            rgb_image = Image.fromarray(rgb_image.astype('uint8'))
            rgb_image.save(os.path.join(rgb_dir, f'rgb_{i:05d}.png'))

            # Extract and save depth image
            depth_image = np.transpose(depths[i], (1, 0))
            depth_image = (depth_image / np.max(depth_image) * 255).astype('uint8')
            depth_image = Image.fromarray(depth_image)
            depth_image.save(os.path.join(depth_dir, f'depth_{i:05d}.png'))

        print("Extraction complete!")


def split_finetune_dataset(image_dir, depth_dir, train_count, out_train_path, out_test_path, random_seed=42):
    random.seed(random_seed)

    names = os.listdir(image_dir)
    random.shuffle(names)
    train_names = []
    test_names = []
    for name in names[:train_count]:
        train_names.append([os.path.join(image_dir, name), os.path.join(depth_dir, "depth" + name[3:])])
    for name in names[train_count:]:
        test_names.append([os.path.join(image_dir, name), os.path.join(depth_dir, "depth" + name[3:])])

    with open(out_train_path, 'w') as f:
        json.dump(train_names, f, indent=6)
    with open(out_test_path, 'w') as f:
        json.dump(test_names, f, indent=6)


if __name__ == '__main__':
    find_original_datapoints('../data/depth_data/original', '../data/depth_data/original_paths.json')
    select_original_portion('../data/depth_data/original_paths.json', '../data/depth_data/original_paths_portion.json', 0.2, '../data/depth_data/original_paths_portion_test.json', 0.02, random_seed=42)
    extract_nyu_labeled_dataset('../data/depth_data/nyu_depth_v2_labeled.mat', '../data/depth_data/labeled')
    split_finetune_dataset('../data/depth_data/labeled/rgb', '../data/depth_data/labeled/depth', 1200, '../data/depth_data/fine_tune_train.json', '../data/depth_data/fine_tune_test.json')
