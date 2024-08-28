import random

import numpy

from fill_depth_colorization import fill_depth_map, advanced_fill_depth_map, fill_depth_colorization
import json
import os
from PIL import Image
import numpy as np
import torchvision.transforms as T
from torchvision.transforms.functional import pil_to_tensor
import torch


def split_data(csv_path, train_path, test_path, train_size=40000, random_seed=42):
    if random_seed:
        random.seed(random_seed)
    csv = pd.read_csv(csv_path)
    names = []
    for index, row in csv.iterrows():
        names.append([row[0], row[1]])

    random.shuffle(names)

    with open(train_path, 'w') as f:
        json.dump({"names": names[:train_size]}, f, indent=6)
    with open(test_path, 'w') as f:
        json.dump({"names": names[train_size:]}, f, indent=6)


def find_original_datapoints(in_dir, out_path):

    base_names = [in_dir]
    data_points = []
    while len(base_names) > 0:
        base_name = base_names.pop(0)
        if os.path.exists(os.path.join(base_name, 'INDEX.txt')):
            with open(os.path.join(base_name, 'INDEX.txt'), 'r') as f:
                lines = f.readlines()
            data_point = [None, None]
            for line in lines:
                if 'd-' in line and '.pgm' in line:
                    data_point[1] = os.path.join(base_name, line[:-1])
                if 'r-' in line and '.ppm' in line:
                    data_point[0] = os.path.join(base_name, line[:-1])
                if None not in data_point:
                    data_points.append(data_point)
                    data_point = [None, None]
        for name in os.listdir(base_name):
            if os.path.isdir(os.path.join(base_name, name)):
                base_names.append(os.path.join(base_name, name))
    with open(out_path, 'w') as f:
        json.dump(data_points, f, indent=6)


def select_original_portion(in_path, out_path, ratio, random_seed=42):
    with open(in_path, 'r') as f:
        data_points = json.load(f)

    selected_count = int(round((len(data_points) * ratio)))
    if random_seed:
        random.seed(random_seed)
    random.shuffle(data_points)
    with open(out_path, 'w') as f:
        json.dump(data_points[:selected_count], f, indent=6)


def fill_depth_images(dp_path, out_dir):
    with open(dp_path, 'r') as f:
        data_points = json.load(f)

    # tensor_to_pil = T.ToPILImage(mode="L")
    for image_path, depth_path in data_points:
        image = Image.open(image_path)
        image = np.array(image, dtype=np.float32)
        image = image - image.min()
        image = image / image.max()
        depth = Image.open(depth_path)
        depth = np.array(depth, dtype=np.float32)
        depth = depth - depth.min()
        depth = depth / depth.max()
        depth_image = Image.fromarray(depth, mode='F')
        depth_name = depth_path.split('/')[-1][:-4] + "-0.tiff"
        depth_image.save(os.path.join(out_dir, depth_name))
        processed = fill_depth_colorization(image, depth)
        processed = processed - np.min(processed)
        processed = processed / np.max(processed)
        processed = Image.fromarray(processed, mode='F')
        depth_name = depth_path.split('/')[-1][:-4] + "-1.tiff"
        save_name = os.path.join(out_dir, depth_name)
        print(save_name)
        processed.save(save_name)


if __name__ == '__main__':
    find_original_datapoints('data/original', 'original_paths.json')
    select_original_portion('original_paths.json', 'original_paths_portion.json', 0.5, random_seed=42)
    # fill_depth_images('original_paths_portion.json', 'data/original_processed')
