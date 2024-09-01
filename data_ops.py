import random
import json
import os


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


if __name__ == '__main__':
    find_original_datapoints('data/original', 'original_paths.json')
    select_original_portion('original_paths.json', 'original_paths_portion.json', 0.5, random_seed=42)
