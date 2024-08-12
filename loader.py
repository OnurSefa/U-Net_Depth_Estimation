from torch.utils.data import Dataset, DataLoader
import pandas as pd
from torchvision.io import read_image, ImageReadMode
import random
import json
import torch


class Normalizer:
    def __init__(self, means_path, stds_path):
        self.means = torch.load(means_path).squeeze()
        self.stds = torch.load(stds_path).squeeze()

    def normalize(self, im):
        return (im - self.means) / self.stds

    def denormalize(self, im):
        return im * self.stds + self.means


class DepthDataset(Dataset):
    def __init__(self, names_path, image_transform=None, depth_transform=None, device=None, with_names=False):
        self.image_transform = image_transform
        self.depth_transform = depth_transform
        self.device = device
        self.with_names = with_names

        with open(names_path, 'r') as f:
            self.names = json.load(f)['names']

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        image_path, depth_path = self.names[idx]
        image = read_image(image_path, ImageReadMode.RGB)
        if self.image_transform:
            image = self.image_transform(image)
        depth = read_image(depth_path, ImageReadMode.GRAY)
        if self.depth_transform:
            depth = self.depth_transform(depth)
        if self.device:
            image = image.to(self.device)
            depth = depth.to(self.device)
        image = image / 255
        depth = depth / torch.max(depth)
        if self.with_names:
            return image, depth, image_path
        return image, depth


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


def save_normalization_factors(train_path, out_dir):
    dataset = DepthDataset(train_path)
    data_loader = DataLoader(dataset, batch_size=32, shuffle=False)

    image_count = 0
    image_means = torch.zeros((1, 3, 480, 640))
    image_stds = torch.zeros((1, 3, 480, 640))
    depth_means = torch.zeros((1, 1, 480, 640))
    depth_stds = torch.zeros((1, 1, 480, 640))
    for i, data in enumerate(data_loader):
        images, depths = data
        images = images.float()
        depths = depths.float()
        image_count += images.shape[0]
        image_means += images.mean(dim=0, keepdim=True)
        image_stds += images.std(dim=0, keepdim=True)
        depth_means += depths.mean(dim=0, keepdim=True)
        depth_stds += depths.std(dim=0, keepdim=True)
    image_means /= image_count
    image_stds /= image_count
    depth_means /= image_count
    depth_stds /= image_count

    torch.save(image_means, f'{out_dir}/image_means.pth')
    torch.save(image_stds, f'{out_dir}/image_stds.pth')
    torch.save(depth_means, f'{out_dir}/depth_means.pth')
    torch.save(depth_stds, f'{out_dir}/depth_stds.pth')


if __name__ == '__main__':
    # split_data('data/nyu2_train.csv', 'data/train_names.json', 'data/test_names.json', 4000)
    save_normalization_factors('data/train_names.json', 'data')
    # normalizer_image = Normalizer('data/image_means.pth', 'data/image_stds.pth')
    # normalizer_depth = Normalizer('data/depth_means.pth', 'data/depth_stds.pth')
    # dataset = DepthDataset('data/train_names.json', normalizer_image.normalize, normalizer_depth.normalize)
    # data_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    # for i, data in enumerate(data_loader):
    #     print('a')


