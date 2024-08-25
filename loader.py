from torch.utils.data import Dataset, DataLoader
import pandas as pd
from torchvision.io import read_image, ImageReadMode
from torchvision.transforms.functional import pil_to_tensor
import random
import json
import torch
from PIL import Image


class Normalizer:
    def __init__(self, means_path, stds_path):
        self.means = torch.load(means_path).squeeze()
        self.stds = torch.load(stds_path).squeeze()

    def normalize(self, im):
        return (im - self.means) / self.stds

    def denormalize(self, im):
        return im * self.stds + self.means


class DepthDataset(Dataset):
    def __init__(self, names_path, device=None, with_names=False, normalize=True):
        self.device = device
        self.with_names = with_names

        self.normalize = normalize
        self.mean_normalizers = torch.tensor([0.485, 0.456, 0.406])
        self.std_normalizers = torch.tensor([0.229, 0.224, 0.225])

        with open(names_path, 'r') as f:
            self.names = json.load(f)['names']

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        image_path, depth_path = self.names[idx]
        image = read_image(image_path, ImageReadMode.RGB)[:, 10:-10, 10:-10]
        if self.normalize:
            image = ((image / 255) - self.mean_normalizers[:, None, None]) / self.std_normalizers[:, None, None]
        else:
            image = (image / 255)
        depth_image = Image.open(depth_path).convert('L')
        depth_image = depth_image.resize((depth_image.size[0]//2, depth_image.size[1]//2))
        depth = pil_to_tensor(depth_image)[:, 5:-5, 5:-5]
        depth = ((depth - torch.min(depth)) / (torch.max(depth) - torch.min(depth)))
        if self.device:
            image = image.to(self.device)
            depth = depth.to(self.device)
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


