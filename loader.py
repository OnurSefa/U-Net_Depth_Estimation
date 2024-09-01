from torch.utils.data import Dataset, DataLoader
import pandas as pd
from torchvision.io import read_image, ImageReadMode
from torchvision.transforms.functional import pil_to_tensor, hflip
import random
import json
import torch
from PIL import Image


class DepthDataset(Dataset):
    def __init__(self, names_path, device=None, normalize=True, mode='train'):
        ### modes:
        # train
        # inference

        self.mode = mode
        self.device = device
        self.normalize = normalize
        self.mean_normalizers = torch.tensor([0.485, 0.456, 0.406])
        self.std_normalizers = torch.tensor([0.229, 0.224, 0.225])

        with open(names_path, 'r') as f:
            self.names = json.load(f)['names']

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        image_path, depth_path = self.names[idx]
        image = read_image(image_path, ImageReadMode.RGB)
        if self.normalize:
            image = ((image / 255) - self.mean_normalizers[:, None, None]) / self.std_normalizers[:, None, None]
        else:
            image = (image / 255)
        depth_image = Image.open(depth_path).convert('L')
        depth_image = depth_image.resize((depth_image.size[0]//2, depth_image.size[1]//2))
        depth = pil_to_tensor(depth_image)
        depth = ((depth - torch.min(depth)) / (torch.max(depth) - torch.min(depth)))
        if self.device:
            image = image.to(self.device)
            depth = depth.to(self.device)

        if self.mode == 'train':
            random_flip = random.random()
            random_channel = random.random()
            if random_flip < 0.5:
                image = hflip(image)
                depth = hflip(depth)
            if random_channel < 0.1:
                image = image[[1, 0, 2], :, :]
            elif random_channel < 0.2:
                image = image[[0, 2, 1], :, :]
            elif random_channel < 0.3:
                image = image[[2, 1, 0], :, :]
            return image, depth
        elif self.mode == 'inference':
            image_flipped = hflip(image)
            return image, image_flipped, depth, depth_path
        else:
            return None
