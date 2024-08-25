from unet_model import UNet
from loader import DepthDataset, Normalizer
from torch.utils.data import DataLoader
import torchvision.transforms as T
import torch
import os


def evaluate(out_dir, model_paths, image_names_path):
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device('cpu')

    cpu = torch.device('cpu')

    transform = T.ToPILImage(mode='L')
    transform_rgb = T.ToPILImage(mode='RGB')

    dataset = DepthDataset(image_names_path, device=device, with_names=True, normalize=False)
    data_loader = DataLoader(dataset, batch_size=4, shuffle=False)

    for i, data in enumerate(data_loader):
        images, depths, paths = data
        for im_index in range(images.shape[0]):
            im = transform(depths[im_index, :, :, :].reshape((1, 230, 310)))
            im.save(f'{out_dir}/{i}_{im_index}_depths.png')
            im = transform_rgb(images[im_index, :, :, :].reshape((3, 460, 620)))
            im.save(f'{out_dir}/{i}_{im_index}_original.png')
        for m, model_path in enumerate(model_paths):
            print(model_path)
            model = torch.load(model_path)
            model = model.to(device)
            model.eval()
            depth_predictions = model(images).to(cpu)
            for im_index in range(depth_predictions.shape[0]):
                depth_prediction = depth_predictions[im_index, :, :, :].reshape((1, 230, 310))
                depth_prediction = (depth_prediction - depth_prediction.min()) / (depth_prediction.max() - depth_prediction.min())
                im = transform(depth_prediction)
                im.save(f'{out_dir}/{i}_{im_index}_prediction_{m:02}.png')
                print(f'{i}-{im_index}-{m}')
        if i == 20:
            break


if __name__ == '__main__':
    mps = []
    for name in os.listdir("models"):
        if '.DS_Store' in name:
            continue
        mps.append(f"models/{name}")
    mps = ['models/023_1_480.pth']

    evaluate('./data/nyu2_evaluate', mps, 'test_names.json')

