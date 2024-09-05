from unet_model import UNet
from loader import DepthDataset
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision.transforms.functional import hflip
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

    dataset = DepthDataset(image_names_path, device=device, normalize=True, mode='inference')
    data_loader = DataLoader(dataset, batch_size=4, shuffle=False)

    models = []
    for model_path in model_paths:
        model = torch.load(model_path, map_location=device).eval()
        models.append(model)

    for i, data in enumerate(data_loader):
        images, images_f, image_o, depths, paths = data
        input_images = []
        # channel_changers = [[0, 1, 2], [1, 0, 2], [0, 2, 1], [2, 1, 0]]
        channel_changers = [[0, 1, 2]]
        for im in [images, images_f]:
            for channel_changer in channel_changers:
                input_images.append(im[:, channel_changer, :, :])
        for im_index in range(images.shape[0]):
            im = transform(depths[im_index, :, :, :].reshape((1, 240, 320)))
            im.save(f'{out_dir}/{i}_{im_index}_depths.png')
            im = transform_rgb(image_o[im_index, :, :, :].reshape((3, 480, 640)))
            im.save(f'{out_dir}/{i}_{im_index}_original.png')
        for m, model in enumerate(models):
            depth_predictions = torch.zeros_like(depths).to(cpu)
            for image_type, input_image in enumerate(input_images):
                if image_type >= len(channel_changers):
                    depth_predictions += hflip(model(input_image[:, channel_changers[image_type-len(channel_changers)], :, :])).to(cpu)
                else:
                    depth_predictions += model(input_image[:, channel_changers[image_type], :, :]).to(cpu)
            depth_predictions /= 8
            for im_index in range(depth_predictions.shape[0]):
                depth_prediction = depth_predictions[im_index, :, :, :].reshape((1, 240, 320))
                depth_prediction = (depth_prediction - depth_prediction.min()) / (depth_prediction.max() - depth_prediction.min())
                im = transform(depth_prediction)
                im.save(f'{out_dir}/{i}_{im_index}_depths_{m:02}.png')
                print(f'{i}-{im_index}-{m}')
        if i == 100:
            break


if __name__ == '__main__':
    mps = []
    names = ["030_10_2700", "030_12_3300", "030_14_300", "030_14_4500"]
    for name in names:
        if '.DS_Store' in name:
            continue
        mps.append(f"../data/depth_models/{name}.pth")

    evaluate('../data/depth_data/evaluate', mps, '../data/depth_data/original_paths_portion_test.json')

