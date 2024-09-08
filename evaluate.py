from unet_model import UNet
from loader import DepthDataset
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision.transforms.functional import hflip
import torch
import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


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


def create_comparison_table(model_path_original, model_path_fine_tuned, data_indices, image_names_path, out_path):
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device('cpu')

    dataset = DepthDataset(image_names_path, device=device, normalize=True, mode='inference')
    models = []
    model_names = []
    if model_path_original:
        model = torch.load(model_path_original, map_location=device).eval()
        models.append(model)
        model_names.append("Raw Dataset Model")
    if model_path_fine_tuned:
        model = torch.load(model_path_fine_tuned, map_location=device).eval()
        models.append(model)
        model_names.append("Fine-Tuned Model")
    cpu = torch.device('cpu')

    transform = T.ToPILImage(mode='L')
    transform_rgb = T.ToPILImage(mode='RGB')

    height_ratios = [0.1]
    for _ in data_indices:
        height_ratios.append(4)

    fig, axs = plt.subplots(len(data_indices) + 1, 2+len(models), figsize=(8*(2+len(models)), len(data_indices)*6+0.6), gridspec_kw={'height_ratios': height_ratios})
    plt.subplots_adjust(hspace=0.05, wspace=0.05)
    axs[0, 0].text(0.5, 0.5, 'RGB Image', ha='center', va='center', fontweight='bold')
    axs[0, 1].text(0.5, 0.5, 'Ground Truth', ha='center', va='center', fontweight='bold')
    for i in range(len(model_names)):
        axs[0, 2+i].text(0.5, 0.5, model_names[i], ha='center', va='center', fontweight='bold')
    for ax in axs[0]:
        ax.axis('off')

    for i, data_index in enumerate(data_indices):
        data = dataset[data_index]
        image, image_f, image_o, depth, path = data
        input_images = []
        for im in [image, image_f]:
            input_images.append(im.unsqueeze(0))
        axs[i+1, 0].imshow(transform_rgb(image_o))
        axs[i+1, 0].axis('off')

        depth = (depth - torch.min(depth)) / (torch.max(depth) - torch.min(depth))
        axs[i+1, 1].imshow(transform(depth), cmap='viridis')
        axs[i+1, 1].axis('off')

        for m, model in enumerate(models):
            depth_predictions = torch.zeros_like(depth).to(cpu).unsqueeze(0)
            for image_type, input_image in enumerate(input_images):
                if image_type == 1:
                    depth_predictions += hflip(model(input_image)).to(cpu)
                else:
                    depth_predictions += model(input_image).to(cpu)
            depth_predictions /= len(input_images)
            depth_predictions = (depth_predictions - torch.min(depth_predictions)) / (torch.max(depth_predictions) - torch.min(depth_predictions))
            output_img = transform(depth_predictions[0, :, :, :])
            axs[i+1, 2+m].imshow(output_img, cmap='viridis')
            axs[i+1, 2+m].axis('off')

    plt.savefig(out_path, dpi=300, bbox_inches='tight')

if __name__ == '__main__':
    # mps = []
    # names = ["037_19_100"]
    # for name in names:
    #     mps.append(f"../data/depth_models/{name}.pth")
    #
    # evaluate('../data/depth_data/evaluate_finetune', mps, '../data/depth_data/fine_tune_test.json')

    create_comparison_table('../data/depth_models/030_14_4500.pth', None, [124, 132, 206, 357, 358], '../data/depth_data/original_paths_portion_test.json', '../data/depth_data/evaluate.png')
    create_comparison_table(None, '../data/depth_models/037_19_100.pth', [24, 74, 114, 167, 172], '../data/depth_data/fine_tune_test.json', '../data/depth_data/evaluate_finetune.png')
    create_comparison_table('../data/depth_models/030_14_4500.pth', '../data/depth_models/037_19_100.pth', [24, 74, 114, 167, 172], '../data/depth_data/fine_tune_test.json', '../data/depth_data/final_comparison_fine_tune_data.png')
    create_comparison_table('../data/depth_models/030_14_4500.pth', '../data/depth_models/037_19_100.pth', [124, 132, 206, 357, 358], '../data/depth_data/original_paths_portion_test.json', '../data/depth_data/final_comparison_raw_data.png')