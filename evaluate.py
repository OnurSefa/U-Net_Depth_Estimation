from unet_model import UNet
from loader import DepthDataset, Normalizer
from torch.utils.data import DataLoader
import torchvision.transforms as T
import torch


def evaluate(model, out_dir, normalizer_paths, image_names_path):
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device('cpu')

    model = model.to(device)
    model.eval()

    cpu = torch.device('cpu')
    transform = T.ToPILImage(mode='L')
    transform_rgb = T.ToPILImage(mode='RGB')

    # normalizer_image = Normalizer(normalizer_paths[0], normalizer_paths[1])
    # normalizer_depth = Normalizer(normalizer_paths[2], normalizer_paths[3])
    dataset = DepthDataset(image_names_path, device=device, with_names=True)
    data_loader = DataLoader(dataset, batch_size=16, shuffle=False)

    for i, data in enumerate(data_loader):
        images, depths, paths = data
        depth_predictions = model(images).to(cpu)
        # depth_predictions = normalizer_depth.denormalize(depth_predictions)
        for im_index in range(depth_predictions.shape[0]):
            im = transform(depth_predictions[im_index, :, :, :].reshape((1, 480, 640)))
            im.save(f'{out_dir}/{i}_{im_index}_prediction.png')
            im = transform(depths[im_index, :, :, :].reshape((1, 480, 640)))
            im.save(f'{out_dir}/{i}_{im_index}_depths.png')
            im = transform_rgb(images[im_index, :, :, :].reshape((3, 480, 640)))
            im.save(f'{out_dir}/{i}_{im_index}_original.png')
            print(f'{i}-{im_index}')
        break


if __name__ == '__main__':
    m = torch.load('./models/001_2_240.pth')
    norm_paths = ['data/image_means.pth', 'data/image_stds.pth', 'data/depth_means.pth', 'data/depth_stds.pth']
    evaluate(m, './data/nyu2_evaluate', norm_paths, 'data/test_names.json')

