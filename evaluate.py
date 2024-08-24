from unet_model import UNet
from loader import DepthDataset, Normalizer
from torch.utils.data import DataLoader
import torchvision.transforms as T
import torch


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

    dataset = DepthDataset(image_names_path, device=device, with_names=True)
    data_loader = DataLoader(dataset, batch_size=4, shuffle=False)

    for i, data in enumerate(data_loader):
        images, depths, paths = data
        if i<3:
            continue
        for im_index in range(images.shape[0]):
            im = transform(depths[im_index, :, :, :].reshape((1, 460, 620)))
            im.save(f'{out_dir}/{i}_{im_index}_depths.png')
            im = transform_rgb(images[im_index, :, :, :].reshape((3, 460, 620)))
            im.save(f'{out_dir}/{i}_{im_index}_original.png')
        break
        for m, model_path in enumerate(model_paths):
            model = torch.load(model_path)
            model = model.to(device)
            model.eval()
            depth_predictions = model(images).to(cpu)
            for im_index in range(depth_predictions.shape[0]):
                depth_prediction = depth_predictions[im_index, :, :, :].reshape((1, 480, 640))
                depth_prediction = (depth_prediction - depth_prediction.min()) / (depth_prediction.max() - depth_prediction.min())
                im = transform(depth_prediction)
                im.save(f'{out_dir}/{i}_{im_index}_prediction_{m:02}.png')
                print(f'{i}-{im_index}-{m}')
        break


if __name__ == '__main__':
    mps = ["005_0_400", "006_0_480", "013_0_240", "013_0_480", "014_0_480", "015_0_240", "015_0_480", "019_0_480"]

    for i in range(len(mps)):
        mps[i] = f"models/{mps[i]}.pth"

    evaluate('./data/nyu2_evaluate', mps, 'test_names.json')

