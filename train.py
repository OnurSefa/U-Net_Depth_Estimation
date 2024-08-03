from unet_model import UNet
from loader import DepthDataset, Normalizer
from torch.utils.data import DataLoader
from loss import depth_loss
import torch.optim as o
import torch


def train(model, optimizer, loss_function, epoch_count, model_dir, model_prefix, save_interval, batch_size, image_names_path, normalizer_paths, shuffle=True):

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device('cpu')

    model = model.to(device)
    model.train()

    normalizer_image = Normalizer(normalizer_paths[0], normalizer_paths[1])
    normalizer_depth = Normalizer(normalizer_paths[2], normalizer_paths[3])
    dataset = DepthDataset(image_names_path, normalizer_image.normalize, normalizer_depth.normalize, device)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    for epoch in range(epoch_count):
        for i, data in enumerate(data_loader):
            optimizer.zero_grad()
            images, depths = data
            depth_predictions = model(images)
            loss = loss_function(depths, depth_predictions)
            loss.backward()
            optimizer.step()

            print(f'{epoch} - {i} - loss: {loss.item():.4f}')
            if i % save_interval == 0:
                torch.save(model, f'{model_dir}/{model_prefix}_{epoch}_{i}.pth')


if __name__ == '__main__':
    m = UNet()
    learning_rate = 0.002
    optim = o.Adam(m.parameters(), lr=learning_rate)
    lf = depth_loss
    ec = 10
    md = 'models'
    mp = '000'
    si = 10
    bs = 16
    im_names_path = 'data/train_names.json'
    norm_paths = ['data/image_means.pth', 'data/image_stds.pth', 'data/depth_means.pth', 'data/depth_stds.pth']
    train(m, optim, lf, ec, md, mp, si, bs, im_names_path, norm_paths)

