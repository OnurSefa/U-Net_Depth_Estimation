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

    dataset = DepthDataset(image_names_path, device=device)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    save_loss = 0
    save_iteration = 0

    for epoch in range(epoch_count):
        for i, data in enumerate(data_loader):
            optimizer.zero_grad()
            images, depths = data
            depth_predictions = model(images)
            loss = loss_function(depths, depth_predictions)
            loss.backward()
            optimizer.step()

            print(f'{epoch} - {i} - loss: {loss.item():.4f}')
            save_loss += loss.item()
            save_iteration += 1
            if i % save_interval == 0:
                torch.save(model, f'{model_dir}/{model_prefix}_{epoch}_{i}.pth')
                print(f"SAVE LOSS {epoch} - {i}: {save_loss/save_iteration}")
                save_loss = 0
                save_iteration = 0
                print()


if __name__ == '__main__':
    m = UNet()
    learning_rate = 0.1
    optim = o.Adam(m.parameters(), lr=learning_rate)
    lf = depth_loss
    ec = 10
    md = 'models'
    mp = '001'
    si = 30
    bs = 16
    im_names_path = 'data/train_names.json'
    norm_paths = ['data/image_means.pth', 'data/image_stds.pth', 'data/depth_means.pth', 'data/depth_stds.pth']
    train(m, optim, lf, ec, md, mp, si, bs, im_names_path, norm_paths)

