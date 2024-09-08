from unet_model import UNet
from loader import DepthDataset
from torch.utils.data import DataLoader
from loss import depth_loss
import torch.optim as o
import torch
import mlflow
import random


def train(model, optimizer, loss_function, epoch_count, model_dir, model_prefix, save_interval, batch_size, image_names_path, shuffle=True, alpha=0., beta=0., theta=0.):

    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print('device is mps')
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print('device is cuda')
    else:
        device = torch.device('cpu')
        print('device is cpu')

    model = model.to(device)
    model.train()

    dataset = DepthDataset(image_names_path, device=device, mode='train')
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    save_loss = 0
    epoch_loss = 0
    save_iteration = 0
    epoch_iteration = 0
    step = 0

    for epoch in range(epoch_count):
        for i, data in enumerate(data_loader):
            optimizer.zero_grad()
            images, depths = data
            depth_predictions = model(images)
            loss = loss_function(depths, depth_predictions, alpha, beta, theta)
            loss.backward()
            optimizer.step()

            print(f'{epoch} - {i} - loss: {loss.item():.10f}')

            mlflow.log_metric('batch loss', loss.item(), step=step)

            save_loss += loss.item()
            epoch_loss += loss.item()
            save_iteration += 1
            epoch_iteration += 1
            if i % save_interval == 0 and i != 0:
                torch.save(model, f'{model_dir}/{model_prefix}_{epoch}_{i}.pth')
                mlflow.log_metric('save loss', save_loss / save_iteration, step=step)
                print(f"SAVE LOSS {epoch} - {i}: {save_loss/save_iteration}")
                save_loss = 0
                save_iteration = 0
                print()
            step += 1
        mlflow.log_metric('epoch loss', epoch_loss/epoch_iteration, step=step)
        epoch_loss = 0
        epoch_iteration = 0


if __name__ == '__main__':
    lf = depth_loss
    ec = 20
    md = '../data/depth_models'
    si = 100
    bs = 8
    im_names_path = '../data/depth_data/fine_tune_train.json'

    for mp_index in range(31, 40):
        # m = UNet()
        m = torch.load('../data/depth_models/030_14_4500.pth')
        learning_rate = 10 ** random.uniform(-5, -3)
        # learning_rate = 0.00033
        optim = o.Adam(m.parameters(), lr=learning_rate)

        a = 10 ** random.uniform(-3, -1)
        b = 10 ** random.uniform(-3, -1)
        c = 10 ** random.uniform(-3, -1)

        # a = 0.02139
        # b = 0.11385
        # c = 0.28639
        mp = f'{mp_index:03}'
        mlflow.start_run(run_name=f'{mp}')
        mlflow.log_param('learning rate', learning_rate)
        mlflow.log_param('batch size', bs)
        mlflow.log_param('loss function', 'depth-gradient-ssim')
        mlflow.log_param('model name', mp)
        mlflow.log_param('model initiated', '030_14_4500')
        mlflow.log_param('alpha', a)
        mlflow.log_param('beta', b)
        mlflow.log_param('theta', c)

        mlflow.log_param('notes', 'fine tune')
        train(m, optim, lf, ec, md, mp, si, bs, im_names_path, True, a, b, c)
        mlflow.end_run()
