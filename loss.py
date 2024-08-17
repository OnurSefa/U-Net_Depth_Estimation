import torch
import torch.nn.functional as F


def depth_loss(y_true, y_pred, alpha=0., beta=0., theta=0., max_depth_val=1.):
    # point-wise depth
    l_depth = torch.mean(torch.abs(y_true - y_pred))

    # edge loss
    dy_true, dx_true = image_gradients(y_true)
    dy_pred, dx_pred = image_gradients(y_pred)
    l_edges = torch.mean(torch.abs(dy_pred - dy_true) + torch.abs(dx_pred - dx_true))

    # SSIM
    l_ssim = torch.mean(torch.clamp((1 - ssim(y_true, y_pred, max_depth_val)) * 0.5, 0, 1))


    return (alpha * l_depth) + (beta * l_edges) + (theta * l_ssim)


def image_gradients(x):
    left = x
    right = F.pad(x, [0, 1, 0, 0])[:, :, :, 1:]
    top = x
    bottom = F.pad(x, [0, 0, 0, 1])[:, :, 1:, :]

    dx, dy = right - left, bottom - top

    dx[:, :, :, -1] = 0
    dy[:, :, -1, :] = 0

    return dy, dx


def ssim(im0, im1, max_val=1.0, filter_size=11, filter_sigma=1.5, k0=0.01, k1=0.03):
    c0 = (k0 * max_val) ** 2
    c1 = (k1 * max_val) ** 2

    mu0 = F.conv2d(im0, gauss_kernel(filter_size, filter_sigma).unsqueeze(0).unsqueeze(0).to(im0.device), padding=filter_size//2)
    mu1 = F.conv2d(im1, gauss_kernel(filter_size, filter_sigma).unsqueeze(0).unsqueeze(0).to(im1.device), padding=filter_size//2)

    mu0_sq = mu0.pow(2)
    mu1_sq = mu1.pow(2)
    mu01 = mu0 * mu1

    sigma0_sq = F.conv2d(im0 * im0, gauss_kernel(filter_size, filter_sigma).unsqueeze(0).unsqueeze(0).to(im0.device), padding=filter_size//2) - mu0_sq
    sigma1_sq = F.conv2d(im1 * im1, gauss_kernel(filter_size, filter_sigma).unsqueeze(0).unsqueeze(0).to(im1.device), padding=filter_size//2) - mu1_sq
    sigma01 = F.conv2d(im0 * im1, gauss_kernel(filter_size, filter_sigma).unsqueeze(0).unsqueeze(0).to(im0.device), padding=filter_size//2) - mu01

    return ((2 * mu01 + c0) * (2 * sigma01 + c1)) / ((mu0_sq + mu1_sq + c0) * (sigma0_sq + sigma1_sq + c1))


def gauss_kernel(size, sigma):
    x = torch.arange(-size // 2 + 1, size // 2 + 1, dtype=torch.float32)
    x = x.view(1, -1).repeat(size, 1)
    y = x.t()
    kernel = torch.exp(-(x.pow(2) + y.pow(2))) / (2 * sigma ** 2)
    return kernel / kernel.sum()


if __name__ == '__main__':
    from loader import Normalizer, DepthDataset
    from torch.utils.data import DataLoader

    normalizer_image = Normalizer('data/image_means.pth', 'data/image_stds.pth')
    normalizer_depth = Normalizer('data/depth_means.pth', 'data/depth_stds.pth')
    dataset = DepthDataset('data/train_names.json', normalizer_image.normalize, normalizer_depth.normalize)
    data_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    for i, data in enumerate(data_loader):
        images, depths = data
        loss = depth_loss(depths, depths)
