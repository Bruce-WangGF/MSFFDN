import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as utils
from torch.utils.data import DataLoader
from torch.autograd import Variable

import math
from math import log10
from math import exp




def to_psnr(dehaze, gt):
    mse = F.mse_loss(dehaze, gt, reduction='none')
    mse_split = torch.split(mse, 1, dim=0)
    mse_list = [torch.mean(torch.squeeze(mse_split[ind])).item() for ind in range(len(mse_split))]

    intensity_max = 1.0
    psnr_list = [10.0 * log10(intensity_max / mse) for mse in mse_list]
    return psnr_list


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return [ssim_map.mean()]
    else:
        return [ssim_map.mean(1).mean(1).mean(1)]


def ssim(img1, img2, window_size=11, size_average=True):
    img1 = torch.clamp(img1, min=0, max=1)
    img2 = torch.clamp(img2, min=0, max=1)
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    return _ssim(img1, img2, window, window_size, channel, size_average)

# def print_log(epoch, train_psnr, Avgloss,loss1, loss2):
#     # --- Write the training log --- #
#     with open('output/log.txt', 'a') as f:
#         print(
#             'Date: {0}s, Epoch: {1}, Train_PSNR: {2:.2f}, Loss: {3:.6f}, Loss1: {4:.6f}, Loss2: {5:.6f}'
#             .format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
#                     epoch, train_psnr, Avgloss, loss1, loss2), file=f)

def print_log(epoch, train_psnr, avgloss):
    # --- Write the training log --- #
    with open('output/log.txt', 'a') as f:
        print(
            'Date: {0}s, Epoch: {1}, Train_PSNR: {2:.2f}, Loss: {3:.6f}'
                .format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                        epoch, train_psnr, avgloss), file=f)



