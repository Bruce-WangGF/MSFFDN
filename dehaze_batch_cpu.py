import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import os
import sys
import argparse
import time
import dataloader
#import net
# import accv_ori as net
#import accv_CBAM as net
#import accv_SimAM as net
#import accv_SKN as net
#import accv_fca as net
#import accv_ECA as net
import accv_capa as net
#import Resnet.oURS as net
#import Resnet.netTemp3 as net
#import SOTS.netForsOTS as net
#import Resnet.MSBDN as net
#import Resnet.AodNet as net
#import Resnet.FFANet as net
#import Resnet.baseNet as net
#import Resnet.GridNet as net
#import Resnet.LightNet as net
#import numpy as np
#import Resnet.baseNet_DoubleDehaze as net
#import Resnet.baseNet_DD_MSFF as net
#import Resnet.idea2 as net
from torchvision import transforms
from PIL import Image
import glob
import math
import numpy as np
from PIL import Image
from scipy.signal import convolve2d
def PSNR(target, ref):
    target = np.array(target, dtype=np.float32)
    ref = np.array(ref, dtype=np.float32)

    if target.shape != ref.shape:
        raise ValueError('输入图像的大小应该一致！')

    diff = ref - target
    diff = diff.flatten('C')

    rmse = math.sqrt(np.mean(diff ** 2.))
    psnr = 20 * math.log10(np.max(target) / rmse)
    # criterion = torch.nn.MSELoss(size_average=True)
    # mse = criterion(target, ref)
    # psnr = 10 * log10(1 / mse)
    return psnr


# def gaussian(window_size, sigma):
#
#     gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
#     return gauss / gauss.sum()
#
# def create_window(window_size, channel):
#
#     _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
#     _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
#     window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
#     return window
#
# def _ssim(img1, img2, window, window_size, channel, size_average=True):
#
#     mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
#     mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)
#     mu1_sq = mu1.pow(2)
#     mu2_sq = mu2.pow(2)
#     mu1_mu2 = mu1 * mu2
#     sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
#     sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
#     sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2
#     C1 = 0.01 ** 2
#     C2 = 0.03 ** 2
#     ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
#
#     if size_average:
#         return ssim_map.mean()
#     else:
#         return ssim_map.mean(1).mean(1).mean(1)
#
#
# def ssim(img1, img2, window_size=11, size_average=True):
#
#     img1 = torch.clamp(img1, min=0, max=1)
#     img2 = torch.clamp(img2, min=0, max=1)
#     (_, channel, _, _) = img1.size()
#     window = create_window(window_size, channel)
#     if img1.is_cuda:
#         window = window.cuda(img1.get_device())
#     window = window.type_as(img1)
#     return _ssim(img1, img2, window, window_size, channel, size_average)

def SSIM(target, ref, K1=0.01, K2=0.03, gaussian_kernel_sigma=1.5, gaussian_kernel_width=11, L=255):
    # 高斯核，方差为1.5，滑窗为11*11
    gaussian_kernel = np.zeros((gaussian_kernel_width, gaussian_kernel_width))
    for i in range(gaussian_kernel_width):
        for j in range(gaussian_kernel_width):
            gaussian_kernel[i, j] = (1 / (2 * math.pi * (gaussian_kernel_sigma ** 2))) * math.exp(
                -(((i - 5) ** 2) + ((j - 5) ** 2)) / (2 * (gaussian_kernel_sigma ** 2)))

    target = np.array(target, dtype=np.float32)
    ref = np.array(ref, dtype=np.float32)
    if target.shape != ref.shape:
        raise ValueError('输入图像的大小应该一致！')

    target_window = convolve2d(target, np.rot90(gaussian_kernel, 2), mode='valid')
    ref_window = convolve2d(ref, np.rot90(gaussian_kernel, 2), mode='valid')

    mu1_sq = target_window * target_window
    mu2_sq = ref_window * ref_window
    mu1_mu2 = target_window * ref_window

    sigma1_sq = convolve2d(target * target, np.rot90(gaussian_kernel, 2), mode='valid') - mu1_sq
    sigma2_sq = convolve2d(ref * ref, np.rot90(gaussian_kernel, 2), mode='valid') - mu2_sq
    sigma12 = convolve2d(target * ref, np.rot90(gaussian_kernel, 2), mode='valid') - mu1_mu2

    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2
    ssim_array = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    ssim = np.mean(np.mean(ssim_array))

    return ssim

def dehaze_image(image_path,i):

    data_hazy = Image.open(image_path)
    data_hazy = data_hazy.resize((256, 256), Image.ANTIALIAS)
    data_hazy = (np.asarray(data_hazy)/255.0)

    data_hazy = torch.from_numpy(data_hazy).float()
    data_hazy = data_hazy.permute(2,0,1)
    data_hazy = data_hazy.cuda().unsqueeze(0)
    #data_hazy = data_hazy.unsqueeze(0)

    dehaze_net = net.dehaze_net().cuda()
    #dehaze_net = net.dehaze_net()
    dehaze_net.load_state_dict(torch.load('snapshots/Epoch'+str(i)+'.pth'))

    clean_image = dehaze_net(data_hazy)
    # print("results/" + image_path.split("/")[-1])
    #torchvision.utils.save_image(torch.cat((data_hazy, clean_image), 0), "results/" + image_path.split("/")[-1])
    torchvision.utils.save_image(clean_image, "D:/dataset/# I-HAZY NTIRE 2018/test/res/" + image_path.split("/")[-1])
    #torchvision.utils.save_image(clean_image, "Results/" + image_path.split("/")[-1])


if __name__ == '__main__':

    # 单图
    #dehaze_image("test_images/47_hazy.png")
    # print("test_images/2.png", "done!")
    # test_list = glob.glob("E:/dataset/SOTS/nyuhaze500/hazy/*")
    for i in range(99, 100):
        test_list = glob.glob("D:/dataset/# I-HAZY NTIRE 2018/test/hazy512/*")
        #print(test_list)
        for image in test_list:
            dehaze_image(image,i)
            print(image, "done!")

        Sumssim = 0
        Sumpsnr = 0


        for j in range(1, 6):
            # 复原图
            print(j)
            ref = Image.open('D:/dataset/# I-HAZY NTIRE 2018/test/res/hazy512/WGF_' + str(j) + '.jpg').convert('L')
            # 原图
            target = Image.open('D:/dataset/# I-HAZY NTIRE 2018/test/gt512/WGF_' + str(j) + '.jpg').convert('L')
            target = target.resize((256, 256), Image.ANTIALIAS)

            psnr = PSNR(target, ref)
            #print(i, psnr)

            ssim = SSIM(target, ref)
            #print(ssim)

            Sumpsnr += psnr
            Sumssim += ssim

        print(i, Sumssim / 5, Sumpsnr / 5)
        #print("AVGSSIM", Sumssim / 5)
