import torch
import torch.nn as nn
import torch.optim
import os
import argparse
import dataloader
# from fvcore.nn import FlopCountAnalysis, parameter_count_table
#mport net
#import accv_ori as net
#import accv_CBAM as net
import accv_SimAM as net
#import accv_SKN as net
#import accv_fca as net
#import accv_ECA as net
#import accv_capa as net
from utils import to_psnr, ssim, print_log


# def weights_init(m):
# 	classname = m.__class__.__name__
# 	if classname.find('Conv') != -1:  # 使用了find函数，如果不存在返回值为-1，所以让其不等于-1
# 		m.weight.data.normal_(0.0, 0.02)
# 	elif classname.find('BatchNorm') != -1:
# 		m.weight.data.normal_(1.0, 0.02)
# 		m.bias.data.fill_(0)

def train(config):
    dehaze_net = net.dehaze_net().cuda()


    # device = torch.device('cuda:0')
    # tensor = torch.tensor(torch.rand(1, 3, 256, 256), device=device)
    #
    # flops = FlopCountAnalysis(dehaze_net, tensor)  # FLOPs
    # print("FLOPs: ", flops.total())
    # dehaze_net.apply(weights_init)
    # dehaze_net.load_state_dict(torch.load('E:/cs1/Epoch42.pth'))

    sum_ = 0
    for name, param in dehaze_net.named_parameters():
        mul = 1
        for size_ in param.shape:
            mul *= size_  # 统计每层参数个数
        #print(name, mul)
        sum_ += mul  # 累加每层参数个数
    print('参数个数：', sum_)  # 打印参数量

    train_dataset = dataloader.dehazing_loader(config.orig_images_path, config.hazy_images_path)
    #val_dataset = dataloader.dehazing_loader(config.orig_images_path, config.hazy_images_path, mode="val")
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True,
                                               num_workers=config.num_workers, pin_memory=True)
    #val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config.val_batch_size, shuffle=True,
                                            # num_workers=config.num_workers, pin_memory=True)
    criterion1 = nn.MSELoss().cuda()


    optimizer = torch.optim.Adam(dehaze_net.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.1)
    dehaze_net.train()
    for epoch in range(config.num_epochs):
        epoch_loss = 0
        psnr_list = []


        for iteration, (img_orig, img_haze) in enumerate(train_loader):
            img_orig = img_orig.cuda()
            img_haze = img_haze.cuda()

            clean_image = dehaze_net(img_haze)
            loss = criterion1(clean_image, img_orig)
            epoch_loss += loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(dehaze_net.parameters(), config.grad_clip_norm)
            optimizer.step()
            psnr_list.extend(to_psnr(img_orig, clean_image))
            if ((iteration + 1) % config.display_iter) == 0:
                print("Epoch:", epoch, "Loss at iteration", iteration + 1, ":", loss.item())
            if ((iteration + 1) % config.snapshot_iter) == 0:
                torch.save(dehaze_net.state_dict(), config.snapshots_folder + "Epoch" + str(epoch) + '.pth')
        scheduler.step()
        train_psnr = sum(psnr_list) / len(psnr_list)
        Avgloss = epoch_loss.item() / len(train_loader)

        # Validation Stage
        # psnr_list1 = []
        # ssim_list1 = []
        # for iter_val, (img_orig, img_haze) in enumerate(val_loader):
        #     img_orig = img_orig.cuda()
        #     img_haze = img_haze.cuda()
        #
        #     clean_image = dehaze_net(img_haze)
        #
        # psnr_list1.extend(to_psnr(clean_image, img_orig))
        # ssim_list1.extend(ssim(clean_image, img_orig))
        # avr_psnr = sum(psnr_list1) / len(psnr_list1)
        # avr_ssim = sum(ssim_list1) / len(ssim_list1)
        # print_log(epoch, train_psnr, Avgloss, avr_psnr, avr_ssim)
        print_log(epoch, train_psnr, Avgloss)
        # torchvision.utils.save_image(torch.cat((img_haze, clean_image, img_orig), 0), config.sample_output_folder+str(iter_val+ 1)+".jpg")

        torch.save(dehaze_net.state_dict(), config.snapshots_folder + "dehazer.pth")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Input Parameters
    parser.add_argument('--orig_images_path', type=str, default="D:/dataset/train_test/train/GT/")
    parser.add_argument('--hazy_images_path', type=str, default="D:/dataset/train_test/train/hazy/")

    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--grad_clip_norm', type=float, default=0.25)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--train_batch_size', type=int, default=1)
    parser.add_argument('--val_batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--display_iter', type=int, default=100)
    parser.add_argument('--snapshot_iter', type=int, default=100)
    parser.add_argument('--snapshots_folder', type=str, default="snapshots/")
    parser.add_argument('--sample_output_folder', type=str, default="samples/")


    config = parser.parse_args()

    if not os.path.exists(config.snapshots_folder):
        os.mkdir(config.snapshots_folder)
    if not os.path.exists(config.sample_output_folder):
        os.mkdir(config.sample_output_folder)

    train(config)









