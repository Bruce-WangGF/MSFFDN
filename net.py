import torch
import torch.nn as nn
import math
from functools import partial
from collections import OrderedDict
import numpy as np
import torch.nn.functional as F

class PALayer(nn.Module):
    def __init__(self, channel):
        super(PALayer, self).__init__()
        self.pa = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0),
            nn.ReLU(),
            nn.Conv2d(channel // 8, 1, 1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.pa(x)
        return x * y

class CALayer(nn.Module):
    def __init__(self, channel):
        super(CALayer, self).__init__()
        #### Done:self.avg_pool = nn.AdaptiveAvgPool2d(1) # 指定输出map的size
        self.ca = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0),
            nn.ReLU(),
            nn.Conv2d(channel // 8, channel, 1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        avg_pool = nn.AvgPool2d((x.shape[-2], x.shape[-1]))
        y = avg_pool(x)
        y = self.ca(y)
        return x * y

class DehazeBlock(nn.Module):
    def __init__(self, kernel_size, in_chan, out_chan):
        super(DehazeBlock, self).__init__()
        self.relu = nn.LeakyReLU()

        self.conv1 = nn.Conv2d(in_chan, out_chan, kernel_size, padding=kernel_size//2, bias=True)
        self.conv2 = nn.Conv2d(in_chan, out_chan, kernel_size, padding=kernel_size//2, bias=True)
        self.conv3 = nn.Conv2d(in_chan, out_chan, kernel_size, padding=kernel_size//2, bias=True)
        self.conv4 = nn.Conv2d(in_chan, out_chan, kernel_size, padding=kernel_size//2, bias=True)
        self.conv5 = nn.Conv2d(in_chan, out_chan, kernel_size, padding=kernel_size//2, bias=True)
        self.conv6 = nn.Conv2d(in_chan, out_chan, kernel_size, padding=kernel_size//2, bias=True)

        self.calayer = CALayer(in_chan)
        self.palayer = PALayer(in_chan)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.relu(self.conv2(x1))
        x2 = x2 + x
        x3 = self.conv3(x2)
        x4 = self.relu(self.conv4(x3))
        x4 = x4 + x2
        x5 = self.conv5(x4)
        x5 = self.conv6(x5)
        res = self.calayer(x5)
        res = self.palayer(res)
        res += x
        return res
class ResidualBlock(torch.nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.relu = nn.PReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        # out = self.conv2(out) * 0.1
        out = self.conv2(out)
        out = torch.add(out, residual)
        return out

class EncoderBlock(nn.Module):
    def __init__(self, kernel_size, in_chan, out_chan):
        super(EncoderBlock, self).__init__()
        self.relu = nn.LeakyReLU()
        self.conv1 = nn.Conv2d(in_chan, out_chan, kernel_size=kernel_size, stride=1, padding=kernel_size//2)  # /2
        self.conv2 = nn.Conv2d(in_chan, out_chan, kernel_size=kernel_size, stride=1, padding=kernel_size//2)
        self.conv3 = nn.Conv2d(in_chan, out_chan, kernel_size=kernel_size, stride=1, padding=kernel_size//2)
        self.conv4 = nn.Conv2d(in_chan, out_chan, kernel_size=kernel_size, stride=1, padding=kernel_size//2)
        self.conv5 = nn.Conv2d(in_chan, out_chan, kernel_size=kernel_size, stride=1, padding=kernel_size//2)
        #
        # self.conv1_ = nn.Conv2d(in_chan, out_chan, kernel_size=3, stride=1, padding=1)
        # self.conv2_ = nn.Conv2d(in_chan, out_chan, kernel_size=3, stride=1, padding=1)
        # self.conv3_ = nn.Conv2d(in_chan, out_chan, kernel_size=3, stride=1, padding=1)
        # self.conv4_ = nn.Conv2d(in_chan, out_chan, kernel_size=3, stride=1, padding=1)
        # self.conv5_ = nn.Conv2d(in_chan, out_chan, kernel_size=1, stride=1, padding=0)
        # self.conv6_ = nn.Conv2d(in_chan, out_chan, kernel_size=3, stride=1, padding=1)
        self.calayer = CALayer(in_chan)
        self.palayer = PALayer(in_chan)
    def forward(self, x):
        x1 = self.conv1(x)

        x2 = self.conv2(x + x1)
        x3 = self.conv3(x + x1 + x2)
        x4 = self.relu(self.conv4(x3))
        x5 = self.conv5(x + x4)
        res = self.calayer(x5)
        res = self.palayer(res)
        res += x
        return res
        # x1_ = self.BN(self.conv1_(x5))
        #
        # x2_ = self.BN(self.conv2_(x1_))
        # x2_ = self.BN(self.conv5_(x2_))
        #
        # x3_ = self.BN(self.conv3_(x5))
        # x4_ = self.BN(self.conv4_(x3_))
        # x4_ = self.BN(self.conv5_(x4_))

        # x6 = self.relu(self.BN(self.conv6_(x5 + x2_ + x4_)))

class FeatureStract(nn.Module):
    def __init__(self, kernel_size, in_chan, out_chan):
        super(FeatureStract, self).__init__()
        self.relu = nn.LeakyReLU()

        self.conv1 = nn.Conv2d(in_chan, out_chan, kernel_size=kernel_size, stride=1, padding=kernel_size//2)
        self.conv2 = nn.Conv2d(in_chan, out_chan, kernel_size=kernel_size, stride=1, padding=kernel_size//2)
        self.conv3 = nn.Conv2d(in_chan, out_chan, kernel_size=kernel_size, stride=1, padding=kernel_size//2)
        self.conv4 = nn.Conv2d(in_chan, out_chan, kernel_size=kernel_size, stride=1, padding=kernel_size//2)
        self.conv5 = nn.Conv2d(in_chan, out_chan, kernel_size=1, stride=1, padding=0)
        self.conv5_ = nn.Conv2d(in_chan, out_chan, kernel_size=1, stride=1, padding=0)
        self.conv6 = nn.Conv2d(in_chan, out_chan, kernel_size=kernel_size, stride=1, padding=kernel_size//2)

    def forward(self, x):
        x1 = self.relu(self.conv1(x))
        x1 = self.relu(self.conv2(x1))
        x1 = self.relu(self.conv5(x1))

        x2 = self.relu(self.conv3(x))
        x2 = self.relu(self.conv4(x2))
        x2 = self.relu(self.conv5_(x2))
        x3 = self.relu(self.conv6(x1+x2+x))
        return x3

class ConvXX(nn.Module):
    def __init__(self, kernel_size, in_chan, out_chan):
        super(ConvXX, self).__init__()
        self.relu = nn.PReLU()
        self.conv1 = nn.Conv2d(in_chan, out_chan, kernel_size=kernel_size, stride=1, padding=kernel_size//2)
        self.conv2 = nn.Conv2d(in_chan, out_chan, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        res = x
        x1 = self.relu(self.conv1(x))
        x1 = self.relu(self.conv2(x1))
        x1 += res
        return x1

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        reflection_padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        # out = self.conv2d(x)
        return out


class UpsampleConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
      super(UpsampleConvLayer, self).__init__()
      self.conv2d = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=kernel_size//2, output_padding=kernel_size//2)

    def forward(self, x):
        out = self.conv2d(x)
        return out

class dehaze_net(nn.Module):

    def __init__(self):
        super(dehaze_net, self).__init__()
        # CNN
        self.relu = nn.PReLU()
        inchannels = 8
        outchannels = 8

        # input,output
        self.convFirst = nn.Conv2d(3, outchannels, kernel_size=11, stride=1, padding=5)


#----------------------------------------------------------------------------------------------
        self.EncoderBlock1 = EncoderBlock(3, inchannels << 0, outchannels << 0)
        self.EncoderBlock2 = EncoderBlock(3, inchannels << 1, outchannels << 1)
        self.EncoderBlock3 = EncoderBlock(3, inchannels << 2, outchannels << 2)
        self.EncoderBlock4 = EncoderBlock(3, inchannels << 3, outchannels << 3)
        self.EncoderBlock5 = EncoderBlock(3, inchannels << 4, outchannels << 4)

        self.conWgf1 = ConvLayer(inchannels << 0, outchannels << 1, 3, 2)
        self.conWgf2 = ConvLayer(inchannels << 1, outchannels << 2, 3, 2)
        self.conWgf3 = ConvLayer(inchannels << 2, outchannels << 3, 3, 2)
        self.conWgf4 = ConvLayer(inchannels << 3, outchannels << 4, 3, 2)
        self.conWgf5 = ConvLayer(inchannels << 4, outchannels << 5, 3, 2)


#----------------------------------------------------------------------------------------------
        self.ConvXX1 = ConvXX(11, inchannels << 0, outchannels << 0)
        self.ConvXX2 = ConvXX(9, inchannels << 1, outchannels << 1)
        self.ConvXX3 = ConvXX(7, inchannels << 2, outchannels << 2)
        self.ConvXX4 = ConvXX(5, inchannels << 3, outchannels << 3)
        self.ConvXX5 = ConvXX(3, inchannels << 4, outchannels << 4)
# ----------------------------------------------------------------------------------------------
        self.FeatureStract1 = FeatureStract(3, inchannels << 0, outchannels << 0)
        self.FeatureStract2 = FeatureStract(3, inchannels << 1, outchannels << 1)
        self.FeatureStract3 = FeatureStract(3, inchannels << 2, outchannels << 2)
        self.FeatureStract4 = FeatureStract(3, inchannels << 3, outchannels << 3)
        self.FeatureStract5 = FeatureStract(3, inchannels << 4, outchannels << 4)
# ----------------------------------------------------------------------------------------------

        self.dense1 = nn.Sequential(
            ResidualBlock(inchannels << 0),
            ResidualBlock(inchannels << 0),
            ResidualBlock(inchannels << 0),
            ResidualBlock(inchannels << 0)
        )
        self.dense2 = nn.Sequential(
            ResidualBlock(inchannels << 1),
            ResidualBlock(inchannels << 1),
            ResidualBlock(inchannels << 1),
            ResidualBlock(inchannels << 1)
        )
        self.dense3 = nn.Sequential(
            ResidualBlock(inchannels << 2),
            ResidualBlock(inchannels << 2),
            ResidualBlock(inchannels << 2),
            ResidualBlock(inchannels << 2)
        )
        self.dense4 = nn.Sequential(
            ResidualBlock(inchannels << 3),
            ResidualBlock(inchannels << 3),
            ResidualBlock(inchannels << 3),
            ResidualBlock(inchannels << 3)
        )
        self.dense5 = nn.Sequential(
            ResidualBlock(inchannels << 4),
            ResidualBlock(inchannels << 4),
            ResidualBlock(inchannels << 4),
            ResidualBlock(inchannels << 4)
        )
        self.dense6 = nn.Sequential(
            ResidualBlock(inchannels << 4),
            ResidualBlock(inchannels << 4),
            ResidualBlock(inchannels << 4),
            ResidualBlock(inchannels << 4)
        )
        self.dense7 = nn.Sequential(
            ResidualBlock(inchannels << 3),
            ResidualBlock(inchannels << 3),
            ResidualBlock(inchannels << 3),
            ResidualBlock(inchannels << 3)
        )
        self.dense8 = nn.Sequential(
            ResidualBlock(inchannels << 2),
            ResidualBlock(inchannels << 2),
            ResidualBlock(inchannels << 2),
            ResidualBlock(inchannels << 2)
        )
        self.dense9 = nn.Sequential(
            ResidualBlock(inchannels << 1),
            ResidualBlock(inchannels << 1),
            ResidualBlock(inchannels << 1),
            ResidualBlock(inchannels << 1)
        )
        self.dense10 = nn.Sequential(
            ResidualBlock(inchannels << 0),
            ResidualBlock(inchannels << 0),
            ResidualBlock(inchannels << 0),
            ResidualBlock(inchannels << 0)
        )
# ----------------------------------------------------------------------------------------------
        self.Deconv64 = UpsampleConvLayer(inchannels << 5, outchannels << 4, kernel_size=3, stride=2)
        self.Dehazeblock5 = DehazeBlock(3, inchannels << 4, outchannels << 4)
        self.Deconv128 = UpsampleConvLayer(inchannels << 4, outchannels << 3, kernel_size=3, stride=2)
        self.Dehazeblock6 = DehazeBlock(3, inchannels << 3, outchannels << 3)
        self.Deconv256 = UpsampleConvLayer(inchannels << 3, outchannels << 2, kernel_size=3, stride=2)
        self.Dehazeblock7 = DehazeBlock(3, inchannels << 2, outchannels << 2)
        self.Deconv512 = UpsampleConvLayer(inchannels << 2, outchannels << 1, kernel_size=3, stride=2)
        self.Dehazeblock8 = DehazeBlock(3, inchannels << 1, outchannels << 1)
        self.Deconv1024 = UpsampleConvLayer(inchannels << 1, outchannels << 0, kernel_size=3, stride=2)
        self.Dehazeblock9 = DehazeBlock(3, inchannels << 0, outchannels << 0)
# ----------------------------------------------------------------------------------------------
        self.convFinal = nn.Conv2d(inchannels, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        # input
        x1 = self.convFirst(x)  # [8,512,512]


        x2 = self.EncoderBlock1(x1)
        x2 = self.dense1(x2)
        x2 = self.conWgf1(x2)  # [16,256,256]

        x3 = self.EncoderBlock2(x2)
        x3 = self.dense2(x3)
        x3 = self.conWgf2(x3)  # [32,128,128]

        x4 = self.EncoderBlock3(x3)
        x4 = self.dense3(x4)
        x4 = self.conWgf3(x4)  # [64,64,64]

        x5 = self.EncoderBlock4(x4)
        x5 = self.dense4(x5)
        x5 = self.conWgf4(x5)  # [128,32,32]

        x6 = self.EncoderBlock5(x5)
        x6 = self.dense5(x6)
        x6 = self.conWgf5(x6)  # [256,16,16]




        xConxx5 = self.ConvXX5(x5)  # [128, 32, 32]
        x7 = self.Deconv64(x6)  # [128,32,32]
        xFeatureSt5 = self.FeatureStract5(xConxx5 + x5 + x7)  # [128,32,32]
        x7 = self.Dehazeblock5(x5 + xFeatureSt5)
        x7 = self.dense6(x7)


        xConxx4 = self.ConvXX4(x4)  # [128, 32, 32]
        x8 = self.Deconv128(x7)  # [128,32,32]
        xFeatureSt4 = self.FeatureStract4(xConxx4 + x4 + x8)  # [128,32,32]
        x8 = self.Dehazeblock6(x4 + xFeatureSt4)
        x8 = self.dense7(x8)

        xConxx3 = self.ConvXX3(x3)  # [128, 32, 32]
        x9 = self.Deconv256(x8)  # [128,32,32]
        xFeatureSt3 = self.FeatureStract3(xConxx3 + x3 + x9)  # [128,32,32]
        x9 = self.Dehazeblock7(x3 + xFeatureSt3)
        x9 = self.dense8(x9)

        xConxx2 = self.ConvXX2(x2)  # [128, 32, 32]
        x10 = self.Deconv512(x9)  # [128,32,32]
        xFeatureSt2 = self.FeatureStract2(xConxx2 + x2 + x10)  # [128,32,32]
        x10 = self.Dehazeblock8(x2 + xFeatureSt2)
        x10 = self.dense9(x10)

        xConxx1 = self.ConvXX1(x1)  # [128, 32, 32]
        x11 = self.Deconv1024(x10)  # [128,32,32]
        xFeatureSt1 = self.FeatureStract1(xConxx1 + x1 + x11)  # [128,32,32]
        x11 = self.Dehazeblock9(x1 + xFeatureSt1)
        x11 = self.dense10(x11)

        x12 = self.convFinal(x11)

        return x12+x
