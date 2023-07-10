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

import net

import numpy as np

from PIL import Image
import glob
from math import log10


def dehaze_image(image_path):

	data_hazy = Image.open(image_path)
	# h,w = data_hazy.size

	data_hazy = data_hazy.resize((512, 512), Image.ANTIALIAS)

	data_hazy = (np.asarray(data_hazy)/255.0)

	data_hazy = torch.from_numpy(data_hazy).float()

	data_hazy = data_hazy.permute(2, 0, 1)
	data_hazy = data_hazy.cuda().unsqueeze(0)
	#data_hazy = data_hazy.unsqueeze(0)

	#dehaze_net = net.dehaze_net()
	dehaze_net = net.dehaze_net().cuda()
	dehaze_net.load_state_dict(torch.load('E:/研二/自己的去雾论文/论文-神经网络/Weight/Ours/O-hazy/Epoch1.pth'))

	#print(torch.load('snapshotsAod/Epoch15.pth'))
	clean_image = dehaze_net(data_hazy)


	#torchvision.utils.save_image(torch.cat((data_hazy, clean_image), 0), "results/" + image_path.split("/")[-1])
	torchvision.utils.save_image(clean_image, "results/" + image_path.split("/")[-1])

if __name__ == '__main__':

	# 单图
	dehaze_image("test_image/Input.png")
	print("done!")

	# test_list = glob.glob("test_image/*")
	# print(test_list)
	#
	# for image in test_list:
	# 	dehaze_image(image)
	# print(image, "done!")
