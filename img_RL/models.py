# -*- coding: utf-8 -*-
from __future__ import division
import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.backends.cudnn as cudnn

import torchvision.transforms as transforms
from torch.autograd import Variable
from math import sqrt
from mobilenetv2 import MobileNetV2
from net import  Baseline_Model


class StockNet(nn.Module):

	def __init__(self, num_actions):
	
		super(StockNet, self).__init__()
		self.vecsize = 128
		self.infosize = 8 
		self.num_actions = num_actions
		#self.MobileNetV2 = MobileNetV2(num_classes= self.vecsize, input_size=128, width_mult=1.)
		self.base =  Baseline_Model(self.vecsize,self.infosize, num_actions,self.vecsize)

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight)
				if m.bias is not None:nn.init.constant_(m.bias, 0)
			elif isinstance(m, nn.Linear):
				nn.init.kaiming_normal_(m.weight)
				if m.bias is not None:nn.init.constant_(m.bias, 0)
			elif isinstance(m, nn.BatchNorm2d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)

	def forward(self, info):
		#out = self.MobileNetV2(img)
		out = self.base(info)
		return out

