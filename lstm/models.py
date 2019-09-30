# -*- coding: utf-8 -*-
from __future__ import division
import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.backends.cudnn as cudnn

#import torchvision.transforms as transforms
from torch.autograd import Variable
from math import sqrt
from net import DenseNet
from attention import AttnEncoder, AttnDecoder


class StockNet(nn.Module):

	def __init__(self, time_step):
	
		super(StockNet, self).__init__()
		vecsize = 256
		#self.net = DenseNet(num_init_features=32, growth_rate=16,block_config=(5,10,14,12), bn_size=4, num_classes=time_step*vecsize)
		self.encoder = AttnEncoder(input_size=vecsize, hidden_size=512, time_step=time_step)
		self.decoder = AttnDecoder(code_hidden_size=vecsize, hidden_size=512, time_step=time_step)
		
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight)
				if m.bias is not None:nn.init.constant_(m.bias, 0)
			elif isinstance(m, nn.Linear):
				nn.init.kaiming_normal_(m.weight)
				if m.bias is not None:nn.init.constant_(m.bias, 0)
			elif isinstance(m, nn.BatchNorm1d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)

	def forward(self, x):
		#out = self.net(x)
		#print(x)
		#x = F.normalize(x)
		#print(x)
		#exit()
		out = self.encoder(x)
		y_res = self.decoder(out)
		return y_res

