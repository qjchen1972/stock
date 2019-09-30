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



class StockNet(nn.Module):

    def __init__(self, classCount):
	
        super(StockNet, self).__init__()		
	self.net = DenseNet(num_init_features=32, growth_rate=16,block_config=(5,10,14,12), bn_size=4, num_classes=classCount)
		
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

    def forward(self, x,feel):
        out = self.net(x,feel)
	return out

