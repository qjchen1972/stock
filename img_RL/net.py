# -*- coding: utf-8 -*-
from __future__ import division
import torch
from torch import nn
import torch.nn.functional as F
from math import floor
from collections import OrderedDict
from math import sqrt

class Baseline_Model(nn.Module):
	def __init__(self, imgvec_size,infovec_size,num_actions,hidden_size):
		super(Baseline_Model, self).__init__()
		self.imgvec_size = imgvec_size
		self.infovec_size = infovec_size
		self.num_actions = num_actions
		self.hidden_size = hidden_size

        
		#self.linear1 = nn.Linear(self.imgvec_size , self.hidden_size)
		self.linear2 = nn.Linear(self.infovec_size , self.hidden_size)
		self.linear3 = nn.Linear(self.hidden_size, self.hidden_size)
		self.linear4 = nn.Linear(self.hidden_size, self.num_actions)
        
	def forward(self, infovec):
        
		#x1 = self.linear1(imgvec)
		#x1 = F.relu(x1)
		x2 = self.linear2(infovec)
		act = F.relu6(x2)
		#act = torch.cat((x1,x2),1)
		act = self.linear3(act)
		act = F.relu6(act)
		act = self.linear4(act)        
		return act




