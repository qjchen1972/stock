# -*- coding:utf-8 -*-
from __future__ import print_function
from __future__ import division
import os
import numpy as np
from PIL import Image,ImageEnhance
import torch
import  torchvision
from torch.utils.data import Dataset
import struct
try:
    import accimage
except ImportError:
    accimage = None
import random
from io import BytesIO

class ExperienceReplay(object):

	def __init__(self, max_memory=100, discount=.9):
		self.max_memory = max_memory
		#self.memory = list()
		self.memory = {}
		self.discount = discount
	

	def remember(self, name,states, game_over):

		self.memory[name] = [states, game_over]
		#print('memory is %d' %(len(self.memory)))
		#self.memory.append([states, game_over])
		#if len(self.memory) > self.max_memory:
			#del self.memory[0]
	

	def get_batch(self, model, device_type,batch_size=10):

		len_memory = len(self.memory)
		num_actions = model.num_actions
		#print(type(self.memory[0]))
		#print(type(self.memory[0][0]))
		#print(self.memory[0][0][0].shape)
		#print(self.memory[0][0][1].shape)
		#print(self.memory[0][0][2])
		#print(self.memory[0][0][3])
		#print(self.memory[0][0][4].shape)
		#print(self.memory[0][0][5].shape)

		#env_dim1 = self.memory[0][0][0].shape
		#env_dim2 = self.memory[0][0][1].shape

		#input_img = np.zeros((min(len_memory, batch_size),)+env_dim1)
		#input_info = np.zeros((min(len_memory, batch_size),)+env_dim1)
		input_info = np.zeros((min(len_memory, batch_size),8))
		#print(input_img.shape,input_info.shape)

		targets = np.zeros((input_info.shape[0], num_actions))
		#print(targets.shape)
    
		#for i, idx in enumerate(np.random.randint(0, len_memory,size=inputs.shape[0])):
		i = 0 
		for key in random.sample(self.memory.keys(),input_info.shape[0]):
			#print('get %d %d'  %(i, idx))
			#print(' %d   %s' %(i,key))
       
			#old_img,old_info,action_t, reward_t, now_img,now_info = self.memory[idx][0]
			old_info,action_t, reward_t, now_info = self.memory[key][0]
			game_over = self.memory[key][1]
			#input_img[i:i+1] = old_img
			input_info[i:i+1] = old_info
			#urimg = old_img.reshape((1,)+old_img.shape)
			urinfo = old_info.reshape((1,)+old_info.shape) 
			#targets[i] = model(torch.Tensor(urimg).to(device),torch.Tensor(urinfo).to(device)).cpu().detach().numpy()[0]
			targets[i] = model(torch.Tensor(urinfo).double().to(device_type)).cpu().detach().numpy()[0]
			print('orain target:',targets[i])
			print(urinfo)
			
			#p = model(torch.Tensor(urimg).to(device),torch.Tensor(urinfo).to(device)).detach().cpu().numpy()[0]
			#print(targets[i])
			#print(p)	
			#urimg = now_img.reshape((1,)+now_img.shape)
			#urimg = torch.Tensor(urimg).to(device)
			#urinfo = now_info.reshape((1,)+now_info.shape)
			#urinfo = torch.Tensor(urinfo).double().to(device_type)
			#p1 = model(urinfo).cpu().detach().numpy()[0]
			#print(urinfo)
			#p1 = model(torch.Tensor(urimg).float().to(device),torch.Tensor(urinfo).float().to(device)).cpu().detach().numpy()[0]
			#print(p)
			#Q_sa = np.max(p1)
			#print(Q_sa)
			if game_over: 
				targets[i, action_t] = reward_t
				print('game is over  %.4f' %(reward_t))
				
			else:
				urinfo = now_info.reshape((1,)+now_info.shape)
				urinfo = torch.Tensor(urinfo).double().to(device_type)
				p1 = model(urinfo).cpu().detach().numpy()[0]
				print('game con')
				print(p1)
				Q_sa = np.max(p1)
				print(Q_sa)
				targets[i, action_t] = reward_t + self.discount * Q_sa
			i += 1
		return input_info,targets
