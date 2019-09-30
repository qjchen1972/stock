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

class Catch(object):

	def __init__(self,imgDir,labelFilePath,transform=None,timeStep = 5):
		self.timeStep = timeStep
		self.listImage = []
		self.listPrice = []

		self.transform = transform
		self.index = 0
		self.step = 0
		self.money = 1.0
		
		with open(labelFilePath, "r") as lf:			
			while True:                
				line = lf.readline()
				if not line:break				
				items = line.split()
				if not items: break            
				imgPath = os.path.join(imgDir, items[0])
				if os.path.exists(imgPath):
					self.listImage.app end(imgPath)
					self.listPrice.append([float(i)/100.0 for i in items[1:]])
		
		npp = np.array(self.listPrice)
		#print(npp.shape)
		self.price_max = []
		self.price_max.append(np.max(npp[:,0]))
		self.price_max.append(np.max(npp[:,1]))
		self.price_max.append(np.max(npp[:,2]))
		self.price_max.append(np.max(npp[:,3]))
		print(self.price_max)
		npmax = np.array(self.price_max)
		#print(npp)
		npp = npp / npmax
		self.listPrice = npp.tolist()
		#print(self.listPrice)
		#self.listPrice = self.listPrice / self.price_max

	def _draw_state(self):
		
		im_size = (self.grid_size,self.timeStep)
	    canvas = np.zeros(im_size)

        canvas[state[0], state[1]] = 1  # draw fruit
        canvas[-1, state[2]-1:state[2] + 2] = 1  # draw basket
        return canvas
	

	

	def reset(self,num = -1):
		if num < 0:
			#self.index = np.random.randint(0, len(self.listPrice) - self.timeStep, size=1)
			self.index = np.random.randint(0, 5, size=1)
		else:
			self.index = num
		self.step = 0
		#self.imgData,price = self._getStockInfo(self.index)
		price = self._getStockInfo(self.index)
		self.moneyInfo =[self.money,self.money,self.money,0,price[0],price[1],price[2],price[3]]

	def _is_over(self):
		if self.step == self.timeStep -1:return True
		else: return False

	def observe(self):
		#return self.imgData.numpy(), np.array(self.moneyInfo)
		return np.array(self.moneyInfo)

	def getimg(self):
		return self.listImage[int(self.index)]

	def _get_reward(self):
		#return (self.moneyInfo[1]  - self.moneyInfo[0])*100.0
		#if self.moneyInfo[1]  - self.moneyInfo[0] >= 0: return 1.0
		#else: return 0.0
		#return self.moneyInfo[0] + self.moneyInfo[1]*self.moneyInfo[2] - self.money
		if self.step == self.timeStep - 1:
			#if self.moneyInfo[1] == 4:return 1
			#else: return 0
			return (self.moneyInfo[1]  - self.moneyInfo[0] + self.moneyInfo[0] - 1.0)*100.0
			return self.moneyInfo[0] + self.moneyInfo[1]*self.moneyInfo[2] - self.money
		else:
			#return self.moneyInfo[1]  - self.moneyInfo[0]
			return 0.0

	def _update_state(self, action):
	
		self.index += 1
		self.step += 1
                #self.imgData,price = self._getStockInfo(self.index)
		price = self._getStockInfo(self.index)
                #self.moneyInfo[4:] = price
	
		self.moneyInfo[0] = self.moneyInfo[1]
		need = float(action - self.moneyInfo[3])
		self.moneyInfo[3] += need
		self.moneyInfo[2] -= need*self.moneyInfo[4]
		self.moneyInfo[4:] = price
		self.moneyInfo[1] = self.moneyInfo[3] * self.moneyInfo[4] + self.moneyInfo[2]
		
				

	def act(self,action):
		#reward = self._get_reward()
		#game_over = self._is_over()
		#if game_over: action = 0		
		self._update_state(action)
		reward = self._get_reward()
		game_over = self._is_over()
		#img,info = self.observe()
		#print(img.size(),info.size())
		info = self.observe()
		return info, reward, game_over
		 

	def _getStockInfo(self,index):
		index = int(index)
		#imagePath = self.listImage[index]
		#imageData = Image.open(imagePath).convert('RGB')
		#if self.transform != None: 
			#imageData = self.transform(imageData)
		price= self.listPrice[index]
		return price		



