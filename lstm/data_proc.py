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
import pandas as pd

__all__ = ["StockData"]

class StockData(Dataset):
	def __init__(self,labelFilePath):
		
		self.listStock = []
		self.listLabel = []
		self.veclen = 10
		self.vecscale = 100.0

		with open(labelFilePath, "r") as lf:		
			while True:                
				line = lf.readline()
				if not line:break				
				items = line.split()
				if not items: break
				#print(line)
				list = []
				for i in range(1,5) :
					list.append(int(items[i*10 + 6]))
				list.append(int(items[2]))
				self.listLabel.append(list)
				self.listStock.append([int(i) for i in items[3:]])

                    

					
	def __getitem__(self, index):
		stock = torch.FloatTensor(np.array(self.listStock[index]).reshape(-1,self.veclen))
		stock = stock / self.vecscale
		label = torch.FloatTensor(self.listLabel[index])
		label = label / self.vecscale
		#print(label)
		#F.normalize(label)
		#print(label)
		#exit()	
		return stock, label

	def __len__(self):        
		return len(self.listLabel)


	def data2img(self,index,imageData):
		tensor_to_pil = torchvision.transforms.Compose([torchvision.transforms.ToPILImage()])
		pil = tensor_to_pil(imageData)
		pil.save('%d.jpg' %index )



class ImageAugment:
	def __init__(self):		
        
		self.widthRange = [0.0, 0.3]

	def process(self, img):
		type = int(round(random.uniform(0,10)))
		if type <= 3: return img
		width, height = img.size
		newLeft =  int(round(width * random.uniform(*self.widthRange)))
		newImg = img.crop([newLeft,0,width,height])

		if type <= 6:
			newImg = newImg.resize((width, height),Image.BILINEAR)
		else:
			newImg = newImg.resize((width, height),Image.BICUBIC)
		return newImg



