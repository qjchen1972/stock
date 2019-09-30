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

__all__ = ["Img_Data"]

class Img_Data(Dataset):
	def __init__(self, img_filepath, label_filepath, transform=None):
    
		self.list_image = []
		self.list_label = []
		self.transform = transform
		
		with open(label_filepath, "r") as lf:			
			while True:                
				line = lf.readline()
				if not line:break
				
				items = line.split()
				if not items: break            
				img = os.path.join(img_filepath, items[0])
				if os.path.exists(img):
					self.list_image.append(img)
					self.list_label.append(int(items[1]))
					
	def __getitem__(self, index):        
		image_path = self.list_image[index]
		image_data = Image.open(image_path).convert('L')
		image_label= self.list_label[index]
		#image_label= torch.FloatTensor([self.list_label[index]])
		if self.transform != None: 
			image_data = self.transform(image_data)
		return image_data, image_label

	def __len__(self):        
		return len(self.list_label)

