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

__all__ = ["StockData"]

class StockData(Dataset):
    def __init__(self, imgDir, labelFilePath, transform=None,istrain = False):
    
        self.listImage = []
        self.listFeel = []
	self.listLabel = []
	self.transform = transform
        self.istrain = istrain
	self.imgAugmenter = ImageAugment()

	with open(labelFilePath, "r") as lf:			
	    while True:                
	        line = lf.readline()
		if not line:break				
		items = line.split()
		if not items: break            
		imgPath = os.path.join(imgDir, items[0])
		if os.path.exists(imgPath):
		    self.listImage.append(imgPath)
                    self.listFeel.append([int(i) for i in items[1:5]])
		    self.listLabel.append([int(i) for i in items[5:]])
                    

					
    def __getitem__(self, index):        
        imagePath = self.listImage[index]
	imageData = Image.open(imagePath).convert('RGB')
        feelData = torch.FloatTensor(self.listFeel[index])
        imageLabel= torch.FloatTensor(self.listLabel[index])
        #print(index,imageLabel)
        feelData = feelData/ 100.0
        imageLabel = imageLabel/100.0
        #print(index,imageLabel)
        #exit()

        if self.istrain:
	    imageData = self.imgAugmenter.process(imageData)
            
	if self.transform != None: 
	    imageData = self.transform(imageData)
        #tensor_to_pil = torchvision.transforms.Compose([torchvision.transforms.ToPILImage()])
        #pil = tensor_to_pil(imageData)
        #pil.save('%d.jpg' %index )
        #if index>= 10: exit()
	return imageData, feelData,imageLabel

    def __len__(self):        
        return len(self.listLabel)


class ImageAugment:
    def __init__(self):		
        
        self.widthRange = [0.0, 0.3]
        #self.initHeight = 150
        #self.heightRange = [0,1]

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
        #img.save('0.jpg')

        #width, height = img.size
        #newImg = img.resize((width, self.initHeight),Image.BILINEAR)
        #newImg.save('1.jpg')
        
	#newLeft =  int(round(width * random.uniform(*self.widthRange)))
        #print(newLeft,0,newImg.size[0],newImg.size[1])
        #nwidth,nheight = newImg.size

        #newImg = img.crop([newLeft,0,width,height])
        #newImg.save('2.jpg')

	#newImg = newImg.resize((width, height),Image.BILINEAR)
        #newImg.save('3.jpg')

        return newImg



