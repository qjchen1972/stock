#!/usr/bin/env python
# -*- coding:utf-8 -*-
from __future__ import print_function
from __future__ import division
import os
import numpy as np
import time
import sys
import logging
from torch.autograd.gradcheck import *
from torch.nn.parameter import Parameter
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.backends.cudnn as cudnn
import math
import argparse

from train_proc import Train_Proc

logging.basicConfig(level=logging.INFO, format='%(message)s')

parser = argparse.ArgumentParser(description='chenqianjiang  model')
parser.add_argument('-m', type=int, default=0, metavar='mode', help='input run mode (default: 0)')
parser.add_argument('-c', action='store_true', default=False, help='disables CUDA training')

def main():
	args = parser.parse_args()
	print(args.m,args.c)

	if args.m == 0:run_train(args.c)
	elif args.m == 1:run_test(args.c)
	elif args.m == 2:run_predict(args.c)
	elif args.m == 3:run_onnx(args.c)
	else: draw_train()

def run_train( iscpu ):    
	img_path = 'data/mini/trainimg'
	label_path = 'data/real/train.txt'
	val_path = 'data/real/val.txt'
	model_path = './models/model-665.pth.tar'
	batch_size = 64
	checkpoint = True
	lr = 0.0001
	max_epoch = 5000
	Train_Proc.train(img_path, label_path, val_path,model_path, batch_size, max_epoch, checkpoint,lr,iscpu)

def run_test( iscpu ):    
	#img_path = 'data/test/testimg'
	#label_path = 'data/test/val.txt'
	img_path = 'data/train/trainimg'
	label_path = 'data/real/label.txt'
	model_path = './models/model-3.pth.tar'
	batch_size = 32
	Train_Proc.test(img_path, label_path, model_path, batch_size, iscpu)

def run_predict( iscpu ):    
	#img_path = 'data/test/testimg'
	#label_path = 'data/test/label_20190423.txt'
	img_path = 'data/train/trainimg'
	label_path = 'data/real/label.txt'
	model_path = './models/model-266.pth.tar'
	batch_size = 32
	Train_Proc.predict(img_path, label_path, model_path, batch_size, iscpu)

def run_onnx(iscpu):
	model_path = './k4/model-169.pth.tar'
	Train_Proc.create_onnx(model_path,iscpu)

def draw_train():
        file = 'answer.txt'
        Train_Proc.draw_train_roc(file)


if __name__ == '__main__':
	main()
