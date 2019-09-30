# -*- coding:utf-8 -*-
from __future__ import print_function
import matplotlib 
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import numpy as np
import logging
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, confusion_matrix,roc_curve, auc
import time
import math
import torch.onnx
from torch.optim.lr_scheduler import ReduceLROnPlateau
from models import *
from torch.autograd import Variable
from catch import *
from experienceReplay import *
logging.basicConfig(level=logging.INFO, format='%(message)s')
loss_log = 'answer.txt'
num_actions = 2
time_step = 3
max_memory = 100
epsilon = 0.1

class Train_Proc():

	@staticmethod
	def train(trainImgDir, trainLabelFilePath, modelFilePath, batchSize, maxEpoch, checkpoint, lr, iscpu = False):
		isgpu = not iscpu and torch.cuda.is_available()
		cudnn.benchmark = True
		#torch.manual_seed(seed)
		device = torch.device("cuda:0" if isgpu else "cpu")

		model = StockNet(num_actions).double().to(device)
		total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
		print(model)
		print('Total number of paramters: %d' % total_params)
		loss_f = nn.MSELoss()
		#optimizer = optim.SGD(model.parameters(), lr=lr,momentum=0.9)
		optimizer = optim.Adam(model.parameters(), lr=lr,weight_decay=0,amsgrad=True)
		if checkpoint == True:
			modelCheckpoint = torch.load(modelFilePath)
			model.load_state_dict(modelCheckpoint['state_dict'])
			optimizer.load_state_dict(modelCheckpoint['optimizer'])
			optimizer.lr = lr
			start = modelCheckpoint['epoch'] + 1
		else:
			start = 0
		
		transformList = []
		transformList.append(transforms.ToTensor())
		transformSequence=transforms.Compose(transformList)
		env = Catch(imgDir = trainImgDir, labelFilePath = trainLabelFilePath,transform=transformSequence,timeStep = time_step)
		exp_replay = ExperienceReplay(max_memory=max_memory)
		oldtime = time.time()
		for epoch_id in range(start, maxEpoch):
			env.reset()
			game_over = False
			#now_img,now_info = env.observe()
			now_info = env.observe()
			#print(type(now_img),now_img.size(),now_img.unsqueeze(0).size())
			#print(type(now_info),now_info.size())
			#print(now_img.shape)
			#now_img = now_img.reshape((1,)+now_img.shape)
			#now_info = now_info.reshape((1,)+now_info.shape)
			#print(now_img.shape)

			#print(now_info.shape)
			#exit()
			batch_loss = 0
			#oldtime = time.time()
			
			#print('\n')
			while not game_over:
				print('---------------------------')
				#old_img,old_info = now_img,now_info
				#print(old_info)
				old_info = now_info
				print(old_info)
				if np.random.rand() <= epsilon:
					action = np.random.randint(0, num_actions, size=1)[0]
					print('rd action:',action)
					urinfo = old_info.reshape((1,)+old_info.shape)
					q = model(torch.Tensor(urinfo).double().to(device))
					print(q)
				else:
					#urimg = old_img.reshape((1,)+old_img.shape)
					urinfo = old_info.reshape((1,)+old_info.shape)
					#print(type(urimg), type(old_img))
					#q = model(torch.Tensor(urimg).double().to(device),torch.Tensor(urinfo).double().to(device))
					q = model(torch.Tensor(urinfo).double().to(device))
					#print(q)
					action = np.argmax(q[0].detach().cpu().numpy())
					print('action:',action)
					print(q)
					#action = 4
				
				#now_info, reward, game_over = env.act(action)
				#print(now_info[0].size(),now_info[1].size())
				#now_img,now_info, reward, game_over = env.act(action)
				name = env.getimg()
				now_info, reward, game_over = env.act(action)
				#print('game_over = ',game_over)
				#print('now:',now_info)
				print('rward:',reward)
				#exp_replay.remember([old_img,old_info, action, reward, now_img,now_info], game_over)
				#if game_over: action = 0
				exp_replay.remember(name,[old_info, action, reward, now_info], game_over)
				input_info,target = exp_replay.get_batch(model=model, batch_size=batchSize,device_type = device)
				print('train   input:',input_info)
				#print(target.shape,target)
				#print(input_img.shape)
				#print(input_info.shape,input_info)
				print(target)
				input_info,target = torch.Tensor(input_info).double().to(device),torch.Tensor(target).double().to(device)
				optimizer.zero_grad()
				output =  model(input_info)
				#print('ans:',output)
				lossValue = loss_f(output, target)
				lossValue.backward()
				optimizer.step()
				batch_loss += lossValue.item()
			#exit()
			#if epoch_id > 5 : exit()
			#total_time = time.time() - oldtime
			if isgpu:torch.cuda.empty_cache()
			if epoch_id % 1 == 0:
				#exit()
				total_time = time.time() - oldtime
				oldtime = time.time()
				torch.save({'epoch':epoch_id, 'state_dict':model.state_dict(), 'optimizer':optimizer.state_dict()}, 
					'./models/'+'model-'+str(epoch_id)+'.pth.tar')
				#logging.info('Epoch %d : training_loss= %.6f,time = %d' % (epoch_id, batch_loss, total_time))
				print('Epoch %d : training_loss= %.6f,time = %d' % (epoch_id, batch_loss, total_time))
				with open(loss_log,'a+') as ftxt:
					ftxt.write('\nEpoch %d : training_loss= %.6f,time = %d\n' % (epoch_id, batch_loss, total_time))


	@staticmethod
	def test(testImgDir, testLabelFilePath, modelFilePath, iscpu = False):
		isgpu = not iscpu and torch.cuda.is_available()
		cudnn.benchmark = True
		device = torch.device("cuda:0" if isgpu else "cpu")
		model = StockNet(num_actions).double().to(device)
		if not isgpu :
			modelCheckpoint = torch.load(modelFilePath,map_location=torch.device('cpu'))
		else: 
			modelCheckpoint = torch.load(modelFilePath)
		model.load_state_dict(modelCheckpoint['state_dict'])
		transformList = []
		transformList.append(transforms.ToTensor())
		transformSequence=transforms.Compose(transformList)
		env = Catch(imgDir = testImgDir, labelFilePath = testLabelFilePath,transform=transformSequence,timeStep = time_step)
		points = 0
		#env.reset()
		index = 0
		for e in range(10):
			loss = 0.
			env.reset(index +e)
			game_over = False
			info = env.observe()
			ur = 0.0
			print('\n')
			while not game_over:
				#urimg = torch.Tensor(img.reshape((1,)+img.shape))
				urinfo = torch.Tensor(info.reshape((1,)+info.shape))
				q = model(urinfo.double().to(device)).detach().cpu().numpy()
				print(q)
				action = np.argmax(q[0])
				print(info)
				info, reward, game_over = env.act(action)
				print('%s -- %d--- %.4f' %(env.getimg(),action,reward))
				#info, reward, game_over = env.act(action)
				points += reward
				ur += reward
			print(info)

			print('%d --- %f' %(e,ur))


		print('total is %f' %(points))
