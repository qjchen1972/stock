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
#import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, confusion_matrix,roc_curve, auc
import time
import math
import torch.onnx
from torch.optim.lr_scheduler import ReduceLROnPlateau

from data_proc import * 
from models import *
from torch.autograd import Variable

logging.basicConfig(level=logging.INFO, format='%(message)s')

loss_log = 'answer.txt'

class Train_Proc():
	
	@staticmethod
	def validate(model, data_loader, loss,isgpu):   		
		
		model.eval()
		out_label = torch.FloatTensor().cuda()
		out_prect = torch.FloatTensor().cuda()
		#total_loss = 0
		for i, (input, target) in enumerate(data_loader):
			if isgpu : 
				input = input.cuda(async=True)
				target = target.cuda(async=True)
			output = model(input)
			lossvalue = loss(output,target)
			#total_loss += lossvalue.item()
			tt = torch.unsqueeze(target, 1).float()
			out_label = torch.cat((out_label, tt), 0)
			out_prect = torch.cat((out_prect, output.data), 0)
			
			# compute loss & accuracy 
        	predicted = torch.max(out_prect, 1)[1]
        	label = torch.max(out_label, 1)[0].long()
			total = len(label)
        	correct = int((predicted == label).sum())
        	acc = 100 * (1.0*correct / total)
	
		t00 =0
		t01 =0
		t02 = 0
		t03 = 0
		t04 = 0
		t05 = 0
		
		t10 =0
                t11 =0
                t12 = 0
                t13 = 0
                t14 = 0
                t15 = 0
		
		t20 =0
                t21 =0
                t22 = 0
                t23 = 0
                t24 = 0
                t25 = 0
		
		t30 =0
                t31 =0
                t32 = 0
                t33 = 0
                t34 = 0
                t35 = 0

		t40 =0
                t41 =0
                t42 = 0
                t43 = 0
                t44 = 0
                t45 = 0

		t50 =0
                t51 =0
                t52 = 0
                t53 = 0
                t54 = 0
                t55 = 0
	
                for i in range(len(label)):
			if label[i] == 0:
                                if predicted[i] == 0:t00 += 1
                                elif predicted[i] == 1:t01 += 1
                                elif predicted[i] == 2:t02 += 1
                                elif predicted[i] == 3:t03 += 1
                                elif predicted[i] == 4:t04 += 1
                                else:t05 +=1

                        if label[i] == 1:
                                if predicted[i] == 0:t10 += 1
                                elif predicted[i] == 1:t11 += 1
                                elif predicted[i] == 2:t12 += 1
                                elif predicted[i] == 3:t13 += 1
                                elif predicted[i] == 4:t14 += 1
                                else:t15 +=1

                        if label[i] == 2:
                                if predicted[i] == 0:t20 += 1
                                elif predicted[i] == 1:t21 += 1
                                elif predicted[i] == 2:t22 += 1
                                elif predicted[i] == 3:t23 += 1
                                elif predicted[i] == 4:t24 += 1
                                else:t25 +=1

                        if label[i] == 3:
                                if predicted[i] == 0:t30 += 1
                                elif predicted[i] == 1:t31 += 1
                                elif predicted[i] == 2:t32 += 1
                                elif predicted[i] == 3:t33 += 1
                                elif predicted[i] == 4:t34 += 1
                                else:t35 +=1

                        if label[i] == 4:
                                if predicted[i] == 0:t40 += 1
                                elif predicted[i] == 1:t41 += 1
                                elif predicted[i] == 2:t42 += 1
                                elif predicted[i] == 3:t43 += 1
                                elif predicted[i] == 4:t44 += 1
                                else:t45 +=1

                        if label[i] == 5:
                                if predicted[i] == 0:t50 += 1
                                elif predicted[i] == 1:t51 += 1
                                elif predicted[i] == 2:t52 += 1
                                elif predicted[i] == 3:t53 += 1
                                elif predicted[i] == 4:t54 += 1
                                else:t55 +=1

		acc0 = 1.0*t00/(t00+t01+t02+t03+t04+t05+0.00000001)
		acc1 = 1.0*t11/(t10+t11+t12+t13+t14+t15+0.00000001)
		acc2 = 1.0*t22/(t20+t21+t22+t23+t24+t25+0.00000001)
		acc3 = 1.0*t33/(t30+t31+t32+t33+t34+t35+0.00000001)
		acc4 = 1.0*t44/(t40+t41+t42+t43+t44+t45+0.00000001)
		acc5 = 1.0*t55/(t50+t51+t52+t53+t54+t55+0.00000001)
		a5= 1.0 * t55/(t05+t15+t25+t35+t45+t55+0.000001)
		return acc,acc0,acc1,acc2,acc3,acc4,acc5,a5

	@staticmethod
	def epoch_train(model, data_loader, optimizer, loss,isgpu):

		model.train()
		loss_train = 0
	    
		for batch_id, (input, target) in enumerate(data_loader):
			if isgpu: input = input.cuda(async=True)
			if isgpu: target = target.cuda(async=True)
			var_input = torch.autograd.Variable(input)
			var_target = torch.autograd.Variable(target)
			var_output =  model(var_input)
			lossvalue = loss(var_output, var_target)
			loss_train += lossvalue.item()
			optimizer.zero_grad()
			lossvalue.backward()
			optimizer.step()
			temploss = loss_train/(batch_id + 1)
			if batch_id % 500 == 0 :
				logging.info(' %d : training_loss = %.6f  whole loss = %.6f' % ( batch_id, lossvalue.item(),temploss ))
		if isgpu:torch.cuda.empty_cache()
		return temploss

	@staticmethod     
	def train(img_path, label_path, model_path, net, batch_size, max_epoch, checkpoint,lr,iscpu):
		
		isgpu = not iscpu and torch.cuda.is_available()
		if isgpu: cudnn.benchmark = True
		rtime = time.time()

		if net == 'STOCK_NET':
			model = StockNet(6)
		if not isgpu:
			model = model.cpu()
			#model = torch.nn.DataParallel(model)
			torch.manual_seed(int(rtime))
		else: 
			model = model.cuda()
			#model = torch.nn.DataParallel(model).cuda()
			torch.cuda.manual_seed(int(rtime))
	
		print(model)

		kwargs = {'num_workers': 16, 'pin_memory': True} if isgpu else {}
		transformList = []
		transformList.append(transforms.ToTensor())
		transformSequence=transforms.Compose(transformList)

		data = Img_Data(img_filepath = img_path, label_filepath = label_path,transform=transformSequence)
		data_loader = DataLoader(dataset = data,batch_size=batch_size,shuffle=True,**kwargs)

		val_kwargs = {'num_workers': 16, 'pin_memory': True} if isgpu else {}
		val_transformList = []
		val_transformList.append(transforms.ToTensor())
		val_transformSequence=transforms.Compose(val_transformList)
		valdata = Img_Data(img_filepath = '/home/max/img_dst',label_filepath = '/home/max/stock_vallabel.txt',transform=val_transformSequence)
		val_data_loader = DataLoader(dataset = valdata,batch_size=32,shuffle=False,**val_kwargs)

		
		loss = torch.nn.CrossEntropyLoss()
		
		#loss = torch.nn.BCELoss(size_average = True)
		
		#optimizer = optim.SGD(model.parameters(), lr=lr,momentum=0.9)
		optimizer = optim.Adam(model.parameters(), lr=lr,weight_decay=0,amsgrad=True)
			
		if checkpoint == True:
			modelCheckpoint = torch.load(model_path)
			model.load_state_dict(modelCheckpoint['state_dict'])
			#optimizer.load_state_dict(modelCheckpoint['optimizer'])
			start = modelCheckpoint['epoch'] + 1
		else:
			start = 0
		
		for epoch_id in range(start, max_epoch):
			oldtime = time.time()                        
			loss_train = Train_Proc.epoch_train(model, data_loader, optimizer,loss,isgpu)
			newtime = time.time()            
			torch.save({'epoch':epoch_id, 'state_dict':model.state_dict(), 'optimizer':optimizer.state_dict()}, 
				'./models/'+'model-'+str(epoch_id)+'.pth.tar')			
			totaltime = newtime - oldtime
			val_sum,acc0,acc1,acc2,acc3,acc4,acc5,sum5 = Train_Proc.validate(model,val_data_loader,loss,isgpu)
			logging.info('Epoch %d : training_loss =  %.6f  time = %d   val = %.6f, %.6f, %.6f, %.6f, %.6f, %.6f,%.6f, %.6f' % ( epoch_id, loss_train , totaltime,val_sum,sum5,acc0,acc1,acc2,acc3,acc4,acc5))
			with open(loss_log,'a+') as ftxt:		
				ftxt.write('\nEpoch %d : training_loss =  %.6f  time = %d  val1 = %.6f, %.6f,%.6f, %.6f, %.6f, %.6f,%.6f, %.6f\n' % ( epoch_id, loss_train , totaltime,val_sum,sum5,acc0,acc1,acc2,acc3,acc4,acc5))

	@staticmethod
	def computeAUROC (dataGT, dataPRED, classCount):

       		outAUROC = [] 
		datanpGT = dataGT.cpu().numpy()
		datanpPRED = dataPRED.cpu().numpy()
		for i in range(classCount):
			outAUROC.append(roc_auc_score(datanpGT[:, i], datanpPRED[:, i]))
		return outAUROC
	
	@staticmethod		
	def test(img_path, label_path, model_path, net, batch_size, iscpu):                 
		
		isgpu = not iscpu and torch.cuda.is_available()
		if isgpu : cudnn.benchmark = True

		if net == 'STOCK_NET':
			model = StockNet(6)

		if not isgpu :
			model = model.cpu()
			#model = torch.nn.DataParallel(model)
			modelCheckpoint = torch.load(model_path,map_location='cpu')
		else: 
			model = model.cuda()
			#model = torch.nn.DataParallel(model).cuda()
			modelCheckpoint = torch.load(model_path)
		
		model.load_state_dict(modelCheckpoint['state_dict'])

		kwargs = {'num_workers': 16, 'pin_memory': True} if isgpu else {}
		transformList = []
		transformList.append(transforms.ToTensor())
		transformSequence=transforms.Compose(transformList)

		data = Img_Data(img_filepath = img_path, label_filepath = label_path, transform=transformSequence)
		data_loader = DataLoader(dataset = data,batch_size=batch_size,shuffle=False,**kwargs)

		model.eval()	
		
		out_label = torch.FloatTensor().cuda()
		out_prect = torch.FloatTensor().cuda()
		good = 0
		bad = 0
		real_good = 0
		fake_bad = 0
				
		for i, (input, target) in enumerate(data_loader):
			if isgpu : 
				target = target.cuda(async=True)
				input = input.cuda(async=True)

			#input_var = torch.autograd.Variable(input,volatile=True)
			output = model(input)
			tt = torch.unsqueeze(target, 1).float()
                        out_label = torch.cat((out_label, tt), 0)
			out_prect = torch.cat((out_prect, output.data), 0)
			
                # compute loss & accuracy 
                predicted = torch.max(out_prect, 1)[1]
                label = torch.max(out_label, 1)[0].long()
		t00 = 0
		t01 = 0
		t02 = 0
		t03 = 0
		t04 = 0
		t05 = 0
		
		t10 = 0
		t11 = 0
		t12 =0
		t13 = 0
		t14 = 0
		t15 = 0
		
		
		t20 = 0
                t21 = 0
                t22 =0
                t23 = 0
                t24 = 0
		t25 = 0
		
		t30 = 0
                t31 = 0
                t32 =0
                t33 = 0
                t34 = 0
		t35 = 0

		t40 = 0
                t41 = 0
                t42 =0
                t43 = 0
                t44 = 0
		t45 = 0

		t50 = 0
                t51 = 0
                t52 =0
                t53 = 0
                t54 = 0
		t55 = 0
		
		for i in range(len(label)):
			if label[i] == 0:
				if predicted[i] == 0:t00 += 1
				elif predicted[i] == 1:t01 += 1
				elif predicted[i] == 2:t02 += 1
				elif predicted[i] == 3:t03 += 1
				elif predicted[i] == 4:t04 += 1
				else:t05 +=1

			if label[i] == 1:
                                if predicted[i] == 0:t10 += 1
                                elif predicted[i] == 1:t11 += 1
                                elif predicted[i] == 2:t12 += 1
                                elif predicted[i] == 3:t13 += 1
				elif predicted[i] == 4:t14 += 1
                                else:t15 +=1

			if label[i] == 2:
                                if predicted[i] == 0:t20 += 1
                                elif predicted[i] == 1:t21 += 1
                                elif predicted[i] == 2:t22 += 1
                                elif predicted[i] == 3:t23 += 1
				elif predicted[i] == 4:t24 += 1
                                else:t25 +=1

			if label[i] == 3:
                                if predicted[i] == 0:t30 += 1
                                elif predicted[i] == 1:t31 += 1
                                elif predicted[i] == 2:t32 += 1
                                elif predicted[i] == 3:t33 += 1
				elif predicted[i] == 4:t34 += 1
                                else:t35 +=1

			if label[i] == 4:
                                if predicted[i] == 0:t40 += 1
                                elif predicted[i] == 1:t41 += 1
                                elif predicted[i] == 2:t42 += 1
                                elif predicted[i] == 3:t43 += 1
				elif predicted[i] == 4:t44 += 1
                                else:t45 +=1

			if label[i] == 5:
                                if predicted[i] == 0:t50 += 1
                                elif predicted[i] == 1:t51 += 1
                                elif predicted[i] == 2:t52 += 1
                                elif predicted[i] == 3:t53 += 1
                                elif predicted[i] == 4:t54 += 1
                                else:t55 +=1

		u0 = 1.0 *t50/(t00+t10+t20+t30+t40+t50+0.000001)
		u1 = 1.0 *t51/(t01+t11+t21+t31+t41+t51+0.000001)
		u2 = 1.0 *t52/(t02+t12+t22+t32+t42+t52+0.000001)
		u3 = 1.0 *t53/(t03+t13+t23+t33+t43+t53+0.000001)
		u4 = 1.0 *t54/(t04+t14+t24+t34+t44+t54+0.000001)
		u5 = 1.0 *t55/(t05+t15+t25+t35+t45+t55+0.000001)
		
		print('         0        1        2        3        4        5')
		print('\n')
		print('0     %4d     %4d     %4d     %4d     %4d     %4d' %(t00,t01,t02,t03,t04,t05))
		print('\n')
		print('1     %4d     %4d     %4d     %4d     %4d     %4d' %(t10,t11,t12,t13,t14,t15))
		print('\n')
		print('2     %4d     %4d     %4d     %4d     %4d     %4d' %(t20,t21,t22,t23,t24,t25))
		print('\n')
		print('3     %4d     %4d     %4d     %4d     %4d     %4d' %(t30,t31,t32,t33,t34,t35))
		print('\n')
		print('4     %4d     %4d     %4d     %4d     %4d     %4d' %(t40,t41,t42,t43,t44,t45))
		print('\n')
		print('5     %4d     %4d     %4d     %4d     %4d     %4d' %(t50,t51,t52,t53,t54,t55))
		print('\n')
		print('      %.2f     %.2f     %.2f     %.2f     %.2f     %.2f' %(u0,u1,u2,u3,u4,u5))

  
                #label = out_label
                total = len(label)
                #print( total, label)
		#print(predicted)
                correct = int((predicted == label).sum())
                acc = 100 * (1.0*correct / total)
		uu = 1.0 *t55/(t05+t15+t25+t35+t45+t55)
		yy = 1.0 * ((t55-t05)*3 +(t45 - t15) * 2 +(t35 - t25)*0.5) /(t05+t15+t25+t35+t45+t55)
		print('acc = ' ,acc, ' rigth =', uu, 'yy =  ',yy)
		
		return

	@staticmethod
	def predict(img_path, label_path, model_path, net, batch_size, iscpu):   

		isgpu = not iscpu and torch.cuda.is_available()
		if isgpu: cudnn.benchmark = True

		if net == 'STOCK_NET':
			model = StockNet(6)
		
		if not isgpu :
			model = model.cpu()
			#model = torch.nn.DataParallel(model)
			modelCheckpoint = torch.load(model_path,map_location='cpu')
		else: 
			model = model.cuda()
			#model = torch.nn.DataParallel(model).cuda()
			modelCheckpoint = torch.load(model_path)
		
		model.load_state_dict(modelCheckpoint['state_dict'])

		kwargs = {'num_workers': 1, 'pin_memory': True} if isgpu else {}
		transformList = []
		transformList.append(transforms.ToTensor())
		transformSequence=transforms.Compose(transformList)

		data = Img_Data(img_filepath = img_path, label_filepath = label_path, transform=transformSequence)
		data_loader = DataLoader(dataset = data,batch_size=batch_size,shuffle=False,**kwargs)

		model.eval()
		
		out_label = torch.FloatTensor().cuda()
		out_prect = torch.FloatTensor().cuda()
		
		for i, (input, target) in enumerate(data_loader):
			if isgpu : input = input.cuda(async=True)
			#input_var = torch.autograd.Variable(input,volatile=True)
			output = model(input)
			#out_label = torch.cat((out_label, target), 0)
			out_prect = torch.cat((out_prect, output.data), 0)
		
		 # save the output
        	predicted = torch.max(out_prect, 1)[1]
       		Y = np.reshape(predicted, -1)
        	np.savetxt('pred.csv', Y, delimiter=",", fmt='%5s')

	@staticmethod
	def create_onnx(net,model_path,iscpu):

		isgpu = not iscpu and torch.cuda.is_available()
		
		if net == 'STOCK_NET':
			model = StockNet(6)	
		if not isgpu:
			#model = torch.nn.DataParallel(model)
			modelCheckpoint = torch.load(model_path,map_location='cpu')
			dummy_input = Variable(torch.randn(1, 1, 128, 128))
		else:
			model = model.cuda()
            		#model = torch.nn.DataParallel(model).cuda()		
			dummy_input = Variable(torch.randn(1, 1, 128, 128)).cuda()
			modelCheckpoint = torch.load(model_path)
			#print('it is gpu')

		model.load_state_dict(modelCheckpoint['state_dict'])
		input_names = [ "input" ] 
		output_names = [ "output" ]
		torch.onnx.export(model, dummy_input, 'k4/stock.proto', verbose=True, input_names=input_names, output_names=output_names)
		

		
