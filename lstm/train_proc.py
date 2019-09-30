# -*- coding:utf-8 -*-
from __future__ import print_function
import matplotlib 
#matplotlib.use('Agg')
matplotlib.use('Qt5Agg')
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
from sklearn.metrics import roc_auc_score, confusion_matrix,roc_curve, auc,classification_report
#import scikitplot as skplt
import time
import math
import torch.onnx
from torch.optim.lr_scheduler import ReduceLROnPlateau
import transforms
from data_proc import * 
from models import *
from torch.autograd import Variable

logging.basicConfig(level=logging.INFO, format='%(message)s')


lossLog = 'answer.txt'
timeStep = 5

class Train_Proc():

	@staticmethod
	def draw_roc(fpr,tpr,index):
		roc_auc = auc(fpr, tpr)
		plt.plot(fpr, tpr, 'b', label='AUC(%d) = %0.2f' %(index,roc_auc))
		plt.legend(loc='lower right')
		plt.plot([0, 1], [0, 1], 'r--')
		plt.xlim([0, 1])
		plt.ylim([0, 1])
		plt.ylabel('True Positive Rate')
		plt.xlabel('False Positive Rate')
		plt.show()
		plt.savefig('png/roc_auc_curve_%d.png' %index)

	@staticmethod
	def draw_train_roc(lossfile):

		trainLoss = []
		valLoss = []
		pos = 0
		with open(lossfile, "r") as lf:
			while True:
				line = lf.readline()
				if not line:break
				items = line.split()
				if not items: break
				trainLoss.append(float(items[2]))
				valLoss.append(float(items[3]))
				pos = int(items[0])
                
		plt.figure(figsize=(8,6), dpi=100)
		plt.plot(range(len(trainLoss)), trainLoss, label='train', color='red')
		plt.plot(range(len(valLoss)), valLoss, label='val', color='blue')
		plt.xlabel('epoch')
		plt.ylabel('loss')
		plt.show()
		plt.savefig('png/train_%d.png' %pos)

	@staticmethod
	def draw_test_roc(out_label,out_prect):
		out_label = out_label * 100.0
		out_prect =  out_prect *100.0
		gt_np = out_label.cpu().numpy()
		pred_np = out_prect.cpu().numpy()

		plt.figure(figsize=(8,6), dpi=100)
		plt.plot(range( gt_np.shape[0]), gt_np, label='train truth', color='red')
		plt.plot(range(pred_np.shape[0]), pred_np, label='predicted train', color='blue')
		plt.xlabel('Days')
		plt.ylabel('Stock price')
		plt.show()
		plt.savefig('png/train.png' )

	@staticmethod
	def print_test_roc(out_label,out_prect):
		gt_np = out_label.cpu().numpy()
		pred_np = out_prect.cpu().numpy()
		for i in range(gt_np.size):
			print('%.4f   %.4f' %(gt_np[i],pred_np[i]))
	

	@staticmethod
	def print_pred(predLabelFilePath,out_prect):
		
		Y = out_prect.cpu().numpy()
		print(Y.shape)	
		pos = 0
		with open(predLabelFilePath, "r") as lf:
			while True:
				line = lf.readline()
				if not line:break
				items = line.split()
				if not items:break
				if pos >= Y.shape[0] : break
				print('%s    %s    %s   %.4f    %s   %.4f    %s    %.4f     %s     %.4f     %s   %.4f'  % (items[0],items[1],items[16], 100.0*Y[pos,0], items[26], 100.0*Y[pos,1], 
				items[36], 100.0*Y[pos,2],items[46], 100.0*Y[pos,3],items[2], 100.0*Y[pos,4]))
				pos += 1
				
	@staticmethod
	def validate(data_loader, model, loss,device):
		model.eval()
		valLoss = 0.0
                
		with torch.no_grad():
			for i, (input, target) in enumerate(data_loader):
				input,target = input.to(device),target.to(device)
				output =  model(input)
				#tnp = target.cpu().numpy()
				#print(tnp.shape)
				#tnp = torch.FloatTensor(target.cpu().numpy()[:,4]).unsqueeze(1).to(device)
				#onp = torch.FloatTensor(output.cpu().numpy()[:,4]).unsqueeze(1).to(device)
				#print(tnp.size(),tnp)
				#print(onp.size(),onp)
				tnp = target[:,4].unsqueeze(1)
				onp = output[:,4].unsqueeze(1)
				#print(tnp.size(),tnp)
				#print(onp.size(),onp)
				valLoss += loss(onp,tnp).item()
		valLoss /= i + 1
		return valLoss

	@staticmethod
	def epoch_train(model, data_loader, optimizer, loss, device ):

		model.train()
		trainLoss = 0.0
		for batch_id, (input, target) in enumerate(data_loader):
			input, target = input.to(device), target.to(device)
			optimizer.zero_grad()
			output =  model(input)
			lossValue = loss(output, target)
			lossValue.backward()
			optimizer.step()
			trainLoss += lossValue.item()			
			tempLoss = trainLoss/(batch_id + 1)
			if (batch_id+1) % 50 == 0 :
				logging.info(' %d : training_loss = %.6f  whole loss = %.6f' % ( batch_id+1, lossValue.item(),tempLoss ))
				#break
		return tempLoss

	@staticmethod     
	def train(trainImgDir, trainLabelFilePath, valLabelFilePath, modelFilePath, batchSize, maxEpoch, checkpoint, lr, iscpu):
		
		isgpu = not iscpu and torch.cuda.is_available()
		cudnn.benchmark = True
		device = torch.device("cuda:0" if isgpu else "cpu")
		model = StockNet(timeStep).to(device)	
		print(model)


		kwargs = {'num_workers': 4, 'pin_memory': True} if isgpu else {}
		transformList = []
		transformList.append(transforms.ToTensor())
		#transformList.append(transforms.Randomblack(mean = [0.4,0.4,0.4]))
		transformList.append(transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))

		transformSequence=transforms.Compose(transformList)

		trainDataSet = StockData(labelFilePath = trainLabelFilePath)
		data_loader = DataLoader(dataset = trainDataSet, batch_size=batchSize,shuffle=True,**kwargs)

		kwargs = {'num_workers': 4, 'pin_memory': True} if isgpu else {}
		transformList = []
		transformList.append(transforms.ToTensor())
		transformList.append(transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
		transformSequence=transforms.Compose(transformList)

		valDataSet = StockData(labelFilePath = valLabelFilePath)
		valdata_loader = DataLoader(dataset = valDataSet, batch_size=batchSize,shuffle=False,**kwargs)

		#loss = torch.nn.CrossEntropyLoss()
		#loss = torch.nn.BCELoss()
		loss = torch.nn.MSELoss()
	
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
		
		for epoch_id in range(start, maxEpoch):
			oldtime = time.time()                        
			loss_train  = Train_Proc.epoch_train(model, data_loader, optimizer,loss,device)
			newtime = time.time() 
			if isgpu:torch.cuda.empty_cache()
			torch.save({'epoch':epoch_id, 'state_dict':model.state_dict(), 'optimizer':optimizer.state_dict()}, 
			'./models/'+'model-'+str(epoch_id)+'.pth.tar')			
			totaltime = newtime - oldtime
			valLoss  = Train_Proc.validate(valdata_loader,model,loss,device)
			logging.info('Epoch %d : training_loss= %.6f,time = %d,valLoss= %.6f' % (epoch_id, loss_train,totaltime,valLoss))
			with open(lossLog,'a+') as ftxt:		
				ftxt.write('%d     %d     %.6f     %.6f\n' % (epoch_id, totaltime,loss_train,valLoss))


	@staticmethod		
	def test(testImgDir, testLabelFilePath, modelFilePath, batchSize, iscpu):                 
		
		isgpu = not iscpu and torch.cuda.is_available()
		cudnn.benchmark = True
		device = torch.device("cuda:0" if isgpu else "cpu")
		
		model = StockNet(timeStep).to(device)	
		
		if not isgpu :
			modelCheckpoint = torch.load(modelFilePath,map_location=torch.device('cpu'))
		else: 
			modelCheckpoint = torch.load(modelFilePath)
		
		model.load_state_dict(modelCheckpoint['state_dict'])

		kwargs = {'num_workers': 4, 'pin_memory': True} if isgpu else {}
		transformList = []
		transformList.append(transforms.ToTensor())
		transformList.append(transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))) 
		transformSequence=transforms.Compose(transformList)

		testDataSet = StockData(labelFilePath = testLabelFilePath)
		data_loader = DataLoader(dataset = testDataSet,batch_size=batchSize,shuffle=False, **kwargs)

		#loss = torch.nn.CrossEntropyLoss()
		#loss = torch.nn.BCELoss()
		loss = torch.nn.MSELoss()
		model.eval()
		
		testLoss = 0.0
		out_label = torch.Tensor().cuda()
		out_prect = torch.Tensor().cuda()
        
		with torch.no_grad():
			for i, (input, target) in enumerate(data_loader):
				input, target = input.to(device),target.to(device)
				output =  model(input)
				testLoss += loss(output,target).item()
				out_label = torch.cat((out_label, target), 0)
				out_prect = torch.cat((out_prect, output), 0)
				#if i >= 1:break
		testLoss /= i+1
		#Train_Proc.draw_test_roc(out_label,out_prect)
		#Train_Proc.print_test_roc(out_label,out_prect)
		print(testLoss)

	'''
        print(len(testDataSet))
        gt_np = out_label.cpu().numpy()
        pred_np = out_prect.cpu().numpy()
        for i in range(4):
            u1 = gt_np[:, i]
            print(u1.size,(u1 == 1).sum())
            fpr, tpr, threshold = roc_curve(gt_np[:, i], pred_np[:, i])
            Train_Proc.draw_roc(fpr,tpr,i)

        min = math.sqrt((tpr[0] -1.0)*(tpr[0] -1.0) + fpr[0]*fpr[0] )
        hold = 0
        for i in range(1,len(fpr)):
            temp = math.sqrt((tpr[i] -1.0)*(tpr[i] -1.0) + fpr[i]*fpr[i] )
            if temp < min: 
                min = temp
                hold = i
        print('%0.4f    %0.4f     %0.4f' %(tpr[hold],fpr[hold],threshold[hold]))

        label1 = out_label.cpu().view(-1).numpy().tolist()
        label2 = out_prect.cpu().view(-1).numpy().tolist()
        c = confusion_matrix(label1,label2)
        print(c)
	'''
        

	@staticmethod
	def predict(predImgDir, predLabelFilePath,modelFilePath, batchSize, iscpu):   

		isgpu = not iscpu and torch.cuda.is_available()
		cudnn.benchmark = True
		device = torch.device("cuda:0" if isgpu else "cpu")
		
		model = StockNet(timeStep).to(device)	
		
		if not isgpu :
			modelCheckpoint = torch.load(modelFilePath,map_location=torch.device('cpu'))
		else: 
			modelCheckpoint = torch.load(modelFilePath)

		

		model.load_state_dict(modelCheckpoint['state_dict'])

		kwargs = {'num_workers': 4, 'pin_memory': True} if isgpu else {}
		transformList = []
		transformList.append(transforms.ToTensor())
		transformList.append(transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
		transformSequence=transforms.Compose(transformList)

		predDataSet = StockData(labelFilePath = predLabelFilePath)
		data_loader = DataLoader(dataset = predDataSet,batch_size=batchSize,shuffle=False,**kwargs)

		model.eval()

		out_prect = torch.FloatTensor().to(device)
		
		with torch.no_grad():
			for i, (input, target) in enumerate(data_loader):
				input, target = input.to(device), target.to(device)
				output =  model(input)
				out_prect = torch.cat((out_prect, output), 0)
				if i>=10: break
		Train_Proc.print_pred(predLabelFilePath,out_prect)
		'''
		Y = out_prect.cpu().numpy()
		pos = 0
		with open(predLabelFilePath, "r") as lf:
			while True:
				line = lf.readline()
				if not line:break
				items = line.split()
				if not items:break
				print('%s   %s   %.4f      %d'  % (items[0],items[41],Y[pos,0], abs(int(float(items[41]) - 100*Y[pos,0]))))
				pos += 1
		'''
		
	@staticmethod
	def create_onnx(model_path,iscpu):		

		protoFile = 'stock.proto'
		inputChn = 3
		inputWidth = 128
		inputHeight = 128

		isgpu = not iscpu and torch.cuda.is_available()
		model = StockNet(timeStep)
		
		if not isgp:
			modelCheckpoint = torch.load(model_path,map_location='cpu')
			dummy_input = torch.randn(1, inputChn, inputHeight,inputWidth)
		else:
			model = model.cuda()
			dummy_input = torch.randn(1, inputChn, inputHeight,inputWidth).cuda()
			modelCheckpoint = torch.load(model_path)
		model.load_state_dict(modelCheckpoint['state_dict'])
		input_names = [ "input" ] 
		output_names = [ "output" ]
		torch.onnx.export(model, dummy_input, protoFile, verbose=True, input_names=input_names, output_names=output_names)

		
