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
import scikitplot as skplt
import time
import math
import torch.onnx
from torch.optim.lr_scheduler import ReduceLROnPlateau
import transforms
from data_proc import * 
from models import *
from torch.autograd import Variable

logging.basicConfig(level=logging.INFO, format='%(message)s')


loss_log = 'answer.txt'
stockClass = 1

class Train_Proc():

    @staticmethod
    def plt_roc(test_y, probas_y, plot_micro=False, plot_macro=False):
        assert isinstance(test_y, list) and isinstance(probas_y, list), 'the type of input must be list'
        skplt.metrics.plot_roc(test_y, probas_y, plot_micro=plot_micro, plot_macro=plot_macro)
        plt.savefig('roc_auc_curve.png')
        plt.close()

    @staticmethod
    def draw_roc(fpr,tpr,i):
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, 'b', label='AUC(%d) = %0.2f' %(i,roc_auc))
        plt.legend(loc='lower right')
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.show()
        plt.savefig('png/roc_auc_curve_%d.png' %i)


    @staticmethod
    def validate(model, loss,isgpu):   		
        valImgDir = 'data/train/trainimg'
	valLabelFilePath = 'data/train/val.txt'

	device = torch.device("cuda:0" if isgpu else "cpu")

	kwargs = {'num_workers': 4, 'pin_memory': True} if isgpu else {}
	transformList = []
	transformList.append(transforms.ToTensor())
        transformList.append(transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))) 
	transformSequence=transforms.Compose(transformList)
	valDataSet = StockData(imgDir = valImgDir,labelFilePath = valLabelFilePath, transform = transformSequence,istrain=False)
	data_loader = DataLoader(dataset = valDataSet,batch_size=32,shuffle=False,**kwargs)

	model.eval()
	valLoss = 0.0
	#correct = 0.0
                
	with torch.no_grad():
	    for i, (input, feel,target) in enumerate(data_loader):
	        input, feel,target = input.to(device), feel.to(device),target.to(device)
		output =  model(input,feel)
		valLoss += loss(output,target).item()
		#pred = output.argmax(dim=1, keepdim=True)
                #tt = torch.unsqueeze(target, 1)
                #print(torch.cat((pred,tt),1))
		#correct += pred.eq(target.view_as(pred)).sum().item()
	#valLoss /= i + 1
        #correct = 1.0 * correct
	#correct /= len(valDataSet)
	return valLoss

    @staticmethod
    def epoch_train(model, data_loader, optimizer, loss, device ):

        model.train()
	trainLoss = 0.0
        #correct =0.0
	for batch_id, (input, feel,target) in enumerate(data_loader):

	    input, feel,target = input.to(device), feel.to(device),target.to(device)
	    optimizer.zero_grad()
	    output =  model(input,feel)
            #pred = output.argmax(dim=1, keepdim=True)
            #correct += pred.eq(target.view_as(pred)).sum().item()
            #print(output.size(),target.size())
	    lossValue = loss(output, target)
	    lossValue.backward()
	    optimizer.step()
	    trainLoss += lossValue.item()			
	    tempLoss = trainLoss/(batch_id + 1)
	    if batch_id % 50 == 0 :
	        logging.info(' %d : training_loss = %.6f  whole loss = %.6f' % ( batch_id, lossValue.item(),tempLoss ))
                #break
        #correct /= len(data_loader.dataset)        
	return tempLoss

    @staticmethod     
    def train(trainImgDir, trainLabelFilePath, modelFilePath, batchSize, maxEpoch, checkpoint, lr, iscpu):
		
        isgpu = not iscpu and torch.cuda.is_available()

	cudnn.benchmark = True

	#torch.manual_seed(seed)
	device = torch.device("cuda:0" if isgpu else "cpu")
		
	model = StockNet(stockClass).to(device)	
	print(model)


	kwargs = {'num_workers': 4, 'pin_memory': True} if isgpu else {}
	transformList = []
	transformList.append(transforms.ToTensor())
        #transformList.append(transforms.Randomblack(mean = [0.4,0.4,0.4]))
        transformList.append(transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))

	transformSequence=transforms.Compose(transformList)

	trainDataSet = StockData(imgDir = trainImgDir, labelFilePath = trainLabelFilePath,transform=transformSequence,istrain=True)
	data_loader = DataLoader(dataset = trainDataSet, batch_size=batchSize,shuffle=True,**kwargs)	
		
	#loss = torch.nn.CrossEntropyLoss()
        #loss = torch.nn.BCELoss()
        loss = torch.nn.MSELoss()
		
	#optimizer = optim.SGD(model.parameters(), lr=lr,momentum=0.9)
	optimizer = optim.Adam(model.parameters(), lr=lr,weight_decay=0,amsgrad=True)
			
	if checkpoint == True:
	    modelCheckpoint = torch.load(modelFilePath)
	    model.load_state_dict(modelCheckpoint['state_dict'])
	    #optimizer.load_state_dict(modelCheckpoint['optimizer'])
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

	    #if epoch_id % 10 == 0 :
	    valLoss  = Train_Proc.validate(model,loss,isgpu)
	    logging.info('Epoch %d : training_loss= %.6f,time = %d,valLoss= %.6f' % (epoch_id, loss_train,totaltime,valLoss))
	    with open(loss_log,'a+') as ftxt:		
	        ftxt.write('\nEpoch %d : training_loss= %.6f,time = %d,valLoss= %.6f\n' % (epoch_id, loss_train, totaltime,valLoss))

		
    @staticmethod		
    def test(testImgDir, testLabelFilePath, modelFilePath, batchSize, iscpu):                 
		
        isgpu = not iscpu and torch.cuda.is_available()
	cudnn.benchmark = True
	device = torch.device("cuda:0" if isgpu else "cpu")
		
	model = StockNet(stockClass).to(device)	
		
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

	testDataSet = StockData(imgDir = testImgDir, labelFilePath = testLabelFilePath, transform=transformSequence,istrain= False)
	data_loader = DataLoader(dataset = testDataSet,batch_size=batchSize,shuffle=False, **kwargs)

	#loss = torch.nn.CrossEntropyLoss()
        #loss = torch.nn.BCELoss()
        loss = torch.nn.MSELoss()

	model.eval()
		
	testLoss = 0.0
	#correct = 0.0
        out_label = torch.Tensor().cuda()
        out_prect = torch.Tensor().cuda()
        ##pred_y = list()
        #test_y = list()
        #probas_y = list()
        
	with torch.no_grad():
	    for i, (input, feel,target) in enumerate(data_loader):
	        input, feel,target = input.to(device),feel.to(device), target.to(device)
		output =  model(input,feel)
		testLoss += loss(output,target).item()
                out_label = torch.cat((out_label, target), 0)
		#pred = output.argmax(dim=1, keepdim=True)
                out_prect = torch.cat((out_prect, output), 0)
                #print(out_prect.size())
                #print(output.cpu().numpy().shape)
                #probas_y.extend(output.cpu().numpy().tolist())
                #pred_y.extend(output.data.cpu().max(1, keepdim=True)[1].numpy().flatten().tolist())
                #print(target.cpu().numpy().shape)
                #print(target.cpu().numpy().flatten().shape)
                
                #test_y.extend(target.cpu().numpy().astype(int).tolist())
                #test_y.extend(target.cpu().numpy().tolist())


		#correct += pred.eq(target.view_as(pred)).sum().item()
	testLoss /= i+1
	#correct /= len(testDataSet)

        print(testLoss)


        #label1 = out_prect.cpu().numpy()
        #label2 = out_label.cpu().numpy().flatten()
        #print(test_y)
        #print('ok')
        #print(probas_y)
        #print(label1.shape,label2.shape)

        #probas_y.extend(out_prect.cpu().numpy().flatten().tolist())
        #test_y.extend(out_label.cpu().numpy().flatten().tolist())

	#print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        #testLoss, correct, len(testDataSet),
        #100. * correct / len(testDataSet)))
        #label =[0,1,2]
        #label1 = out_label.cpu().view(-1).numpy().tolist()
        #label2 = out_prect.cpu().view(-1).numpy().tolist()
        #print(test_y.shape)
        #print(probas_y.shape)
        #p =  [int(i) for i in test_y]
        '''
        print(len(testDataSet))
        gt_np = out_label.cpu().numpy()
        pred_np = out_prect.cpu().numpy()
        for i in range(4):
            u1 = gt_np[:, i]
            print(u1.size,(u1 == 1).sum())
            fpr, tpr, threshold = roc_curve(gt_np[:, i], pred_np[:, i])
            Train_Proc.draw_roc(fpr,tpr,i)
            #l = len(fpr)
            #print('len :',l)
            #for o in range(len(fpr)):
                #print('%d    %d     %0.4f' %(tpr[o]*m1,fpr[o]*m2,threshold[o]))

            #print(np.hstack((tpr.reshape(-1,1),np.hstack((fpr.reshape(-1,1),threshold.reshape(-1,1))))))
            #break
        '''

        '''
        min = math.sqrt((tpr[0] -1.0)*(tpr[0] -1.0) + fpr[0]*fpr[0] )
        hold = 0
        for i in range(1,len(fpr)):
            temp = math.sqrt((tpr[i] -1.0)*(tpr[i] -1.0) + fpr[i]*fpr[i] )
            if temp < min: 
                min = temp
                hold = i

        print('%0.4f    %0.4f     %0.4f' %(tpr[hold],fpr[hold],threshold[hold]))
        

        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, 'b', label='AUC= %0.2f' %roc_auc)
        plt.legend(loc='lower right')
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.show()
        '''
        

        #c = confusion_matrix(label1,label2)
        #print(c)
        #target_names = ['class 0', 'class 1', 'class 2']
        #print(classification_report(label1, label2, target_names=target_names))
        

    @staticmethod
    def predict(predImgDir, predLabelFilePath,modelFilePath, batchSize, iscpu):   

        isgpu = not iscpu and torch.cuda.is_available()
	cudnn.benchmark = True
	device = torch.device("cuda:0" if isgpu else "cpu")
		
	model = StockNet(stockClass).to(device)	
		
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

	predDataSet = StockData(imgDir = predImgDir, labelFilePath = predLabelFilePath, transform=transformSequence,istrain=False)
	data_loader = DataLoader(dataset = predDataSet,batch_size=batchSize,shuffle=False,**kwargs)

	model.eval()

	out_prect = torch.FloatTensor().to(device)
		
	with torch.no_grad():
	    for i, (input, feel,target) in enumerate(data_loader):
	        input, feel,target = input.to(device), feel.to(device),target.to(device)
		output =  model(input,feel)
		#pred = output.argmax(dim=1, keepdim=True)
		out_prect = torch.cat((out_prect, output), 0)

	#predicted = torch.max(out_prect, 1)[1]
	#Y = np.reshape(predicted.cpu(), -1)
        Y = out_prect.cpu().numpy()
        print(Y.size,Y.shape)

        pos = 0
        with open(predLabelFilePath, "r") as lf:
            while True:
                line = lf.readline()
                if not line:break
                items = line.split()
                if not items: break
                print('%s   %s    %s    %s     %s     %s     %.4f      %d'  % (items[0],items[1],items[2],items[3],items[4],items[5],Y[pos,0], abs(int(float(items[5]) - 100*Y[pos,0]))))
                pos += 1

	#np.savetxt('pred.csv', Y, delimiter=",", fmt='%5s')
		

    @staticmethod
    def create_onnx(model_path,iscpu):		

        protoFile = 'stock.proto'
	inputChn = 1
	inputWidth = 128
	inputHeight = 128

	isgpu = not iscpu and torch.cuda.is_available()
	model = StockNet(stockClass)
		
	if not isgpu:
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
		

		
