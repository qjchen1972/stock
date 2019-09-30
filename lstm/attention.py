import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F


class AttnEncoder(nn.Module):

	def __init__(self, input_size, hidden_size, time_step):
		super(AttnEncoder, self).__init__()
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.T = time_step
		self.dpvecsize =10
        

		self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=1)
		#self.attn1 = nn.Linear(in_features=2 * hidden_size, out_features=input_size)
		#self.attn2 = nn.Linear(in_features=self.dpvecsize, out_features=input_size)
		self.attn1 = nn.Sequential(nn.Linear(in_features=2 * hidden_size, out_features=input_size),nn.ReLU(inplace=True))
		#self.attn2 = nn.Sequential(nn.Linear(in_features=input_size, out_features=input_size),nn.BatchNorm1d(input_size,affine=True),nn.ReLU(inplace=True))
		self.attn2 = nn.Sequential(nn.Linear(in_features=self.dpvecsize, out_features=input_size),nn.ReLU(inplace=True))
		#self.attn4 = nn.Sequential(nn.Linear(in_features=input_size, out_features=input_size),nn.BatchNorm1d(input_size,affine=True),nn.ReLU(inplace=True))

		#self.tanh = nn.Tanh()
		self.attn3 =  nn.Sequential(nn.Linear(in_features=2*input_size, out_features= input_size),nn.ReLU(inplace=True))
		#self.attn6 = nn.Sequential(nn.Linear(in_features=input_size, out_features=input_size),nn.BatchNorm1d(input_size,affine=True),nn.ReLU(inplace=True))

		#self.attn4 = nn.Linear(in_features=2*input_size, out_features= input_size)
		#self.attn5 = nn.Linear(in_features=3*input_size, out_features= 2 * hidden_size)


	def forward(self, stockvec):
		batch_size = stockvec.size(0)
		#print(stockvec.size())
		
		# batch_size * time_step * hidden_size
		code = self.init_variable(batch_size, self.T, self.hidden_size)
		# initialize hidden state
		h = self.init_variable(1, batch_size, self.hidden_size)
		# initialize cell state
		s = self.init_variable(1, batch_size, self.hidden_size)
		for t in range(self.T):
			#batch_size * input_size * (2 * hidden_size + time_step)
			x = torch.cat((h,s),2).squeeze(0)
			z1 = self.attn1(x)
			#z1 = self.attn2(z1)
			#z1 = self.tanh(z1)
			
			#z1 = F.softmax(z1, dim=1)


			vec = stockvec[:,t,:]
			#print(vec.size())
			z2 = self.attn2(vec)
			#z2 = self.attn4(z2)
			#z2 = self.tanh(z2)
			#z2 = F.softmax(z2, dim=1)

			x = torch.cat((z1,z2),1)
			x = self.attn3(x)
			#x = self.attn6(x)
			#x = self.tanh(x)
			_, states = self.lstm(x.unsqueeze(0), (h, s))
			h = states[0]
			s = states[1]
	
			#h = F.glu(h).unsqueeze(0)
			#s = self.attn5(x)
			#s = F.glu(s).unsqueeze(0)

			code[:, t, :] = h

		return code

	def init_variable(self, *args):
		zero_tensor = torch.zeros(args)
		if torch.cuda.is_available():
			zero_tensor = zero_tensor.cuda()
		return Variable(zero_tensor)

	def embedding_hidden(self, x):
		return x.repeat(self.input_size, 1, 1).permute(1, 0, 2)


class AttnDecoder(nn.Module):

	def __init__(self, code_hidden_size, hidden_size, time_step):
		super(AttnDecoder, self).__init__()
		self.code_hidden_size = code_hidden_size
		self.hidden_size = hidden_size
		self.T = time_step

		self.attn1 = nn.Sequential(nn.Linear(in_features=2 * hidden_size, out_features=code_hidden_size),nn.ReLU(inplace=True))
		#self.attn2 = nn.Sequential(nn.Linear(in_features=code_hidden_size, out_features=code_hidden_size),nn.BatchNorm1d(code_hidden_size,affine=True),nn.ReLU(inplace=True))
		self.attn2 = nn.Sequential(nn.Linear(in_features=hidden_size, out_features=code_hidden_size),nn.ReLU(inplace=True))
		#self.attn4 = nn.Sequential(nn.Linear(in_features=code_hidden_size, out_features=code_hidden_size),nn.BatchNorm1d(code_hidden_size,affine=True),nn.ReLU(inplace=True)) 
		#self.tanh = nn.Tanh()
		#self.attn3 = nn.Sequential(nn.Linear(in_features=1, out_features=code_hidden_size),nn.ReLU(inplace=True))
		#self.attn6 = nn.Sequential(nn.Linear(in_features=code_hidden_size, out_features=code_hidden_size),nn.BatchNorm1d(code_hidden_size,affine=True),nn.ReLU(inplace=True))
		#self.attn4 = nn.Linear(in_features=3*code_hidden_size, out_features= 2 * hidden_size)
		#self.attn5 = nn.Linear(in_features=3*code_hidden_size, out_features= 2 * hidden_size)

		#self.attn1 = nn.Linear(in_features=2 * hidden_size, out_features=code_hidden_size)
		#self.attn2 = nn.Linear(in_features=hidden_size, out_features=code_hidden_size)
		#self.attn3 = nn.Linear(in_features=1, out_features=code_hidden_size)
		self.lstm = nn.LSTM(input_size=code_hidden_size, hidden_size=self.hidden_size)
		self.tilde = nn.Linear(in_features=2*self.code_hidden_size, out_features=code_hidden_size)
		self.fc1 = nn.Linear(in_features=2*code_hidden_size, out_features=1)
		#self.fc2 = nn.Linear(in_features=hidden_size, out_features=1)

	def forward(self, h):
		batch_size = h.size(0)
		# initialize hidden state
		d = self.init_variable(1, batch_size, self.hidden_size)
                # initialize cell state
		s = self.init_variable(1, batch_size, self.hidden_size)
		y_res = torch.Tensor().cuda()
		for t in range(self.T):
			x = torch.cat((d,s),2).squeeze(0)
			z1 = self.attn1(x)
			#z1 = self.attn2(z1)

			#z1 = self.tanh(z1)
			#z1 = F.softmax(z1, dim=1)

			vec = h[:,t,:]
			z2= self.attn2(vec)
			#z2 = self.attn4(z2)

			#z2 = self.tanh(z2)
			#z2 = F.softmax(z2, dim=1)

			#vec = y_seq[:,t,3].unsqueeze(1)
			#vec = y_seq[:,t,3]
			#print(vec.size(),vec)
			#exit()
			#z3= self.attn3(vec)
			#z3 = self.attn6(z3)

			#z3 = self.tanh(z3)
			#z3 = F.softmax(z3, dim=1)
			x = torch.cat((z1,z2),1)
			y_tilde = self.tilde(x)
			_, states = self.lstm(y_tilde.unsqueeze(0), (d, s))
			d = states[0]
			s = states[1]

			y_res = torch.cat((y_res, self.fc1(x)), 1)
			#y_res = self.fc1(x)

			#if t < self.T - 1:
				#x = torch.cat((z1,z2,z3),1)
				#d = self.attn4(x)
				#d = F.glu(d).unsqueeze(0)
				#s = self.attn5(x)
				#s = F.glu(s).unsqueeze(0)
				#y_tilde = self.tilde(x)
				#y_tilde = self.tanh(y_tilde)
				#_, states = self.lstm(y_tilde.unsqueeze(0), (d, s))
				#d = states[0]
				#s = states[1]
		#batch_size * 1
		#x = torch.cat((z1,z2),1)
		#y_res = self.fc1(x)
		#y_res = self.tanh(y_res)
		return y_res

	def init_variable(self, *args):
		zero_tensor = torch.zeros(args)
		if torch.cuda.is_available():
			zero_tensor = zero_tensor.cuda()
		return Variable(zero_tensor)
	
	def embedding_hidden(self, x):
		return x.repeat(self.T, 1, 1).permute(1, 0, 2)


