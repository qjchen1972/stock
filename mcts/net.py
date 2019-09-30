# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np


def set_learning_rate(optimizer, lr):

	for param_group in optimizer.param_groups:
		param_group['lr'] = lr


class Net(nn.Module):

	def __init__(self, time_step):
		super(Net, self).__init__()

		self.time_step = time_step

		self.conv1 = nn.Conv2d(4, 32, kernel_size=3, padding=1)
		self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
		self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
		# action policy layers
		self.act_conv1 = nn.Conv2d(128, 4, kernel_size=1)
		self.act_fc1 = nn.Linear(4*time_step*time_step,time_step)
		# state value layers
		self.val_conv1 = nn.Conv2d(128, 2, kernel_size=1)
		self.val_fc1 = nn.Linear(2*time_step*time_step, 64)
		self.val_fc2 = nn.Linear(64, 1)

	def forward(self, state_input):
		# common layers
		x = F.relu6(self.conv1(state_input))
		x = F.relu6(self.conv2(x))
		x = F.relu6(self.conv3(x))
		# action policy layers
		x_act = F.relu6(self.act_conv1(x))
		#print(x_act.size(1))
		x_act = x_act.view(-1, x_act.size(1)*x_act.size(2)*x_act.size(3))
		#print(self.act_fc1(x_act).size())
		x_act = F.log_softmax(self.act_fc1(x_act),dim = 1)
		# state value layers
		x_val = F.relu6(self.val_conv1(x))
		x_val = x_val.view(-1, x_val.size(1)*x_val.size(2)*x_val.size(3))
		x_val = F.relu6(self.val_fc1(x_val))
		x_val = F.tanh(self.val_fc2(x_val))
		return x_act, x_val




class PolicyValueNet():

	def __init__(self, time_step,model_file=None, use_gpu=True):

		self.use_gpu = use_gpu
		self.time_step = time_step
		self.l2_const = 1e-4
		self.device = torch.device("cuda:0" if use_gpu else "cpu")
		# the policy value net module
			
		self.policy_value_net = Net(time_step).to(self.device)

		self.optimizer = optim.Adam(self.policy_value_net.parameters(),weight_decay=self.l2_const,amsgrad=True)

		if model_file:
			net_params = torch.load(model_file)
			self.policy_value_net.load_state_dict(net_params['state_dict'])


	def policy_value(self, state_batch):
        
		log_act_probs, value = self.policy_value_net(torch.FloatTensor(state_batch).to(self.device))
		act_probs = np.exp(log_act_probs.detach().cpu().numpy())
		return act_probs, value.detach().cpu().numpy()
						

	def policy_value_fn(self, board):

		legal_positions = board.availables
		current_state = np.ascontiguousarray(board.current_state().reshape(-1, 4, self.time_step, self.time_step))
		log_act_probs, value = self.policy_value_net(torch.from_numpy(current_state).float().to(self.device))
		act_probs = np.exp(log_act_probs.cpu().detach().numpy().flatten())
		
		act_probs = zip(legal_positions, act_probs[legal_positions])
		value = value.cpu().detach()[0][0]
		return act_probs, value


	def train_step(self, state_batch, mcts_probs, winner_batch, lr):

		state_batch = torch.FloatTensor(state_batch).to(self.device)
		mcts_probs = torch.FloatTensor(mcts_probs).to(self.device)
		winner_batch = torch.FloatTensor(winner_batch).to(self.device)
		
		# zero the parameter gradients
		self.optimizer.zero_grad()
		# set learning rate
		set_learning_rate(self.optimizer, lr)

		# forward
		log_act_probs, value = self.policy_value_net(state_batch)
		# define the loss = (z - v)^2 - pi^T * log(p) + c||theta||^2
		# Note: the L2 penalty is incorporated in optimizer
		value_loss = F.mse_loss(value.view(-1), winner_batch)
		policy_loss = -torch.mean(torch.sum(mcts_probs*log_act_probs, 1))
		loss = value_loss + policy_loss
		# backward and optimize
		loss.backward()
		self.optimizer.step()
		# calc policy entropy, for monitoring only
		entropy = -torch.mean(torch.sum(torch.exp(log_act_probs) * log_act_probs, 1))

		return loss.item(), entropy.item()


	def save_model(self, model_file):
        
		net_params = self.policy_value_net.state_dict()
		torch.save({'state_dict':net_params},model_file)


