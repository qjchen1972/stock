# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import division

from collections import defaultdict, deque
import random
import numpy as np
from game import *
from mcts import *
from net import *
from scene import *


def policy_value_fn(board):

        #action_probs = np.ones(len(board.availables))/len(board.availables)
        action_probs = np.array([0.5,0.5,0.5,0.5,0.5,1.5,1.5,1.5,2.,2.])
        return zip(board.availables, action_probs), 0

class Train(object):

	def __init__(self, init_model=None):
	
		self.time_step = 10
		self.env = Env(self.time_step)
		self.scene = Scene()
		self.game = Game(self.env,self.scene)

		self.buffer_size = 1024
		self.data_buffer = deque(maxlen=self.buffer_size)

		# training params
		self.learn_rate = 2e-3
		self.lr_multiplier = 1.0  # adaptively adjust the learning rate based on KL
		self.batch_size = 256  # mini-batch size for training
		self.play_batch_size = 1
		self.epochs = 10  # num of train_steps for each update
		self.kl_targ = 0.02
		self.check_freq = 30
		self.game_batch_num = 1500
		self.best_win_ratio = 0.0


		if init_model:
			print(init_model)	
			self.policy_value_net = PolicyValueNet(self.time_step,model_file=init_model)
			#self.policy_value_net1 = PolicyValueNet(self.time_step)
		else:
			self.policy_value_net = PolicyValueNet(self.time_step)

		
		self.temp = 1.0  # the temperature param
		self.n_playout = 100  # num of simulations for each move
		self.c_puct = 5
		
		self.mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn,
                                      c_puct=self.c_puct,
                                      n_playout=self.n_playout,
                                      is_selfplay=1)


	
	def human_play_game(self):
		print('profit:',self.game.human_play_game())		


	def collect_selfplay_data(self, n_games=1):

		self.episode_len = 0
		for i in range(n_games):
			profit,play_data = self.game.self_play_game(self.mcts_player,temp=self.temp)
			
			play_data = list(play_data)[:]
			#print('self game', profit)
			self.episode_len += len(play_data)
			self.data_buffer.extend(play_data)



	def policy_update(self):

		mini_batch = random.sample(self.data_buffer, self.batch_size)
		state_batch = [data[0] for data in mini_batch]
		mcts_probs_batch = [data[1] for data in mini_batch]
		winner_batch = [data[2] for data in mini_batch]

		old_probs, old_v = self.policy_value_net.policy_value(state_batch)
		for i in range(self.epochs):
			loss, entropy = self.policy_value_net.train_step(
                    							state_batch,
                    							mcts_probs_batch,
                    							winner_batch,
                    							self.learn_rate*self.lr_multiplier)

			new_probs, new_v = self.policy_value_net.policy_value(state_batch)
			kl = np.mean(np.sum(old_probs * (np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)),axis=1))
			if kl > self.kl_targ * 4:  # early stopping if D_KL diverges badly
				break

		# adaptively adjust the learning rate
		if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1: self.lr_multiplier /= 1.5
		elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:self.lr_multiplier *= 1.5


		explained_var_old = (1 -
                             np.var(np.array(winner_batch) - old_v.flatten()) /
                             (np.var(np.array(winner_batch) + 1e-10)))
		explained_var_new = (1 -
                             np.var(np.array(winner_batch) - new_v.flatten()) /
                             (np.var(np.array(winner_batch)+1e-10)))

		print(("kl:{:.5f},"
			"lr_multiplier:{:.3f},"
			"loss:{},"
			"entropy:{},"
			"explained_var_old:{:.3f},"
			"explained_var_new:{:.3f}"
			).format(kl,
				self.lr_multiplier,
				loss,
				entropy,
				explained_var_old,
				explained_var_new))

		return loss, entropy


	def run(self):

		try:
			for i in range(self.game_batch_num):
				self.collect_selfplay_data(self.play_batch_size)
				print("batch i:{}, episode_len:{},buf_size:{}".format(i+1, self.episode_len,len(self.data_buffer)))
				if len(self.data_buffer) > self.batch_size:
					loss, entropy = self.policy_update()
				# check the performance of the current model,
				# and save the model params
				if (i+1) % self.check_freq == 0:
					print("current self-play batch: {}".format(i+1))
					win_ratio = self.policy_evaluate()
					self.policy_value_net.save_model('./current_policy.model')
					if win_ratio > self.best_win_ratio:
						print("New best policy!!!!!!!!")
						self.best_win_ratio = win_ratio
						# update the best_policy
						self.policy_value_net.save_model('./best_policy.model')
		except KeyboardInterrupt:
			print('\n\rquit')





	def policy_evaluate(self, n_games=10):

		current_mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn,
                                         	c_puct=self.c_puct,
                                         	n_playout=self.n_playout)
		

		win_cnt = 0.
		total = 0

		for i in range(n_games):
			profit = self.game.AI_play_game(current_mcts_player,temp=self.temp)
			
			total += profit
			if profit >= 15:win_cnt += 1


		win_ratio = win_cnt / n_games
		print('test game ratio is %.4f, total is  %.4f' %(win_ratio,total))
		return win_ratio
			



