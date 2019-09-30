# -*- coding: utf-8 -*-

from __future__ import print_function
import numpy as np
import random

class Env(object):

	def __init__(self,time_step = 10):
		self.time_step = time_step
		self.availables = list(range(time_step))
		self.index = [i for i in range(self.time_step)]

	def init_env(self,scene,prob):
		self.bet_states = []
		self.game_scene = []
		self.prob_scene = []
		self.game_scene.append(scene)
		self.prob_scene.append(prob)
		self.last_move = -1
		self.turn = 1


	def update_scene(self,scene,prob = None):
		self.game_scene.append(scene+self.turn*self.time_step)
		if prob is None:
			prob = _get_prob(scene,self.prob_scene[-1]) 
		self.prob_scene.append(prob)
		self.turn += 1
	

	def _get_prob(self,scene,last_prob):

		prob = last_prob

		if scene > 5:

			leave = scene -5

			prob[3] += leave
			if prob[3] > 9:
				leave = prob[3] - 9
				prob[3]= 9
			else: return prob

			prob[2] += leave
			if prob[2] > 9:
				leave = prob[2] - 9
				prob[2]= 9
			else: return prob

			prob[1] += leave
			if prob[1] > 9:
				leave = prob[1] - 9
				prob[1]= 9
			else: return prob

			prob[0] += leave
			if prob[0] > 9:
				prob[0] =  9
			return prob

		elif scene < 4:
			leave = 4 - scene

			prob[6] += leave
			if prob[6] > 9:
				leave = prob[6] - 9
				prob[6]= 9
			else: return prob

			prob[7] += leave
			if prob[7] > 9:
				leave = prob[7] - 9
				prob[7]= 9
			else: return prob

			prob[8] += leave
			if prob[8] > 9:
				leave = prob[8] - 9
				prob[8]= 9
			else: return prob

			prob[9] += leave
			if prob[9] > 9:
				prob[9] =  9
			return prob
		else:
			return prob
			

	def _softmax(self,x):

		probs = np.exp(x - np.max(x))
		probs /= np.sum(probs)
		return probs

	def _get_profit(self,move,prob):
		
		act_prob = _softmax(np.array(prob))
		profit = np.sum((move - self.index)*act_prob)
		return profit
		
	def current_state(self):
		
		square_state = np.zeros((4, self.time_step, self.time_step))
		if self.bet_states:
			bet_states = np.array(self.bet_states)
			square_state[0][self.index[0:len(bet_states)], bet_states % self.time_step] = 1.0

		scene = np.array(self.game_scene)	
		square_state[1][self.index[0:len(scene)],scene % self.time_step] = 1.0

		if self.last_move >= 0:
			square_state[2][self.last_move// self.time_step,self.last_move % self.time_step] = 1.0
	
		square_state[3][self.index,self.prob_scene[-1]] = 1.0
		return square_state


	def do_move(self,move):
		move += self.turn*self.time_step
		self.bet_states.append(move)
		self.last_move = move
		self.turn += 1

	def game_profit(self):
	
		if not self.bet_states :return 0.0
		profit = 0
		moves = np.array(self.bet_states) % self.time_step
		for i in range(len(moves)):
			profit += _get_profit(move[i],self.prob_scene[i])
		return profit

	def game_end(self):		
 
		if len(self.bet_states) == self.time_step:return True
		else:return False


	def game_show(self):		

		show_state = np.zeros((2*self.time_step, self.time_step))
		scene = np.array(self.game_scene)
		show_state[scene // self.time_step , scene % self.time_step] = 1
		if self.bet_states:
			bet_states = np.array(self.bet_states)
			show_state[bet_states// self.time_step, bet_states% self.time_step] = 1
	
		return show_state




class Game(object):

	def __init__(self,env,scene):
		self.env = env
		self.scene = scene
		#print(self.scene)

	#def set_scene(self,scene):
		#p,q = zip(*scene)
		#self.scene = scene
		#print(self.scene)

	def _get_human_action(self):

		try:
			location = input("Your move: ")
			if isinstance(location, str):  # for python3
				move = int(location, 10)
		except Exception as e:
			move = -1
		return move


		
	def human_play_game(self):

		scene = self.scene.get_item()
		print(scene)
		print(scene[0][0],scene[0][1])
		self.env.init_env(scene)
		turn = 0
		while True:
			print('\n')
			print(self.env.game_show())
			move =self. _get_human_action()
			turn += 1
			print(scene[int(turn)][0],scene[int(turn)][1])
			self.env.do_move(move)
			#print(self.env.current_state())
			profit = self.env.game_profit()
			print('profit:',self.env.game_profit())
			if self.env.game_end():
				return profit

			
	
	def self_play_game(self,player,temp=1e-3):

		scene = self.scene.get_item()
		self.env.init_env(scene)
		states, mcts_probs = [], []
		while True:
					
			move, move_probs = player.get_action(self.env,temp = temp,return_prob=1)
			#print(self.env.current_state())
			#print(self.env.game_show())	
			states.append(self.env.current_state())
			mcts_probs.append(move_probs)
			self.env.do_move(move)
			if self.env.game_end():
				profit = self.env.game_profit()
				#print('profit:',profit)
				winners_z = np.zeros(len(mcts_probs))
				if profit >0:win=1.0
				elif profit == 0: win = 0.0
				else:win=-1.0
				winners_z[:] = win

				return profit,zip(states, mcts_probs, winners_z)



	def AI_play_game(self,player,temp=1e-3,rand = 0):

		scene = self.scene.get_item()
		self.env.init_env(scene)
		print('\n',self.env.game_scene)
		ans =[]
		pro = []
		s1 =[i%10 for i in self.env.game_scene[0:10]]
		s2 = [i%10 for i in  self.env.game_scene[1:11]]
		while True:

			#print('\n')
			#print(self.env.game_show())
			if rand == 0:move = player.get_action(self.env,temp = temp)
			else:move = random.randint(0,9)
			self.env.do_move(move)
			
			ans.append(move)
			profit = self.env.game_profit()
			pro.append(profit)
			if self.env.game_end():
				#profit = self.env.game_profit()
				#ans.append(profit)
				print('\n',list(zip(ans,s1,s2,pro)))
				return profit


	def reset(self):
		self.scene.reset()



	def get_distrib(move,last_dis):

		new_dis = last_dis
		if move > 5:

			leave = move -5
			new_dis[3] += leave
			if new_dis[3] > 9:
				leave = new_dis[3] - 9
				new_dis[3]= 9
			else: return new_dis
			new_dis[2] += leave

			if new_dis[2] > 9:
				leave = new_dis[2] - 9
				new_dis[2]= 9
			else: return new_dis

			new_dis[1] += leave
			if new_dis[1] > 9:
				leave = new_dis[1] - 9
				new_dis[1]= 9
			else: return new_dis

			new_dis[0] += leave
			if new_dis[0] > 9:
				new_dis[0] =  9
			return new_dis
		elif move < 4:
			leave = 4 - move
			new_dis[6] += leave
			if new_dis[6] > 9:
				leave = new_dis[6] - 9
				new_dis[6]= 9
			else: return new_dis
			new_dis[7] += leave

			if new_dis[7] > 9:
				leave = new_dis[7] - 9
				new_dis[7]= 9
			else: return new_dis

			new_dis[8] += leave
			if new_dis[8] > 9:
				leave = new_dis[8] - 9
				new_dis[8]= 9
			else: return new_dis

			new_dis[9] += leave
			if new_dis[9] > 9:
				new_dis[0] =  9
			return new_dis
		else:
			return new_dis

	
				


	



			
	
			

