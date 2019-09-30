# -*- coding: utf-8 -*-

import numpy as np
import copy

class TreeNode(object):

	def __init__(self, parent, prior_p):
		self._parent = parent
		self._children = {}  # a map from action to TreeNode
		self._n_visits = 0
		self._Q = 0
		self._u = 0
		self._P = prior_p

	def expand(self, action_priors):
		    
		for action, prob in action_priors:
			#print(action, prob)
			if action not in self._children:
				self._children[action] = TreeNode(self, prob)

	def select(self, c_puct):

		#act_visits = [(act, node.get_value(c_puct)) for act, node in self._children.items()]        
		#print('\n',act_visits)
		return max(self._children.items(),
			key=lambda act_node: act_node[1].get_value(c_puct))


	def update(self, leaf_value):
        
		#self._Q = (self._Q * self._n_visits + leaf_value)
		#self._n_visits += 1
		#self._Q /= self._n_visits
		self._n_visits += 1
		self._Q += 1.0*(leaf_value - self._Q) / self._n_visits


	def update_recursive(self, leaf_value):
        
		if self._parent:
			self._parent.update_recursive(leaf_value)
		self.update(leaf_value)


	def get_value(self, c_puct):
        
		self._u = (c_puct * self._P *np.sqrt(self._parent._n_visits) / (1 + self._n_visits))
		return self._Q + self._u


	def is_leaf(self):
 
		return self._children == {}


	def is_root(self):
		return self._parent is None


class MCTS(object):
 
	def __init__(self, policy_value_fn, c_puct=5, n_playout=10000):
     
		self._root = TreeNode(None, 1.0)
		self._policy = policy_value_fn
		self._c_puct = c_puct
		self._n_playout = n_playout


	def _playout(self, state):
    
		node = self._root
		
		while(1):
			if node.is_leaf(): break
			action, node = node.select(self._c_puct)
			state.do_move(action)
			#print(state.game_show())

		#print(state.game_show())
		end = state.game_end()
		if not end:
			action_probs, leaf_value = self._policy(state)
			node.expand(action_probs)
		else:   
			profit = state.game_profit()
			if profit >= 50: leaf_value = 1.0
			#elif profit > 0: leaf_value = 1.0
			else: leaf_value = -1.0

		node.update_recursive(leaf_value)


	def get_move_probs(self, state, temp=1e-3):
        
		for n in range(self._n_playout):
			state_copy = copy.deepcopy(state)
			self._playout(state_copy)

		
		act_visits = [(act, node._n_visits) for act, node in self._root._children.items()]
		#print('\n')
		#print(act_visits)
		acts, visits = zip(*act_visits)
		act_probs = self._softmax(1.0/temp * np.log(np.array(visits) + 1e-10))

		return acts, act_probs


	def update_with_move(self, last_move):
        
		if last_move in self._root._children:
			self._root = self._root._children[last_move]
			self._root._parent = None
		else:
			self._root = TreeNode(None, 1.0)

	def _softmax(self,x):

		probs = np.exp(x - np.max(x))
		probs /= np.sum(probs)
		return probs


	def __str__(self):
		return "MCTS"


class MCTSPlayer(object):


	def __init__(self, policy_value_function,c_puct=5, n_playout=2000, is_selfplay=0):
		
		self.mcts = MCTS(policy_value_function, c_puct, n_playout)
		self._is_selfplay = is_selfplay


	def reset_player(self):
		
		self.mcts.update_with_move(-1)


	def get_action(self, board, temp=1e-3, return_prob=0):
        
		move_probs = np.zeros(board.time_step)
		sensible_moves = board.availables
		#print(sensible_moves)
		
		
		acts, probs = self.mcts.get_move_probs(board, temp)
		#print('acts',acts)
		#print('probs',probs)
		move_probs[list(acts)] = probs
		#print('ok',move_probs)
		if self._is_selfplay:

			move = np.random.choice(acts,p=0.75*probs + 0.25*np.random.dirichlet(0.3*np.ones(len(probs))))
			#print(move,np.argmax(probs))
			#move = np.argmax(probs)
			self.mcts.update_with_move(move)
		else:
			move = np.random.choice(acts, p=probs)

			self.mcts.update_with_move(-1)

		#exit()
		if return_prob:return move, move_probs
		else:return move

	def __str__(self):
		return "MCTS {}".format('alpha zero')