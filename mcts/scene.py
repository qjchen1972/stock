# -*- coding:utf-8 -*-
from __future__ import print_function
from __future__ import division
import random
import numpy as np

class Scene(object):

	def __init__(self,file=None):
		
		
		self.scene = []

		if file:
			
			with open(file, "r") as lf:			
				while True:                
					line = lf.readline()
					if not line:break				
					items = line.split()
					if not items: break            
					self.scene.append([items[0],[int(i) for i in items[1:]]])
		else:
			
			#seq = np.random.normal(loc = 5,scale=10,size =30)
			#print(seq)
			seq=(1,3,5,7,9,9,7,5,3,1)
			for i in range(1000):
				prob = seq 
				#random.sample(seq,10)
				scene = random.randint(0,9)
				self.scene.append([scene,prob])


	def get_item(self,scene_num = 11, shuffle = True):
		
		if shuffle:	
			index = random.randint(0,len(self.scene) - scene_num)
		else:

			self.index += 2
			index = self.index
			
		return self.scene[index:index+scene_num]

	def reset(self):
		self.index = 0


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
			


				


