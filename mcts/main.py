#!/usr/bin/env python
# -*- coding:utf-8 -*-
from __future__ import print_function
from __future__ import division

from collections import defaultdict, deque
import random
from utils import *
from train import *


args = dotdict({
	'init_model':'./current_policy.model',
})







def main():
	
	train = Train(args.init_model)
	#train = Train()
	#train.human_play_game()
	#train.run()
	train.policy_evaluate(300)
		
			



if __name__ == '__main__':
	main()
