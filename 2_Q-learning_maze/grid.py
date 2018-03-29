import numpy as np
import pandas as pd
import time
EPSILON = 0.9


class Grid:

	# num = 4
	# obstacles = [[1,2],[2,1]]
	# goal = [2,2]
	# pos_i = 0
	# pos_j = 0

	def __init__(self, n=4):
		self.mat = np.zeros((n,n))
		self.num = n
		self.obstacles = [[1,2],[3,0]]
		self.goal = [2,2]
		self.pos_i = 0
		self.pos_j = 0

	def display_mat(self):
		# global num
		for i in range(self.num):
			for j in range(self.num):
				if [self.pos_i,self.pos_j] == [i,j]:
					print('o',end=' ') 
				elif [i,j] in self.obstacles :
					print('#',end=' ')
				elif [i,j]==self.goal :
					print('G',end=' ')
				else :
					print('-',end=' ')
			print('\n')
		# time.sleep(0.2)

	def choose_action(self,q_table, actions ):
		if (np.random.uniform() > EPSILON ) or ((q_table.loc[self.pos_i*self.num + self.pos_j,:] == 0).all()):
			action_name = np.random.choice(actions)
		else :
			action_name = q_table.loc[self.pos_i*self.num + self.pos_j ,:].idxmax()
		return action_name

	def get_reward_and_update_pos_and_return_previous_state(self,A):
		prev_state = self.pos_i*self.num + self.pos_j
		i_changed=0		
		j_changed=0		
		if A == 'l':
			if self.pos_j != 0:
				j_changed=1
				self.pos_j-=1;
		if A == 'r':
			if self.pos_j != self.num-1:
				j_changed=1
				self.pos_j+=1;
		if A == 'u':
			if self.pos_i != 0:
				i_changed=1
				self.pos_i-=1;
		if A == 'd':
			if self.pos_i != self.num-1:
				i_changed=1
				self.pos_i+=1;

		if ([self.pos_i,self.pos_j] in self.obstacles):
			self.pos_i = 'terminated'
			self.pos_j = 'terminated'
			return prev_state,-1
		elif ([self.pos_i,self.pos_j] == self.goal):
			self.pos_i = 'terminated'
			self.pos_j = 'terminated'
			return prev_state,10
		elif i_changed or j_changed:
			return prev_state,0
		else:
			return prev_state,-0.3
	
class Q_Table:
	def __init__(self, states , actions ):
		self.mat = pd.DataFrame(
			np.zeros((states, len(actions))),
			columns=actions,
		)
