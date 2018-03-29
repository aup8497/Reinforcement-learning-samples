import numpy as np
import pandas as pd
from grid import Grid, Q_Table
import os

LEARNING_RATE = 0.1
EPSILON = 0.9
GAMMA = 0.9
MAX_EPISODES = 50
n = 4
ACTIONS = ['l','r','u','d']
N_STATES = n*n

grid = Grid()

def RL():
	q_table = Q_Table(N_STATES,ACTIONS)
	for episode in range(MAX_EPISODES):

		grid.pos_i = 0
		grid.pos_j = 0
		is_terminated = False
		grid.display_mat()
		os.system("clear")
		A = grid.choose_action(q_table.mat, ACTIONS)
		while not is_terminated:

			prev_state , R = grid.get_reward_and_update_pos_and_return_previous_state( A)
			print("A=",A)
			q_predict = q_table.mat.loc[ prev_state , A ]
			if grid.pos_i == 'terminated':
				q_target = R
				is_terminated = True
			else:
# NOTE : updating the table using both the action is not required.		
				A_ = grid.choose_action(q_table.mat, ACTIONS)
				q_target = R + GAMMA * q_table.mat.loc[grid.pos_i*n + grid.pos_j, A_ ]   # next state is not terminal
			q_table.mat.loc[prev_state, A] += LEARNING_RATE * (q_target - q_predict)  # update
			# step_counter = step_counter+1
			# update_the_env(S, episode,step_counter)
			A=A_
			grid.display_mat()
			os.system("clear")
		# print(q_table.mat)
RL()