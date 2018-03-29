import numpy as np
import pandas as pd	
import time

N_STATES = 6 	# number of states
ACTIONS = ['left','right']
MAX_EPISODES = 13
REFRESH_TIME = 0.3
LR = 0.1 		# learning rate
EPSILON = 0.9 # to select either using exploration or using exploitation
GAMMA = 0.9

def build_q_table(states, actions):
	''' 
		@states : int 
		@actions : list of strings, (basically list of names of all the actions)
	'''
	# q table is of the form states_actions
	q_table = pd.DataFrame(
		np.zeros((states, len(actions))),
		columns=actions,
	)
	return q_table

def choose_action(state , q_table):
	if (np.random.uniform() > EPSILON ) or ((q_table.iloc[state,:] == 0).all()):
		action_name = np.random.choice(ACTIONS)
	else :
		action_name = q_table.iloc[state,:].idxmax()
	return action_name

def get_feedback_from_env(current_S, action):
	reward = 0
	next_S = 0
	if action == 'left':
		if current_S == 0 :
			next_S = 0
		else:
			next_S = current_S-1
	else:
		if current_S == N_STATES-2:
			next_S = 'terminated'
			reward=1
		else:
			next_S = current_S+1
	return next_S,reward

def update_the_env(S, episode,step_counter):
	disp_str = ['-']*(N_STATES-1)+['G']
	if S=='terminated':
		print('\rEpisode = {} and Steps = {}'.format(episode,step_counter))
	else:
		disp_str[S] = 'o'
		disp_str = ''.join(disp_str)
		print('\r{}'.format(disp_str),end='')
	time.sleep(REFRESH_TIME)

def rl():
	# main part of the code
	q_table = build_q_table(N_STATES, ACTIONS)
	for episode in range(MAX_EPISODES):
		S = 0
		is_terminated = False
		step_counter = 0
		update_the_env(S, episode,step_counter);
		while not is_terminated:

			A = choose_action(S,q_table)
			next_S,R = get_feedback_from_env(S, A)

			q_predict = q_table.loc[S, A]
			if next_S == 'terminated':
				q_target = R
				is_terminated = True
			else:
# NOTE : updating the table using both the action is not required.		
				q_target = R + GAMMA * q_table.iloc[next_S, :].max()   # next state is not terminal
			q_table.loc[S, A] += LR * (q_target - q_predict)  # update

			S = next_S
			step_counter = step_counter+1
			update_the_env(S, episode,step_counter)

	return q_table

rl()



