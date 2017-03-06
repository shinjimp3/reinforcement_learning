import numpy as np
import random
import matplotlib.pyplot as plt

class RandomWalk:
	def __init__(self, initial_state = 0):
		self.state = initial_state
		self.initial_state = initial_state
		self.step_cnt = 0
		self.state_size = 5
		self.action_size = 2


	def step(self, action_index):
		next_state = self.trans_state(self.state, action_index)
		reward = self.give_reward(self.state, action_index, next_state)
		s = self.state
		self.state = next_state
		self.step_cnt += 1
		return s, action_index, reward, next_state

	def trans_state(self, state_index, action_index):
		# P(s' | s, a) :define translate probably
		action_set = np.array([-1, 1])
		state = state_index
		action = action_set[action_index]

		next_state = state + action
		if random.random() < 0.1:
			next_state = state - action

		if next_state < 0 or next_state > 4:
			next_state = state
		
		return next_state

	def give_reward(self, state, action, next_state):
		# R(r | s, a, s') :define reward function
		if state == 4:
			reward = 10
		else:
			reward = 0
		return reward

	def render(self):
		view = str(self.step_cnt) + ": " + "-"*self.state + "o" + "-"*(4-self.state)
		print(view)

	def reset(self):
		self.state = self.initial_state
		self.step_cnt = 0

class RandomWalk2d(RandomWalk):
	def __init__(self, initial_state = np.array([0,0])):
		self.state = initial_state
		self.initial_state = initial_state
		self.step_cnt = 0
		self.state_size = np.array([5,5])
		self.action_size = np.array([3,3])

	def trans_state(self, state_index, action_index):
		# P(s' | s, a) :define translate probably
		state = state_index
		action = np.array(action_index) - 1

		next_state = state + action
		if random.random() < 0.1:
			next_state = state - action
		next_state = np.clip(next_state,0,4)
		
		return next_state

	def give_reward(self, state, action, next_state):
		# R(r | s, a, s') :define reward function
		if state[0] == 4:
			reward = 10
		else:
			reward = 0
		return reward

	def render(self):
		mat = np.zeros((5,5))
		mat[4,:] = np.ones((1,5))
		mat[self.state[0], self.state[1]] = 0.5
		plt.imshow(mat, interpolation='none', cmap='brg')
		plt.pause(.001)
		"""print(self.step_cnt)
		for i in range(5):
			if i == self.state[0]:
				view = "+"*self.state[1] + "o" + "+"*(4-self.state[1])
			else:
				view = "+"*5
			if i == 0:
				sys.stdout.write("\r%s\n" % view)
			else:
				sys.stdout.write("%s\n" % view)
		sys.stdout.flush()
		"""