import numpy as np
import random

class QlearningAgent:
	def __init__(self, state_size, action_size):
		#self.Qtable = np.zeros(state_size+action_size)
		self.state = [0]
		self.qtable = Qtable(state_size, action_size)
		self.action_size = action_size
		self.state_size = state_size

	def observe_state(self, new_state):
		self.state = new_state

	def select_action(self, state, eps = 0.1, train = True):
		# epsilon greedy
		epsilon = eps
		if random.random() > epsilon or not train:
			action = self.qtable.argmax(state)
		else:
			action = []
			for a_s in self.action_size:
				action.append(np.random.randint(a_s))
		#print(action)
		return action

	def learn(self, state, action , reward, next_state):
		alpha = 0.1
		gamma = 0.9
		self.qtable.update_Q(state,
			action,
			self.qtable.ref_Q(state, action)*(1-alpha) + alpha*(reward + gamma*self.qtable.max(next_state)))
		

class Qtable:
	def __init__(self, state_size, action_size):
		self.straight_Q = np.zeros(np.prod(state_size)*np.prod(action_size))
		self.state_size = state_size
		self.action_size = action_size

	def update_Q(self, state, action, new_value):
		index = self.get_sa_index(state, action)
		self.straight_Q[index] = new_value

	def ref_Q(self, state, action):
		index = self.get_sa_index(state, action)
		return self.straight_Q[index]

	def max(self, state):
		start_i = self.get_sa_index(state, [0]*len(self.action_size))
		end_i =  start_i + np.prod(self.action_size)
		return np.max(self.straight_Q[start_i:end_i])

	def argmax(self, state):
		start_i = self.get_sa_index(state, [0]*len(self.action_size))
		end_i =  start_i + np.prod(self.action_size)
		index = np.argmax(self.straight_Q[start_i:end_i])
		action_i = []
		size_list = self.action_size
		for i in range(len(size_list)):
			action_i.append(index%size_list[-i-1])
			index = (index-index%size_list[-i-1])/size_list[-i-1]
		action_i.reverse()
		return action_i

	def get_sa_index(self, state, action):
		index_list = state+action
		size_list = self.state_size + self.action_size
		index = 0
		for sa_index in range(len(index_list)):
			if sa_index < len(index_list)-1:
				index += index_list[sa_index]*np.prod(size_list[sa_index+1:])
			else:
				index += index_list[sa_index]
		return index

	def show(self):
		self.recurent_show([0],self.state_size+self.action_size)

	def recurent_show(self, index_list, size_list):
		dim = len(index_list)
		full_dim = len(size_list)
		if dim == full_dim:
			start_i = self.get_sa_index(index_list[:len(self.state_size)], index_list[len(self.state_size):])
			end_i = start_i + size_list[-1]
			print(self.straight_Q[start_i:end_i])
		else:
			for index in range(size_list[dim-1]):
				self.recurent_show(index_list+[index], size_list)
			print("")
