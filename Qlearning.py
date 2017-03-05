import numpy as np
import random
import Environment as environment
import QlearningAgent as qagent

env = environment.RandomWalk2d([0,0])
agent = qagent.QlearningAgent(env.state_size, env.action_size)
for episode in range(1):
	# "Episodic task" has the unit "episode"
	env.reset()
	agent.observe_state(env.state)
	print('episode{}'.format(episode))
	reward_sum = 0

	for step in range(500):
		env.render()
		state = agent.state
		if step < 400:
			eps = 1
		else:
			eps = 0.1
		action = agent.select_action(state,eps = eps)
		state, action, reward, next_state = env.step(action)
		reward_sum += reward
		agent.observe_state(next_state)
		agent.learn(state, action, reward, next_state)
	print('Reward sum is {}.'.format(reward_sum))
	print('')
print('Qtable')
agent.qtable.show()
