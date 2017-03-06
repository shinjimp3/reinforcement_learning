# reinforcement_learning
This repository contains these script
* Qlearning.py  
  This runs Qlearning program.
  It makes an agent, an environment, learning loops and interaction between the agent and environment.
* QlearningAgent.py  
  This is the agent which learns policy by Q-learning method and epsilon-greedy search.
* Environment.py  
  This contains class RandomWalk and RandomWalk2d.
  It has interfaces step() and reset() like gym.

# How to use class Qtable
As a numpy array is accessed by integer index, the class Qtable enables you to access Q-table by numpy array.
Even if you define multi dimension states and actions,
you need not input each element of them to Q-table like Q(s[0],s[1],s[2],a[0],a[1]).
You need only input numpy states and actions like qtable.ref_Q(s,a).
* update_Q(state, action, new_value)
  * state:numpy
  * action:numpy
  * new_value:float  
  Update Q-table like Q(s,a) <- new_value.
* ref_Q(state, action)
  * state:numpy
  * action:numpy  
  Return the value of Q(s,a)
* max(state)
  * state:numpy  
  Return the value of max Q(s,:).
* argmax(state)
  * state:numpy  
  Return the action which maximize Q(s,:).
* show()
  * Show Q-table
