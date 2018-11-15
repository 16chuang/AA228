import numpy as np
from cart_pole_controller import CartPoleController

"""
Controller used to stabilize cart-pole with Q learning + epsilon greedy. 
Discretizes theta and dtheta to use as the state.
"""
class QLearningController(CartPoleController):
	"""
	Creates new simple Q learning controller: saves hyperparameters and other helper info,
	and initializes a Q value table whose rows are states and columns are actions.

	Inputs:
		theta_range		dictionary of min, max, and num_bins for theta
		dtheta_range	dictionary of min, max, and num_bins for dtheta
		epsilon			hyperparameter for probability of random vs greedy action
		learning_rate	hyperparameter for step size per Q value update
		discount		hyperparameter for how much reward is discounted over time
	"""
	def __init__(self, theta_range, dtheta_range, epsilon=0.25, learning_rate=.5, discount=0.9):
		# Ranges for discretizing theta and dtheta
		self.theta_num_bins = theta_range['num_bins']
		self.dtheta_num_bins = dtheta_range['num_bins']
		self.theta_bins = np.linspace(theta_range['min'], theta_range['max'], theta_range['num_bins']-1)
		self.dtheta_bins = np.linspace(dtheta_range['min'], dtheta_range['max'], dtheta_range['num_bins']-1)

		# Hyperparameters
		self.discount = discount
		self.epsilon = epsilon
		self.learning_rate = learning_rate

		# Action and state space
		self.actions = np.array([0, 1])
		self.num_actions = len(self.actions)
		self.num_states = theta_range['num_bins'] * dtheta_range['num_bins']
		# Q value table
		self.Q = np.zeros((self.num_states, self.num_actions))

	"""
	Private helper function to discretize a full state and map it to an index in the Q table.
	index / # dtheta bins = theta
	index % # dtheta bins = dtheta

	Inputs:
		state 			Full state (x, dx, theta, dtheta)
	Returns:
		index 			Corresponding index into a row of the Q table
	"""
	def state_to_index(self, state):
		x, dx, theta, dtheta = state
		discretized_theta = np.digitize(theta, self.theta_bins)
		discretized_dtheta = np.digitize(dtheta, self.dtheta_bins)
		
		index = discretized_theta * self.dtheta_num_bins + discretized_dtheta
		return index

	"""
	Updates Q value given a new observation. Called while controller is learning.

	Inputs:
		state 			State we were just in
		action 			Action we just took
		reward 			Reward we received for that state, action pair
		next_state		Sampled next state
	"""
	def update(self, state, action, reward, next_state):
		s = self.state_to_index(state)
		sp = self.state_to_index(next_state)
		Q_update = reward + self.discount * np.max(self.Q[sp])
		self.Q[s, action] = (1 - self.learning_rate) * self.Q[s, action] + self.learning_rate * Q_update

	"""
	Returns a well-performing action given the current state. When learning is true,
	selects actions with an epsilon greedy exploration policy. When learning is false,
	selects actions solely based on the Q table.

	Inputs:
		state 			Current state observed that we're deciding an action for
		learning 		Whether to use epsilon greedy exploration
	"""
	def calc_response(self, state, learning=False): 
		# If learning, not evaluating, include epsilon greedy exploration
		if learning and np.random.rand() < self.epsilon:
			return np.random.choice(self.actions)
		else:
			s = self.state_to_index(state)
			return np.argmax(self.Q[s])

	def print_Q(self):
		print(self.Q)