"""
NOTES ON Q-LEARNING:
====================
theta_range:
	- performance jumps (for 100 episodes) from avg reward 10 when you decrease min-max range to [-.22,.22]
		- makes sense b/c episode ends when abs(theta) is past 12 degrees = .21 radians so all seen thetas fall in that range

dtheta_range:
	- performance jumps (for 100 episodes) from avg reward 10 to avg reward 100+ when you increase num_bins to 6


Good configuration:
	- theta_range = { 'min': -.2, 'max': .2, 'num_bins': 3 }
	- dtheta_range = { 'min': -1.0, 'max': 1.0, 'num_bins': 6 }
	- discount = .85
	- epsilon = .25
	- learning_rate = .5
	- num_episodes = 100
Results: 
	- Average training reward = 45.48
	- Test episode reward = 249.0
"""

import gym.spaces
import numpy as np

from cart_pole_controller import CartPoleController, BaselineController
from q_learning_controller import QLearningController

np.set_printoptions(precision=2,suppress=True)

env = gym.make('CartPole-v1')

# Constants
MAX_TIMESTEPS = 500

def main():
	theta_range = { 
		'min': -.2, 
		'max': .2, 
		'num_bins': 3
	}
	dtheta_range = {
		'min': -1.0,
		'max': 1.0,
		'num_bins': 6
	}
	controller = QLearningController(theta_range=theta_range, dtheta_range=dtheta_range, discount=0.85)
	train_controller(controller, num_episodes=100)

	test_controller(controller)

"""
Runs one episode using the given CartPoleController until termination.
Inputs:
	controller 			CartPoleController (BaseLineController or QLearningController)
Returns:
	timesteps 			number of successful timesteps
	total_reward 		total award accrued during the episode
"""
def run_episode(controller, learning):
	"""
	Episode stops when:
		- Pole Angle is more than +/-12 degrees
		- Cart Position is more than +/-2.4 (center of the cart reaches the edge of the display)
		- Episode length is greater than 200
	"""
	total_reward = 0

	state = env.reset()
	prev_state = state
	# controller.print_Q()

	for timesteps in range(MAX_TIMESTEPS):
		# Render visualization of cart pole
		# env.render()

		# Pick random action from action space
		action = controller.calc_response(state, learning)

		# Take action and observe reward and next state 
		prev_state = state
		state, reward, done, info = env.step(action)
		total_reward += reward

		# Update controller with newly observed info
		if learning:
			controller.update(prev_state, action, reward, state)

		# Terminate episode if it has reached stop condition(s)
		if done: break

	env.close()
	return timesteps+1, total_reward

"""
Trains a given CartPoleController for some number of episodes.
Inputs:
	controller 			CartPoleController (BaseLineController or QLearningController)
"""
def train_controller(controller, num_episodes):
	print("\nLET THE TRAINING BEGIN...")
	avg_reward = 0

	for i in range(num_episodes):
		timesteps, reward = run_episode(controller, learning=True)
		avg_reward += reward
		if i % (num_episodes/10) == 0:
			print("Episode {} finished after {} timesteps with total reward {}".format(i, timesteps+1, reward))

	print("Average reward: {}".format(avg_reward * 1.0 / num_episodes))

"""
Tests a given CartPoleController for one episode.
Inputs:
	controller 			CartPoleController (BaseLineController or QLearningController)
Returns:
	timesteps 			number of successful timesteps
"""
def test_controller(controller):
	print("\nTIME TO TEST WHAT YOU HAVE LEARNED, YOUNG PADAWAN...")
	timesteps, reward = run_episode(controller, learning=False)
	print("Episode finished after {} timesteps with total reward {}".format(timesteps+1, reward))

if __name__ == "__main__":
	main()