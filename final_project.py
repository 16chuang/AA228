"""
NOTES ON Q-LEARNING:
====================
theta_range:
	- performance jumps (for 100 episodes) from avg reward 10 when you decrease min-max range to [-.22,.22]
		- makes sense b/c episode ends when abs(theta) is past 12 degrees = .21 radians so all seen thetas fall in that range
	- performance seems to steadily increase when adding more bins to theta, but also takes longer to train

dtheta_range:
	- performance jumps (for 100 episodes) from avg reward 10 to avg reward 100+ when you increase num_bins to 6
	- performance doesn't steadily increase when adding more bins to dtheta

discount:
	- cannot discount (i.e. discount = 1) otherwise Q values will eventually converge to same number and performance will drop

learning_rate:
	- learning rates higher than .1 don't seem to do well


Solved configuration (only happened once though):
	- theta_range = { 'min': -.2, 'max': .2, 'num_bins': 20 }
	- dtheta_range = { 'min': -1.0, 'max': 1.0, 'num_bins': 6 }
	- discount = 1
	- epsilon = .25
	- learning_rate = .1
Results:
	- number of training episodes = 3882
	- average reward of test episodes = 500
"""

import gym.spaces
import numpy as np
from collections import deque

from cart_pole_controller import CartPoleController, BaselineController
from q_learning_controller import QLearningController

np.set_printoptions(precision=2,suppress=True)

env = gym.make('CartPole-v1')

# Constants
MAX_TIMESTEPS 		= 500
NUM_TEST_EPISODES 	= 100
SOLVED_REWARD 		= 450

def main():
	theta_range = { 
		'min': -.2, 
		'max': .2, 
		'num_bins': 15
	}
	dtheta_range = {
		'min': -1.0,
		'max': 1.0,
		'num_bins': 6
	}
	controller = QLearningController(theta_range=theta_range, dtheta_range=dtheta_range, learning_rate=.1, discount=1)
	# train_controller_until_solved(controller)
	train_controller_for_episodes(controller, num_episodes=2000)

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
Trains a given CartPoleController for some number of episodes. Prints the average reward over
all training episodes.

Inputs:
	controller 			CartPoleController (BaseLineController or QLearningController)
"""
def train_controller_for_episodes(controller, num_episodes):
	print("\nLET THE TRAINING BEGIN...")
	avg_reward = 0

	for i in range(num_episodes):
		timesteps, reward = run_episode(controller, learning=True)
		avg_reward += reward
		if i % (num_episodes/10) == 0:
			print("Episode {} finished after {} timesteps with total reward {}".format(i, timesteps+1, reward))

	print("Average reward: {}".format(avg_reward * 1.0 / num_episodes))

"""
Continues training a given CartPoleController until it "solves" the environment by achieving an average
reward of SOLVED_REWARD over NUM_TEST_EPISODES episodes. Prints the average reward over the most recent
NUM_TEST_EPISODES episodes.

Inputs:
	controller 			CartPoleController (BaseLineController or QLearningController)
"""
def train_controller_until_solved(controller):
	print("\nLET THE TRAINING BEGIN...")
	sliding_reward_deque = deque([])
	sliding_reward = 0
	avg_reward = 0
	num_episodes = 0

	while avg_reward < SOLVED_REWARD:
		timesteps, reward = run_episode(controller, learning=True)

		# Append reward
		sliding_reward_deque.append(reward)
		sliding_reward += reward
		if len(sliding_reward_deque) > NUM_TEST_EPISODES: 
			sliding_reward -= sliding_reward_deque.popleft()

		avg_reward = sliding_reward * 1.0 / NUM_TEST_EPISODES

		# Increment number of episodes
		num_episodes += 1
		if num_episodes % 100 == 0:
			# controller.print_Q()
			print("Episode {}, current average reward: {}".format(num_episodes, avg_reward))

	print("Average reward achieved after {} training episodes: {}".format(avg_reward, num_episodes))

"""
Tests a given CartPoleController for NUM_TEST_EPISODES episodes. Prints average reward
over all the test episodes.

Inputs:
	controller 			CartPoleController (BaseLineController or QLearningController)
Returns:
	timesteps 			number of successful timesteps
"""
def test_controller(controller):
	print("\nTIME TO TEST WHAT YOU HAVE LEARNED, YOUNG PADAWAN...")
	avg_reward = 0

	for i in range(NUM_TEST_EPISODES):
		timesteps, reward = run_episode(controller, learning=False)
		avg_reward += reward
		if i % (NUM_TEST_EPISODES/10) == 0:
			print("Episode {} finished after {} timesteps with total reward {}".format(i, timesteps+1, reward))

	print("Average reward over {} episodes: {}\n".format(NUM_TEST_EPISODES, avg_reward * 1.0 / NUM_TEST_EPISODES))

if __name__ == "__main__":
	main()