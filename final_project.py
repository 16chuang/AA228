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
from q_learning_acrobot import QLearningControllerAcrobot

from logger import Logger

np.set_printoptions(precision=2,suppress=True)

env_name = 'Acrobot-v1'
env = gym.make(env_name)

# Constants
MAX_TIMESTEPS 		= 500
NUM_TEST_EPISODES 	= 100
SOLVED_REWARD 		= 450
NUM_TRAIN_EPISODES	= 10000

def main():
	# theta_range = { 
	# 	'min': -.2, 
	# 	'max': .2, 
	# 	'num_bins': 20
	# }
	# dtheta_range = {
	# 	'min': -1.0,
	# 	'max': 1.0,
	# 	'num_bins': 6
	# }
	theta_range = { 
		'min': -1, 
		'max': 1, 
		'num_bins': 10
	}
	dtheta_range = {
		'min': -2.0,
		'max': 2.0,
		'num_bins': 6
	}
	# controller = QLearningController(theta_range=theta_range, dtheta_range=dtheta_range, learning_rate=.1, discount=1)
	controller = QLearningControllerAcrobot(theta_range=theta_range, 
		dtheta_range=dtheta_range, learning_rate=.1, discount=1)
	
	logger = Logger(env_name=env_name, algorithm_name='0_QLearning')
	logger.log_params([theta_range, dtheta_range, ])

	# train_controller_until_solved(controller)
	# train_controller_for_episodes(controller, num_episodes=NUM_TRAIN_EPISODES, logger=logger)
	
	for i in range(50):
		batch_rets = train_controller_one_epoch(controller, i, logger)
		print('epoch: %3d \t return: %.3f'%
                (i, np.mean(batch_rets)))
		logger.reset_episode_count()

	logger.log_data()

	# test_controller(controller)

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

def train_controller_one_epoch(controller, epoch_num, logger, batch_size=5000):
	batch_rewards = []
	num_obs = 0

	state = env.reset()
	done = False
	ep_rewards = 0

	while True:
		action = controller.calc_response(state, True)
		prev_state = state
		state, reward, done, _ = env.step(action)
		ep_rewards += reward
		num_obs += 1

		# Update controller with newly observed info
		controller.update(prev_state, action, reward, state)

		if done:
			batch_rewards.append(ep_rewards)
			logger.save_episode_data([epoch_num, ep_rewards])

			state, ep_rewards, done = env.reset(), 0, False

			if num_obs > batch_size:
				break

	return batch_rewards

"""
Trains a given CartPoleController for some number of episodes. Prints the average reward over
all training episodes.

Inputs:
	controller 			CartPoleController (BaseLineController or QLearningController)
"""
def train_controller_for_episodes(controller, num_episodes, logger):
	print("\nLET THE TRAINING BEGIN...")
	avg_reward = 0
	data = []

	for i in range(num_episodes):
		timesteps, reward = run_episode(controller, learning=True)
		logger.save_episode_data([0, reward])
		avg_reward += reward
		if i % (num_episodes/10) == 0:
			print("Episode {} finished after {} timesteps with total reward {}".format(i, timesteps+1, reward))

	print("Average reward over training episodes: {}".format(avg_reward * 1.0 / num_episodes))

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

	print("Average reward over 100 episodes achieved after {} training episodes: {}".format(avg_reward, num_episodes))

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