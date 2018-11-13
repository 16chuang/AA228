import gym.spaces
import numpy as np
np.set_printoptions(precision=2,suppress=True)

'''
Create CartPole environment
https://gym.openai.com/envs/CartPole-v1/
https://github.com/openai/gym/wiki/CartPole-v0
'''
env = gym.make('CartPole-v1')

'''
Action space: {-1 left, 1 right}
'''
action_space = env.action_space			

'''
State space: { cart position, car velocity, pole angle, pole velocity at tip }
'''
state_space = env.observation_space		
max_state = state_space.high			# [4.80 3.40e+38 .419 3.40e+38]
min_state = state_space.low				# [-4.80e -3.40e+38 -.419 -3.40e+38]

'''
Constants
'''
MAX_TIMESTEPS = 200



'''
Run one episode of the environment with random actions.

Episode stops when:
	- Pole Angle is more than +/-12 degrees
	- Cart Position is more than +/-2.4 (center of the cart reaches the edge of the display)
	- Episode length is greater than 200
'''
state = env.reset()
for t in range(MAX_TIMESTEPS):
	# Render visualization of cart pole
	env.render()

	# Pick random action from action space
	action = action_space.sample()

	# Take action and observe reward and next state 
	state, reward, done, info = env.step(action)
	
	# Terminate episode if it has reached stop condition(s)
	if done:
		print("Episode finished after {} timesteps".format(t+1))
		break