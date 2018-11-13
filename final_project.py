import gym.spaces
import numpy as np

from cart_pole_controller import CartPoleController, BaselineController, QLearningController

np.set_printoptions(precision=2,suppress=True)

env = gym.make('CartPole-v1')

# Constants
MAX_TIMESTEPS = 200

def main():
	crappy_controller = BaselineController(Kp=1, Kd=0.5)
	test_controller(crappy_controller)

"""
Tests a given CartPoleController.
Inputs:
	controller = CartPoleController (BaseLineController or QLearningController)
Returns:
	num_successful_timesteps = number of successful timesteps
"""
def test_controller(controller):
	'''
	Run one episode of the environment with random actions.

	Episode stops when:
		- Pole Angle is more than +/-12 degrees
		- Cart Position is more than +/-2.4 (center of the cart reaches the edge of the display)
		- Episode length is greater than 200
	'''
	state = env.reset()
	num_successful_timesteps = 0
	for t in range(MAX_TIMESTEPS):
		# Render visualization of cart pole
		env.render()

		# Pick random action from action space
		action = controller.calc_response(state)

		# Take action and observe reward and next state 
		state, reward, done, info = env.step(action)

		# Terminate episode if it has reached stop condition(s)
		if done:
			print("Episode finished after {} timesteps".format(t+1))
			env.close()
			return t+1
	env.close()
	return MAX_TIMESTEPS

if __name__ == "__main__":
	main()