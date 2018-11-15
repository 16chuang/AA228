import numpy as np

"""
Abstract class for controllers used to stabilize cart-pole.
"""
class CartPoleController:
	"""
	Calculates an action given the current state of the cart-pole.
	Inputs:
		state
			state[0] = x (cart position)
			state[1] = dx (cart velocity)
			state[2] = theta (pole angle, CW+ from straight up)
			state[3] = pole tip velocity (CW+)
	Returns:
		action
			1 = push cart to right
			-1 = push cart to left
	"""
	def calc_response(self, state):
		pass

"""
Controller used to stabilize cart-pole with a PD control loop.
"""
class BaselineController(CartPoleController):
	"""
	Initializes a BaselineController
	Inputs:
		Kp = proportional gain
		Kd = derivative gain
	Returns:
		None
	"""
	def __init__(self, Kp, Kd):
		self.Kp = Kp
		self.Kd = Kd

	def calc_response(self, state):
		# break out state into variables
		theta = state[2]
		# divide tip velocity by circumference of arc for pole length 1 to get angular velocity
		dtheta = state[3] / (2 * np.pi)

		response = self.Kp * theta + self.Kd * dtheta
		if response > 0:
			return 1
		else:
			return 0