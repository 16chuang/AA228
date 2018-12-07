import numpy as np

"""
Abstract class for controllers used to stabilize acrobot.
"""
class AcrobotController:
	"""
	Calculates an action given the current state of the acrobot.
	Inputs:
		state
			state[0] = cos(theta1)
			state[1] = sin(theta1)
			state[2] = cos(theta2)
			state[3] = sin(theta2)
	Returns:
		action
			1 = + torque on joint 2
			0 = 0 torque on joint 2
			-1 = - torque on joint 2
	"""
	def calc_response(self, state):
		pass

"""
Controller used to stabilize cart-pole with a PD control loop.
"""
class BaselineAcrobotController(AcrobotController):
	"""
	Initializes a BaselineController
	Inputs:
		Kp = proportional gain
		Kd = derivative gain
	Returns:
		None
	"""
	def __init__(self, Kp, Kd, max_alpha):
		self.Kp = Kp
		self.Kd = Kd
		self.max_alpha = max_alpha

		self.theta1_old = - np.pi / 2
		self.theta2_old = - np.pi / 2

		self.dtheta1_old = 1
		self.dtheta2_old = 1

		self.target_alpha = np.pi/2 # starting value

	def calc_response(self, state):
		theta1 = np.arctan2(state[1], state[0])
		theta2 = np.arctan2(state[3], state[2])
		dtheta1 = state[4]
		dtheta2 = state[5]

		# dtheta1 = theta1 - self.theta1_old
		# dtheta2 = theta2 - self.theta2_old

		if (self.dtheta1_old > 0 and dtheta1 <= 0):
			# pendulum has switched direction of swing from CW to CCW
			# make theta2 +pi/2
			self.target_alpha = self.max_alpha
		elif (self.dtheta1_old < 0 and dtheta1 >= 0):
			# pendulum has switched direction of swing from CCW to CW
			# make theta2 -pi/2
			self.target_alpha = -self.max_alpha

		# torque on joint 2 to maintain target alpha
		theta2_err = theta2 - self.target_alpha
		theta2_derr = dtheta2
		theta2_torque = -self.Kp * theta2_err - self.Kd * theta2_derr

		# update stored vars
		self.theta1_old = theta1
		self.theta2_old = theta2
		self.dtheta1_old = dtheta1
		self.dtheta2_old = dtheta2

		if theta2_torque > 0:
			print("CW\t\t dtheta1={:.2f}".format(dtheta1))
			return 1 # CW torque
		elif theta2_torque < 0:
			print("CCW\t\t dtheta1={:.2f}".format(dtheta1))
			return -1 # CCW torque
		else:
			print("0")
			return 1 # no torque