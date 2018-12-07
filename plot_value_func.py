import pandas as pd
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

filename = 'data/Acrobot-v1/5_pg_value_horizon_1544151772_values.csv'

def visualize(file):
	df = pd.read_csv(file)
	mean_values = df.groupby(['theta2'])['value'].mean()
	mean_values = np.array(mean_values)
	theta = np.linspace(0,2*np.pi,36)
	area = .5*(mean_values+60)**2
	r = [.1] * 36 # fixed radius
	colors = (mean_values - np.min(mean_values))/np.ptp(mean_values)

	fig = plt.figure()
	ax = fig.add_subplot(111, projection='polar')
	ax.set_theta_zero_location('S')
	ax.set_rmin(0)
	ax.set_rmax(.2)
	c = ax.scatter(theta, r, s=area, alpha=0.75, c=colors, cmap='winter')
	plt.title('Acrobot-v1, second link angle, learned values')

	plt.show()

if __name__ == "__main__":
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('--file', type=str)
	args = parser.parse_args()

	visualize(filename)