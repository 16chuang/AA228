import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

ENV_FILES = {
	'CartPole-v0': {
		# 'RL baseline': 'data/CartPole-v0/0_QLearning_1544178826.csv',
		'PG basic': 'data/CartPole-v0/1_pg_simple_1544111565.csv',
		'Future reward': 'data/CartPole-v0/2_pg_future_reward_1544114754.csv',
		'With value': 'data/CartPole-v0/3_pg_with_baseline_1544116631.csv',
		'With GAE': 'data/CartPole-v0/4_pg_gae_1544211165.csv', #gam=.98,lam=.94
	},
	'CartPole-v1': {
		# 'RL baseline': 'data/CartPole-v1/0_QLearning_1544177831.csv',
		'PG basic': 'data/CartPole-v1/1_pg_simple_1544113163.csv',
		'Future reward': 'data/CartPole-v1/2_pg_future_reward_1544115237.csv',
		'With value': 'data/CartPole-v1/3_pg_with_baseline_1544117249.csv',
		'With GAE': 'data/CartPole-v1/4_pg_gae_1544210649.csv', #gam=.98,lam=.94
		# 'With value horizon': 'data/CartPole-v1/5_pg_value_horizon_1544135663.csv',
	}, 
	'Acrobot-v1': {
		# 'RL baseline': 'data/Acrobot-v1/0_QLearning_1544180230.csv',
		'PG basic': 'data/Acrobot-v1/1_pg_simple_1544113774.csv',
		'Future reward': 'data/Acrobot-v1/2_pg_future_reward_1544115746.csv',
		'With value': 'data/Acrobot-v1/3_pg_with_baseline_1544117832.csv',
		'With GAE': 'data/Acrobot-v1/4_pg_gae_1544119937.csv',
		'With value horizon': 'data/Acrobot-v1/5_pg_value_horizon_1544168801.csv',
	},
	'GAE': {
		'gam = .95, lam = .92': 'data/CartPole-v1/4_pg_gae_1544190508.csv',
		'gam = .95, lam = .935': 'data/CartPole-v1/4_pg_gae_1544190811.csv',
		'gam = .95, lam = .95': 'data/CartPole-v1/4_pg_gae_1544191127.csv',
		'gam = .95, lam = .965': 'data/CartPole-v1/4_pg_gae_1544191440.csv',
		'gam = .95, lam = .98': 'data/CartPole-v1/4_pg_gae_1544191756.csv',
		'gam = .95, lam = .995': 'data/CartPole-v1/4_pg_gae_1544192081.csv',

		'gam = .96, lam = .92': 'data/CartPole-v1/4_pg_gae_1544192394.csv',
		'gam = .96, lam = .935': 'data/CartPole-v1/4_pg_gae_1544192712.csv',
		'gam = .96, lam = .95': 'data/CartPole-v1/4_pg_gae_1544193013.csv',
		'gam = .96, lam = .965': 'data/CartPole-v1/4_pg_gae_1544193328.csv',
		'gam = .96, lam = .98': 'data/CartPole-v1/4_pg_gae_1544193634.csv',
		'gam = .96, lam = .995': 'data/CartPole-v1/4_pg_gae_1544193940.csv',
	}
}

def plot_one_file(filename, plot_episode=False):
	if 'csv' not in filename:
		filename = "{}.csv".format(filename)
	df = pd.read_csv(filename)
	assert('episode' in df.columns and 'epoch' in df.columns and 'reward' in df.columns)

	x = 'episode' if plot_episode else 'epoch'
	sns.lineplot(x=x, y="reward", data=df)#, hue='')#, ci='sd')
	plt.show()

def plot_one_env(env_name, files_dict, plot_episode=False):
	frames = []

	for key, file in files_dict.items():
		df = pd.read_csv(file)
		df['algorithm'] = key
		frames.append(df)

	all_data = pd.concat(frames)

	x = 'episode' if plot_episode else 'epoch'
	fig, ax = plt.subplots()
	sns.lineplot(x=x, y="reward", data=all_data, hue='algorithm', ax=ax)#, ci='sd')
	# plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
	plt.title(env_name)
	plt.show()

if __name__ == "__main__":
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('--env', type=str)
	parser.add_argument('--file', type=str)
	parser.add_argument('--episode', '--ep', action='store_true')
	args = parser.parse_args()

	sns.set(style="white")

	if args.file:
		plot_one_file(args.file, args.episode)
	elif args.env:
		if args.env in ENV_FILES:
			plot_one_env(args.env, ENV_FILES[args.env], args.episode)
	