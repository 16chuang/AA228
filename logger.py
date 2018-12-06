import pandas as pd
import json
import time
import os

class Logger(object):
	def __init__(self, env_name, algorithm_name, columns=[
			'episode',
			'epoch',
			'reward',
			# 'value',
			# 'action',
			# 'state',
		]):
		self.filename = "data/{}/{}_{}".format(env_name, algorithm_name, int(time.time()))
		if not os.path.exists(os.path.dirname(self.filename)):
			try:
				os.makedirs(os.path.dirname(self.filename))
			except OSError as exc: # Guard against race condition
				if exc.errno != errno.EEXIST:
					raise

		self.columns = columns
		self.data = []
		self.episode_counter = 0

	def log_params(self, params):
		logfile = open("{}.params".format(self.filename), 'w+')
		logfile.write(json.dumps(params))
		logfile.close()

	def save_episode_data(self, row):
		row = [self.episode_counter] + row
		assert(len(self.columns) == len(row))
		self.data.append(dict(zip(self.columns, row)))
		self.episode_counter += 1

	def reset_episode_count(self):
		self.episode_counter = 0

	def log_data(self):
		logfile = "{}.csv".format(self.filename)
		self.df = pd.DataFrame(self.data, columns=self.columns)
		self.df.to_csv(logfile)
		