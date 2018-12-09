import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

df = pd.read_csv('gae_tuning.csv')
pivoted = pd.pivot_table(df, index='gamma', columns='lambda', values='avg_return')
sns.heatmap(pivoted, cmap='Blues')
plt.show()