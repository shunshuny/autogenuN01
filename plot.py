import numpy as np
import matplotlib.pylab as plt
#import seaborn as sns
import os
import math
import csv

logname = 'tank01'
xlog = logname + '_x.log'
tlog = logname + '_t.log'
logdir = os.path.join('.', 'logs', logname)
#logdir = os.path.join('.', 'generated', logname, 'log')
x_file = os.path.join(logdir, xlog)
t_file = os.path.join(logdir, tlog)
# データを読む
x_data = np.genfromtxt(x_file)
t_data = np.genfromtxt(t_file)

# Replace NaN with 0.
t_data[np.isnan(t_data)] = 0
x_data[np.isnan(x_data)] = 0


# グラフを描画
plt.plot(t_data, x_data[:, 0], color = 'blue', label='water level 1')
plt.plot(t_data, x_data[:, 1], color = 'red', label='water level 2')
plt.plot(t_data, x_data[:, 2], color = 'green', label='water level 3')
#plt.plot(t_data, x_data[:, 59], color = 'black')
plt.xlabel('t')
plt.ylabel('water level')
plt.grid(True)
plt.legend()
plt.show()