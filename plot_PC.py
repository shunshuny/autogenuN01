import numpy as np
import matplotlib.pylab as plt
#import seaborn as sns
import os
import math
import csv

logname = 'linear_PC01'
xlog = logname + '_x.log'
tlog = logname + '_t.log'
ulog = logname + '_u.log'
logdir = os.path.join('.', 'logs', logname)
# logdir = os.path.join('.', 'linear', 'generated', logname, 'log')
x_file = os.path.join(logdir, xlog)
t_file = os.path.join(logdir, tlog)
u_file = os.path.join(logdir, ulog)
# データを読む
x_data = np.genfromtxt(x_file)
t_data = np.genfromtxt(t_file)
u_data = np.genfromtxt(u_file)

# Replace NaN with 0.
t_data[np.isnan(t_data)] = 0
x_data[np.isnan(x_data)] = 0
u_data[np.isnan(u_data)] = 0

p_terms = 4
u_nominal = u_data[:,[0,p_terms]]

def calcurate_variance(data, index, p_terms):
    var = 0
    for i in range(1,p_terms):
        var += (data[:,(index-1)*p_terms+i])**2
    return var

p_terms = 4
mean_x1 = x_data[:,0]
mean_x2 = x_data[:,p_terms]
var_x1 = calcurate_variance(x_data, 1, p_terms) 
var_x2 = calcurate_variance(x_data, 2, p_terms)

# t-xのグラフを描画
plt.plot(t_data, mean_x1, color = 'blue', label='mean_x1')
plt.fill_between(t_data, mean_x1 - np.sqrt(var_x1), mean_x1 + np.sqrt(var_x1), alpha=0.3, color="blue", label="±1σ")
plt.plot(t_data, mean_x2, color = 'red', label='mean_x2')
plt.fill_between(t_data, mean_x2 - np.sqrt(var_x2), mean_x2 + np.sqrt(var_x2), alpha=0.3, color="red", label="±1σ")
plt.xlabel('t', fontsize=14)
plt.ylabel('x', fontsize=14)
plt.grid(True)
plt.legend(fontsize=12)
plt.show()

# t-uのグラフを描画
plt.plot(t_data, u_nominal[:,0], color = 'blue', label='u1')
plt.plot(t_data, u_nominal[:,1], color = 'red', label='u2')
plt.xlabel('t', fontsize=14)
plt.ylabel('u', fontsize=14)
plt.grid(True)
plt.legend(fontsize=12)
plt.show()