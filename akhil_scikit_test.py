#!/usr/bin/python
import math
import random
import csv
import traceback
from sklearn import linear_model
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

FILE = 'admission.csv'
lines = csv.reader(open(FILE))
data = list(lines)
all_data = np.array(data)[:,[0,1,2]].astype(int)
all_x = np.array(data)[:,[0,1]].astype(int)
x_1 = np.array(data)[:,[0]].astype(int)
x_2 = np.array(data)[:,[1]].astype(int)
y = np.array(data)[:,[2]].astype(int)

x_min, x_max = all_x[:, 0].min() - .5, all_x[:, 0].max() + .5
y_min, y_max = all_x[:, 1].min() - .5, all_x[:, 1].max() + .5

plt.xlim(x_min, x_max)
plt.xticks(np.arange(x_min, x_max,0.2))
plt.ylim(y_min, y_max+40)
plt.yticks(np.arange(y_min, y_max,0.2))

Z = y
ax = plt.gca()
x_1_line = np.linspace(x_min, x_max, 50)
x_2_line = np.linspace(y_min, y_max, 50)
cs = ax.contourf(x_1, x_2, Z, cmap='RdBu', alpha=.5)
cs2 = ax.contour(x_1, x_2, Z, cmap='RdBu', alpha=.5)
plt.clabel(cs2, fmt = '%2.1f', colors = 'k', fontsize=14)


#ytrain = np.ndarray(y)

zero_array = []
one_array = []
for x in all_data :
	if x[2] == 0:
		zero_array.append((x[0],x[1]))
	else:
		one_array.append((x[0],x[1]))
		
zero_array = np.array(zero_array)
one_array = np.array(one_array)
		
ax.plot(zero_array[:,0], zero_array[:,1], 'ro', label='Class 1')
ax.plot(one_array[:,0], one_array[:,1], 'bo', label='Class 2')

plt.legend(loc='upper left', scatterpoints=1, numpoints=1)

plt.show()



