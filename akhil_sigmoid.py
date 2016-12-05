#!/usr/bin/python

import matplotlib.pyplot as plt
import numpy as np
import math
import csv

def sigmoid(x):
	return (1/(1+math.exp(-x)))
	
FILE = 'admission.csv'
lines = csv.reader(open(FILE))
data = list(lines)

all_data = np.array(data)[:,[0,1,2]].astype(int)
Xaxis = []
Yaxis = []
w_0 = 0.1
w_1 = 0.02
for x in all_data:
	fx = w_0*x[0] + w_1*x[1]
	sig = sigmoid(fx)
#	sig = x[2]
#	print fx , sig
	Xaxis.append(fx)
	Yaxis.append(sig)
print Yaxis
#plt.plot(Xaxis[0:-1],Yaxis[0:-1])
plt.scatter(Xaxis[0:-1],Yaxis[0:-1])
plt.show()