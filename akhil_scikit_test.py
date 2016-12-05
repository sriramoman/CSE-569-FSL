#!/usr/bin/python
import math
import random
import csv
import traceback
from sklearn import linear_model
import numpy as np

FILE = 'admission.csv'
lines = csv.reader(open(FILE))
data = list(lines)
norm_x = np.array(data)[:,[0,1]]
y_flip = np.array(data)[:,[2]]
logreg = linear_model.LogisticRegression()
logreg.fit(norm_x, y_flip)

print logreg.predict(norm_x)

