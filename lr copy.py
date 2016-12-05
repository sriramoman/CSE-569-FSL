import sys
import math
import random
import csv
import traceback
import numpy as np

def split_dataset(data, train_percentage):
	train = []
	test = list(data)
	train_size = int(train_percentage * len(data))
	
	while len(train) <= train_size:
		i = random.randrange(len(test))
		line = test.pop(i)
		train.append(line)
	return [train, test]

def W_random():
	W = [0] * 10
	for i in range(len(W)):
		W[i] = int(random.randrange(30))
	return W

def fx(line, W):
	fx = W[0]
	for i in range(1, len(W)-1):
		fx += line[i] * W[i]
	result = round(math.exp(fx), 2) / round(((1 + math.exp(fx))), 2)
	return result

def predict(line, W):
	result = fx(line, W)
	if result >= 0.5:
		return 1
	return 0

def sigma(train, W, i):
	sum = 0.0
	for line in train:
		if i == 0:
			sum += (line[-1] - fx(line, W))
		else:
			sum += line[i] * (line[-1] - fx(line, W))
	return sum

def hinge_loss(w, x, y):
    loss, grad = 0, 0
    for (x_, y_) in zip(x, y):
        v = y_ * np.dot(w, x_)
        loss += max(0, 1 - v)
        grad += 0 if v > 1 else -y_ * x_
        # print(grad)
    return (loss, grad)

def LR(train):
	# initialize randomly
	#   	[0,1,2,3,4,5,6,7,8,9]
	# W_old = W_random()
	# W_new = [0] * 11

	W_old = [0,1,2,3,2,5,6,7,8,9]
	W_new = [0,0,0,0,0,0,0,0,0,0]
	
	MAX_ITR = 100
	itr = 0
	N=0.01

	while itr < MAX_ITR:
		for i in range(0, len(W_old)):
			W_new[i] = W_old[i] + ( N * sigma(train, W_old, i))
		W_old = W_new
		# if itr > 90:
			# print W_new
		itr += 1
	return W_new

def rep(i, data):
	if i == (len(data) - 1):
		if data[i] == '2':
			return 1
		else:
			return 0
	x = data[i]
	if x == '?' or x == '':
		return 0
	return int(x)

def run(TRAIN_SIZE=.7):
	FILE = 'dataset.csv'
	lines = csv.reader(open(FILE))
	data = list(lines)

	for i in range(len(data)):
		data[i] = [rep(j, data[i]) for j in range(len(data[i]))]

	# print data
	train, test = split_dataset(data, TRAIN_SIZE)
	# trainSet = 500
	# train = data[:trainSet]
	# test = data[trainSet+1:]
	total = len(train)
	# print "train " + str(len(train))
	# print "test " + str(len(test))

	# Model W on train set
	
	# if sys.argv[1] == '1':
	W = hinge(train)
	# else:
	# W = LR(train)
	# print W

	# test on test dataset
	passed = 0
	for line in test:
		if line[-1] == predict(line, W):
			passed += 1

	print "total " + str(len(test))
	print "failed " + str(len(test) - passed)
	print "pass percentage " + str(passed / float(len(test)))
	print "###################################################"
	return 1.0 - (passed / float(len(test)))

def hinge(train):
	x1 = []
	x2 = []
	x3 = []
	x4 = []
	x5 = []
	x6 = []
	x7 = []
	x8 = []
	x9 = []
	x10 = []
	y = []

	for line in train:
		x1.append(line[0])
		x2.append(line[1])
		x3.append(line[2])
		x4.append(line[3])
		x5.append(line[4])
		x6.append(line[5])
		x7.append(line[6])
		x8.append(line[7])
		x9.append(line[8])
		x10.append(line[9])
		y.append(line[10])

	x = np.vstack((x1, x2, x3, x4, x5, x6, x7, x8, x9, x10)).T
	return test1(x, y)


def hinge_loss(w, x, y):
	""" evaluates hinge loss and its gradient at w

	rows of x are data points
	y is a vector of labels
	"""
	loss, grad = 0, 0
	for (x_, y_) in zip(x, y):
		v = y_ * np.dot(w, x_)
		loss += max(0, 1 - v)
		grad += 0 if v > 1 else -y_ * x_
		# print(grad)
	return (loss, grad)


def grad_descent(x, y, w=np.array((0, 0, 0, 0, 0, 0, 0, 0, 0, 0)), step=0.1, thresh=0.001):
	# print(x,y,w,step,thresh)
	grad = np.inf
	ws = np.zeros((10, 0))
	ws = np.hstack((ws, w.reshape(10, 1)))
	step_num = 1
	delta = np.inf
	loss0 = np.inf
	while np.abs(delta) > thresh:
		loss, grad = hinge_loss(w, x, y)
		delta = loss0 - loss
		loss0 = loss
		if np.linalg.norm(grad) == 0:
			break
		grad_dir = grad / np.linalg.norm(grad)
		w = w - step * grad_dir / step_num
		ws = np.hstack((ws, w.reshape((10, 1))))
		step_num += 1
	return np.sum(ws, 1) / np.size(ws, 1)


def test1(x, y):
	w = grad_descent(x, y)
	# loss, grad = hinge_loss(w, x, y)
	# print(loss,grad)
	return w


if __name__ == '__main__':
	try:		
		TRAIN_FRACTIONS = [.75]#, .02, .03, .125, .625, 0.99]
		avg = 1
		accuracy_list = [0.0] * len(TRAIN_FRACTIONS)
		for j in range(len(TRAIN_FRACTIONS)):
			accuracy = 0.0
			for i in range(avg):
				accuracy += run(TRAIN_FRACTIONS[j])
			accuracy /= avg
			accuracy_list[j] = accuracy
		# test1()

	except Exception as e:
		print e
		traceback.print_exc()
