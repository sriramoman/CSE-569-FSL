import math
import random
import csv
import traceback
import numpy as np

momentum="nag"
# momentum=None
# momentum = "polyak"
filename='rand.csv'
iters=0

nfeatures = 10
Wo=[1, 2, 3, 2, 5, 6, 7, 8, 9,10][:nfeatures]
Wn=[0, 0, 0, 0, 0, 0, 0, 0, 0,0][:nfeatures]
predicted_thresh = 0.5


def split_dataset(data, train_percentage):
    train = []
    test = list(data)
    train_size = int(train_percentage * len(data))

    while len(train) <= train_size:
        i = random.randrange(len(test))
        line = test.pop(i)
        train.append(line)
    return [train, test]


def fx(line, W):
    fx = W[0]
    for i in range(1, len(W)):
        line_i = line[i]
        w_i = W[i]
        fx += line[i] * W[i]
    return sigmoid(fx)


def predict(line, W):
    predcited_value = fx(line,W)
    expected_value = line[-1] if line[-1]==1 else 0
    predicted_difference = np.abs(np.abs(expected_value)-predcited_value)
    if predicted_difference <= predicted_thresh:
        return 1
    else:
        return 0


def sigma(train, W, i):
    sum = 0.0
    for line in train:
        if i == 0:
            sum += (line[-1] - fx(line, W))
        else:
            sum += line[i] * (line[-1] - fx(line, W))
    return sum

def rep(i, data):
    if i == (len(data) - 1):
        if data[i] == '2' or data[i] == '1':
            return 1
        else:
            return -1
    x = data[i]
    if x == '?' or x == '':
        return 0
    return int(x)


def run(TRAIN_SIZE):
    FILE = filename
    lines = csv.reader(open(FILE))
    data = list(lines)
    global nfeatures,Wo,Wn
    nfeatures = len(data[0]) - 1
    Wo=[Wo[:nfeatures]]
    Wn=[Wn[:nfeatures]]
    for i in range(len(data)):
        data[i] = [rep(j, data[i]) for j in range(len(data[i]))]
    # print data
    train, test = split_dataset(data, TRAIN_SIZE)
    # compute the hinge loss and assign it to W
    W = hinge(train)
    passed = 0
    # Validate predicted outcome against
    for line in test:
        passed += predict(line, W)
    print "Loss type: Hinge"
    print "Training Fraction: " + str(TRAIN_SIZE)
    print "Iterations: "+str(iters)
    print "Total Data Tested: " + str(len(test))
    print "Failed: " + str(len(test) - passed)
    print "Pass: " + str(passed*100 / float(len(test))), "%"
    print "Momentum: ",str(momentum)
    return 1.0 - (passed / float(len(test)))


def hinge(training_data):
    all_data = np.array(training_data).astype(int)
    all_x = all_data[:,range(0,np.shape(all_data)[1]-1)]
    y = (all_data[:,-1]).T
    return doTraning(all_x, y)


# http://stackoverflow.com/questions/3985619/how-to-calculate-a-logistic-sigmoid-function-in-python
def sigmoid(x):
    f = 1 / float(1 + math.exp(-x))
    return f


def hinge_loss(w, x, y):
    loss=0
    grad=np.zeros((1,nfeatures))
    for (x_, y_) in zip(x, y):
        x_T = np.array([x_]).transpose()
        fx = np.dot(w, x_T)
        sig = sigmoid(fx)
        v = y_ * sig
        loss += max(0, 1 - v)
        g = sig*(1 - sig)
        zero_vectors =  np.zeros((1,nfeatures))
        grad += (zero_vectors) if v > 1 else (-y_ *x_* g)
    return (loss, grad)


def doTraning(x, y, thet=np.array((.1, .02, .3, .4, .5, .6, .3, .5, .2,.7)), nita=0.001, thresh=0.0001):
    grad = np.inf
    thet= np.array([thet[:nfeatures]])
    ws = np.zeros((nfeatures, 0))
    ws = np.hstack((ws, thet.reshape(nfeatures, 1)))
    ctr = 1
    delta = np.inf
    loss_prev = np.inf
    v = np.ones((1,nfeatures))
    u = 0 if momentum is None else 1
    while np.abs(delta) > thresh:
        feedback = u*v if momentum is 'nag' else 0
        loss, grad = hinge_loss(thet+feedback, x, y)
        delta = loss_prev - loss
        loss_prev = loss
        v = u*v - nita*grad
        thet = thet + v
        ctr += 1
    ws = thet[0]
    global iters
    iters=ctr
    print "Final Weight Vector:",ws
    return ws


if __name__ == '__main__':
    try:
        TRAIN_FRACTIONS = [.62]#, .62, .53, .625, 0.99]
        avg = 1
        accuracy_list = [0.0] * len(TRAIN_FRACTIONS)
        for j in range(len(TRAIN_FRACTIONS)):
            accuracy = 0.0
            for i in range(avg):
                accuracy += run(TRAIN_FRACTIONS[j])
            accuracy /= avg
            accuracy_list[j] = accuracy
    except Exception as e:
        print e
        traceback.print_exc()
