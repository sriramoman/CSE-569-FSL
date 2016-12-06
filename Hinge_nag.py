import math
import random
import csv
import traceback
import numpy as np

# momentum="nag"
momentum=None
# momentum = "polyak"
filename='rand5k.csv'
iters=0

nfeatures = 10
Wo=[1, 2, 3, 2, 5, 6, 7, 8, 9,10][:nfeatures]
Wn=[0, 0, 0, 0, 0, 0, 0, 0, 0,0][:nfeatures]
# filename='dataset.csv'


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
    for i in range(1, len(W) - 1):
        fx += line[i] * W[i]
    result = round(math.exp(fx), 2) / round(((1 + math.exp(fx))), 2)
    return result


def predict(line, W):
    if fx(line, W) >= 0.5:
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


def LR(train):
    W_old = Wo
    W_new = Wn

    MAX_ITR = 100
    itr = 0
    n = 0.01

    while itr < MAX_ITR:
        for i in range(0, len(W_old)):
            W_new[i] = W_old[i] + (n * sigma(train, W_old, i))
        W_old = W_new
        itr += 1
    return W_new


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
    # W = LR(train)
    passed = 0
    # Validate predicted outcome against
    for line in test:
        p = predict(line, W)
        if line[-1] == p:
            passed += 1
    print "Fraction:" + str(TRAIN_SIZE)
    print "iter:"+str(iters)
    print "total:" + str(len(test))
    print "failed:" + str(len(test) - passed)
    print "pass:" + str(passed*100 / float(len(test)))
    print ""
    return 1.0 - (passed / float(len(test)))


def hinge(training_data):
    all_data = np.array(training_data).astype(int)
    # print all_data.shape
    all_x = all_data[:,range(0,np.shape(all_data)[1]-1)]
    # print all_x
    y = (all_data[:,-1]).T
    return doTraning(all_x, y)


# http://stackoverflow.com/questions/3985619/how-to-calculate-a-logistic-sigmoid-function-in-python
def sigmoid(x):
    print "sigoid:",x
    f = 1 / float(1 + math.exp(-x))
    if f > 0:
        return 1
    else:
        return -1


def hinge_loss(w, x, y):
    loss, grad = 0, 0
    for (x_, y_) in zip(x, y):
        print "w :",w
        print "x_ :",x_
        x_T = np.array([x_]).transpose()
        print "x_T :",x_T
        ind = 0
        fx = 0
        # for ind in range(len(w)-1):
        #     print "w_[",ind,"]",w[ind]," | x_[",ind,"]",x_[ind]
        #     fx += w[ind]*x_[ind]
        #     print "fx_[",ind,"]",fx
        fx = np.dot(w, x_T)
        print "fx : ",fx
        sig = sigmoid(fx)
        v = y_ * sig
        loss += max(0, 1 - v)
        print "loss",loss
        g = sig*(1 - sig)
        grad += 0 if v > 1 else -y_ *x* g
    return (loss, grad)


def doTraning(x, y, thet=np.array((.1, .02, .3, .4, .5, .6, .3, .5, .2,.7)), nita=0.001, thresh=0.0001):
    grad = np.inf
    thet= np.array([thet[:nfeatures]])
    ws = np.zeros((nfeatures, 0))
    ws = np.hstack((ws, thet.reshape(nfeatures, 1)))
    ctr = 1
    delta = np.inf
    loss0 = np.inf
    theta = 1
    v = 1
    u = 0 if momentum is None else 1
    feedback = u*v if momentum is 'nag' else 0
    while np.abs(delta) > thresh:
        loss, grad = hinge_loss(thet+feedback, x, y)
        # print loss
        delta = loss0 - loss
        loss0 = loss
        # if np.linalg.norm(grad) == 0:
        #     break
        nabla = grad / np.linalg.norm(grad)
        v = u*v - nita*grad
        thet = thet + v
        ctr += 1
        print "iters",ctr
    ws = np.hstack((ws, thet.reshape((nfeatures, 1))))
    global iters
    iters=ctr
#    print ws
    return np.sum(ws, 1) / np.size(ws, 1)


if __name__ == '__main__':
    try:
        TRAIN_FRACTIONS = [.7]#, .62, .53, .625, 0.99]
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
