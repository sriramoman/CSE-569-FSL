import math
import random
import csv
import traceback
import numpy as np

nfeatures = 3

momentum="nag"
# momentum=None
# momentum = "polyak"
dataset = 'admission.csv'


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
        return -1


def sigma(train, W, i):
    sum = 0.0
    for line in train:
        if i == 0:
            sum += (line[-1] - fx(line, W))
        else:
            sum += line[i] * (line[-1] - fx(line, W))
    return sum


def LR(train):
    W_old = [1, 2, 3]
    W_new = [0, 0, 0]

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
        if data[i] == '2':
            return 1
        else:
            return -1
    x = data[i]
    if x == '?' or x == '':
        return 0
    return int(x)


def run(TRAIN_SIZE=.7):
    FILE = dataset
    lines = csv.reader(open(FILE))
    data = list(lines)
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
    print "total " + str(len(test))
    print "failed " + str(len(test) - passed)
    print "pass percentage " + str(passed / float(len(test)))
    print "###################################################"
    return 1.0 - (passed / float(len(test)))


def hinge(training_data):
    x1 = []
    x2 = []
    x3 = []
    y = []

    for parameter in training_data:
        x1.append(parameter[0])
        x2.append(parameter[1])
        x3.append(parameter[2])
        y.append(parameter[3])
    x = np.vstack((x1, x2, x3)).T
    return doTraning(x, y)


# http://stackoverflow.com/questions/3985619/how-to-calculate-a-logistic-sigmoid-function-in-python
def sigmoid(x):
    f = 1 / float(1 + math.exp(-x))
    if f > 0:
        return 1
    else:
        return -1


def hinge_loss(w, x, y):
    loss, grad = 0, 0
    for (x_, y_) in zip(x, y):
        fx = np.dot(w, x_)
        sig = sigmoid(fx)
        v = y_ * sig
        loss += max(0, 1 - v)
        # g = sigmoid(fx)*(1 - sigmoid(fx))
        grad += 0 if v > 1 else -y_ * x_
    return (loss, grad)


def doTraning(x, y, thet=np.array((.1, .02, .3)), nita=0.1, thresh=0.1):
    grad = np.inf
    ws = np.zeros((nfeatures, 0))
    ws = np.hstack((ws, thet.reshape(nfeatures, 1)))
    ctr = 1
    delta = np.inf
    loss0 = np.inf
    theta = 1
    v = 1
    u = 1
    if momentum == None:
        u = 0
    while np.abs(delta) > thresh:
        feedback = {}
        feedback['polyak'] = 0
        feedback['nag'] = u * v
        feedback[None] = 0
        loss, grad = hinge_loss(thet + u * v, x, y)
        print loss
        delta = loss0 - loss
        loss0 = loss
        if np.linalg.norm(grad) == 0:
            break
        nabla = grad / np.linalg.norm(grad)
        v = u * v - nita * nabla
        thet = thet + v
        ws = np.hstack((ws, thet.reshape((nfeatures, 1))))
        ctr += 1

    print "Hinge loss with " + momentum + " took " + str(ctr) + " iterations."
    print "<weight>"+str(ws)+"</weight>"
    return np.sum(ws, 1) / np.size(ws, 1)


if __name__ == '__main__':
    try:
        TRAIN_FRACTIONS = [.75]  # , .02, .03, .125, .625, 0.99]
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