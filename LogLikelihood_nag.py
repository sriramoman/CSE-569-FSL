import math
import random
import csv
import traceback
import numpy as np

nfeatures=9
Wo = [0, 1, 2, 3, 2, 5, 6, 7, 8, 9,1][:nfeatures]
# Wo = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0][:nfeatures]
Wn = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0][:nfeatures]

# momentum="nag"
# momentum=None
momentum = "polyak"
filename='dataset.csv'

def split_dataset(data, train_percentage):
    train = []
    test = list(data)
    train_size = int(train_percentage * len(data))

    while len(train) <= train_size:
        i = random.randrange(len(test))
        line = test.pop(i)
        train.append(line)
    return [train, test]


def log_loss(line, W):
    fx = W[0]
    for i in range(1, len(W) - 1):
        fx += line[i] * W[i]
        # print W[i]
    result = round(math.exp(fx), 2) / round(((1 + math.exp(fx))), 2)
    return result


def predict(line, W):
    result = log_loss(line, W)
    # print result
    if result >= 0.5:
        return 1
    return 0


def sigma(train, W, i,u,v):
    sum = 0.0
    total_loss=0
    for k in range(len(W)):
        W[k] += u*v
    for line in train:
        loss = log_loss(line, W)
        total_loss+=loss
        # print loss
        if i == 0:
            sum += (line[-1] - loss)
        else:
            sum += line[i] * (line[-1] - loss)
    return sum,total_loss


def LR(train):
    W_old = Wo
    W_new = Wn
    MAX_ITR = 100
    itr = 0
    N = 0.01
    thresh = 0.01
    v=1
    u = {}
    u['polyak'] = 0
    u['nag'] = 1/100
    u[None] = 0
    feedback = u[momentum] if momentum is 'nag' else 0
    delta = np.inf
    loss0 = np.inf
    while itr<100:
        for i in range(0, len(W_old)):
            sig,loss=sigma(train, W_old, i,feedback,v)
            # print "loss:",loss
            # print "loss0:",loss0
            delta = loss0 - loss
            print "delta:",np.abs(delta)
            loss0 = loss
            v = u[momentum]*v + ( N * sig)
            W_new[i] = W_old[i]+v
        W_old = W_new
        itr += 1
    print "itr:",itr
    # print W_new
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


def run(TRAIN_SIZE):
    FILE = filename
    lines = csv.reader(open(FILE))
    data = list(lines)
    for i in range(len(data)):
        data[i] = [rep(j, data[i]) for j in range(len(data[i]))]
    # print data
    train, test = split_dataset(data, TRAIN_SIZE)
    W = LR(train)
    passed = 0
    for line in test:
        if line[-1] == predict(line, W):
            passed += 1
    print "total " + str(len(test))
    print "failed " + str(len(test) - passed)
    print "pass percentage " + str(passed / float(len(test)))
    print "###################################################"
    return 1.0 - (passed / float(len(test)))



if __name__ == '__main__':
    try:
        TRAIN_FRACTIONS = [.35]
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
