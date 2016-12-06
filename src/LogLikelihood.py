import math
import csv
import traceback
import random
import sys

try:
    momentum = sys.argv[1]
    print momentum
except:
    momentum = "nag"

nfeatures = 9
Wo = [0, 1, 2, 3, 2, 5, 6, 7, 8, 9, 1][:nfeatures]
Wn = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0][:nfeatures]
itr = 0

predicted_thresh = 0.5
filename = 'src/dataset.csv'

def read_csv(data, train_percentage):
    #referred http://www.pythonforbeginners.com/systems-programming/using-the-csv-module-in-python/
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
    # referred http://stackoverflow.com/questions/3985619/how-to-calculate-a-logistic-sigmoid-function-in-python
    result = round(math.exp(fx), 2) / round(((1 + math.exp(fx))), 2)
    return result

def test_accuracy(line, W):
    result = log_loss(line, W)
    if result >= predicted_thresh:
        return 1
    return 0

def sigma(train, W, i, u, v):
    sum = 0.0
    total_loss = 0
    for k in range(len(W)):
        W[k] += u * v
    for line in train:
        loss = log_loss(line, W)
        total_loss += loss
        if i == 0:
            sum += (line[-1] - loss)
        else:
            sum += line[i] * (line[-1] - loss)
    return sum, total_loss

def log_likelihood(train):
    W_old = Wo
    W_new = Wn
    global itr
    itr = 0
    N = 0.01
    v = 1
    u = {}
    u['polyak'] = 0
    u['nag'] = .01
    u[None] = 0
    u["none"] = 0
    feedback = u[momentum] if momentum is 'nag' else 0
    while itr < 100:
        for i in range(0, len(W_old)):
            #referred https://github.com/ragavvenkatesan/yann/tree/master/yann/core
            sig, loss = sigma(train, W_old, i, feedback, v)
            v = u[momentum] * v + (N * sig)
            W_new[i] = W_old[i] + v
        W_old = W_new
        itr += 1
    return W_new


def data_lookup(i, data):
    if i == (len(data) - 1):
        if data[i] == '2':
            return 1
        else:
            return 0
    x = data[i]
    if x == '?' or x == '':
        return 0
    return int(x)

def train(training_portion):
    FILE = filename
    lines = csv.reader(open(FILE))
    data = list(lines)
    for i in range(len(data)):
        data[i] = [data_lookup(j, data[i]) for j in range(len(data[i]))]
    train, test = read_csv(data, training_portion)
    weights = log_likelihood(train)
    passed = 0
    for entry in test:
        if entry[-1] == test_accuracy(entry, weights):
            passed += 1
    print "Loss type: Log Likelihood"
    print "Training Fraction: " + str(training_portion)
    print "Iterations: " + str(itr)
    print "Total Data Tested: " + str(len(test))
    print "Failed: " + str(len(test) - passed)
    print "Pass: " + str(passed * 100 / float(len(test))), "%"
    print "Momentum: ", str(momentum)
    return 1.0 - (passed / float(len(test)))

def exec_loglikelihood():
    try:
        train_component = [.35]
        average = 1
        accuracy_list = [0.0] * len(train_component)
        for j in range(len(train_component)):
            accuracy = 0.0
            for i in range(average):
                accuracy += train(train_component[j])
            accuracy /= average
            accuracy_list[j] = accuracy
    except Exception as e:
        print e
        traceback.print_exc()

exec_loglikelihood()
