import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time

path = 'LogiReg_data.txt'
pdData = pd.read_csv(path, header=None, names=['Exam 1', 'Exam 2', 'Admitted'])

positive = pdData[pdData['Admitted'] == 1]
negative = pdData[pdData['Admitted'] == 0]


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def model(X, theta):
    return sigmoid(np.dot(X, theta.T))


pdData.insert(0, 'Ones', 1)
orig_data = pdData.as_matrix()

cols = orig_data.shape[1]

X = orig_data[:, 0:cols - 1]
y = orig_data[:, cols-1:cols]

theta = np.zeros([1, 3])

# print(str(X.shape) + str(y.shape) + str(theta.shape))

# cost fun
def cost_fun(X, y, theta):
    left = np.multiply(-y, np.log(model(X, theta)))
    right = np.multiply(1-y, np.log(1 - model(X, theta)))
    return np.sum(left - right) / (len(X))

def gradient(X, y, theta):
    grad = np.zeros(theta.shape)
    error = (model(X, theta) - y).ravel()
    for j in range(len(theta.ravel())):
        term = np.multiply(error, X[:,j])
        grad[0, j] = np.sum(term) / len(X)
    return grad

STOP_ITER = 0
STOP_COST = 1
STOP_GRAD = 2

def stop(type, value, threshold):
    if type == STOP_ITER:
        return  value > threshold
    if type == STOP_COST:
        return abs(value[-1] - value[-2]) < threshold
    if type == STOP_GRAD:
        return np.linalg.norm(value) < threshold

# random wash data
def shuffleData(data):
    np.random.shuffle(data)
    cols = data.shape[1]
    X = data[:, 0:cols-1]
    y = data[:, cols-1:]
    return X, y

def descent(data, theta, batchSize, stopType, thresh, alpha):
    start_time = time.time()
    # epoch
    i = 0
    # batch
    k = 0
    X, y = shuffleData(data)
    grad = np.zeros(theta.shape)
    cost = [cost_fun(X, y, theta)]

    while True:
        grad = gradient(X[k:k+batchSize], y[k:k+batchSize], theta)
        k += batchSize
        if k >= len(X):
            k = 0
            X, y = shuffleData(data)
        theta -= alpha * grad
        cost.append(cost_fun(X, y, theta))#?
        i += 1
        if stopType == STOP_ITER:
            value = i
        elif stopType == STOP_COST:
            value = cost
        elif stopType == STOP_GRAD:
            value = grad
        if stop(stopType, value, thresh): break

    end_time = time.time()
    return theta, i - 1, cost, grad, end_time - start_time


def run(data, theta, batchSize, stopType, thresh, alpha):
    theta, i, cost, grad, time = descent(data, theta, batchSize, stopType, thresh, alpha)
    name = "Original" if (data[:, 1] > 2).sum() > 1 else "Scaled"
    name += " data - learning rate: {} - ".format(alpha)
    if batchSize == n:
        strDescType = "Gradient"
    elif batchSize == 1:
        strDescType = "Stochastic"
    else:
        strDescType = "Mini-batch ({})".format(batchSize)
    name += strDescType + " descent - Stop: "
    if stopType == STOP_ITER:
        strStop = "{} iterations".format(thresh)
    elif stopType == STOP_COST:
        strStop = "costs change < {}".format(thresh)
    else:
        strStop = "gradient norm < {}".format(thresh)
    name += strStop
    print("***{}\nTheta: {} - Iter: {} - Last cost: {:03.2f} - Duration: {:03.2f}s".format(
        name, theta, i, cost[-1], time))
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(np.arange(len(cost)), cost, 'r')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Cost')
    ax.set_title(name.upper() + ' - Error vs. Iteration')
    return theta

n=100
run(orig_data, theta, n, STOP_ITER, thresh=5000, alpha=0.000001)



