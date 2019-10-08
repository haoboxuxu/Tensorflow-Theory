import numpy as np


def sigmoid(x, deriv=False):
    if deriv == True:
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))


X = np.array([[0, 0, 1],
              [0, 1, 1],
              [1, 0, 1],
              [1, 1, 1]])
# print(X.shape)
# (4, 3)

y = np.array([[0],
              [1],
              [1],
              [0]])


# print(y.shape)
# (4, 1)

np.random.seed(1)

# w0 = np.random.random((3, 4))
# w1 = np.random.random((4, 1))
# print(w0)
# print(w1)
# print(w0.shape)
# print(w1.shape)

w0 = 2 * np.random.random((3, 4)) - 1
w1 = 2 * np.random.random((4, 1)) - 1
# print(w0)
# print(w1)
# print(w0.shape)
# print(w1.shape)

for j in range(50000):
    l0 = X
    l1 = sigmoid(np.dot(l0, w0))
    l2 = sigmoid(np.dot(l1, w1))

    l2_error = y - l2

    if (j % 100) == 0:
        print("err = " + str(np.mean(np.abs(l2_error))))

    l2_delta = l2_error * sigmoid(l2, True)

    l1_error = l2_delta.dot(w1.T)
    l1_delta = l1_error * sigmoid(l1, True)

    w1 += l1.T.dot(l2_delta)
    w0 += l0.T.dot(l1_delta)
