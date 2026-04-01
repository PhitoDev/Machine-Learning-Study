import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-np.asarray(x)))


def softmax(x):
    x = np.asarray(x)
    max_a = np.max(x)
    axis = None if x.ndim < 2 else 1

    dividend = np.exp(x - max_a)
    divisor = np.sum(np.exp(x - max_a), axis=axis, keepdims=True)

    return dividend / divisor


def tanh(x):
    x = np.asarray(x)
    e_x = np.exp(x)
    e_nx = np.exp(-x)

    return (e_x - e_nx) / (e_x + e_nx)


def elu(x, alpha):
    x = np.asarray(x)
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))


def relu(x):
    x = np.asarray(x)
    return np.maximum(0, x)
