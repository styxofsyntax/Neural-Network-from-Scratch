import numpy as np


def relu(z):
    return z.clip(min=0)


def softmax(z):
    z_max = np.max(z, axis=1, keepdims=True)
    expZ = np.exp(z - z_max)
    return expZ / np.sum(expZ, axis=1, keepdims=True)


def deriv_relu(z):
    return z > 0


def deriv_softmax(z):
    s = softmax(z)
    return s * (1 - s)


activations = {
    'relu': relu,
    'softmax': softmax
}

deriv_activations = {
    'relu': deriv_relu,
    'softmax': deriv_softmax
}
