import numpy as np


def relu(z):
    return z.clip(min=0)


def softmax(z):
    expZ = np.exp(z)
    return expZ / np.sum(expZ, axis=1, keepdims=True)


activations = {
    'relu': relu,
    'softmax': softmax
}
