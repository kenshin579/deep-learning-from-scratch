#!/usr/bin/env python3
import numpy as np

from common.functions import softmax, sigmoid, identity_function


def init_network():
    network = {}
    network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    network['b1'] = np.array([0.1, 0.2, 0.3])
    network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    network['b2'] = np.array([0.1, 0.2])
    network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
    network['b3'] = np.array([0.1, 0.2])
    return network

def forward(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']


    a1 = np.dot(x, W1) + b1     # X: 입력층; A1의 결과값
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2    # A2의 결과값
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3    # A3의 겺과값
    y = identity_function(a3)   # Y: 출력층

    return y

network = init_network()
print("network['W1'].shape", network['W1'].shape)
print("network['W2'].shape", network['W2'].shape)
print("network['W3'].shape", network['W3'].shape)

print("network['b1'].shape", network['b1'].shape)
print("network['b2'].shape", network['b2'].shape)
print("network['b3'].shape", network['b3'].shape)

x = np.array([1.0, 0.5])
print("x", x.shape)
y = forward(network, x)
print(y)