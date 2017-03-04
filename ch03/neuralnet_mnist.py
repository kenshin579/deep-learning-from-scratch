#!/usr/bin/env python3
# coding: utf-8
import sys, os

sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
import pickle
from dataset.mnist import load_mnist
from common.functions import sigmoid, softmax


def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test


def init_network():
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
    return network


def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)

    return y


x, t = get_data()  #
print("x.shape", x.shape)
print("t.shape", t.shape)
print("x[0].shape", x[0].shape)

network = init_network()

print("network['W1'].shape", network['W1'].shape)
print("network['W2'].shape", network['W2'].shape)
print("network['W3'].shape", network['W3'].shape)

print("network['b1'].shape", network['b1'].shape)
print("network['b2'].shape", network['b2'].shape)
print("network['b3'].shape", network['b3'].shape)

accuracy_cnt = 0
print("len(x): ", len(x))

y = predict(network, x[0])
print("y", y)

for i in range(len(x)):
    y = predict(network, x[i])
    p = np.argmax(y)  # 확률이 가장 높은 원소의 인덱스를 얻는다.
    if p == t[i]:
        accuracy_cnt += 1

print("Accuracy:" + str(float(accuracy_cnt) / len(x)))
