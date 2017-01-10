#!/usr/bin/env python3
import numpy as np

x = np.array([1.0, 2.0, 3.0])
print(x)
print(type(x))

# 넘파이의 N차원 배열
A = np.array([[1, 2], [3, 4]])
print(A)
print(A.shape)  # (2,2) <-- 2X2 이라는 의미
print(A.dtype)
B = np.array([[3, 0], [0, 6]])
print(A + B)
print(A * B)

X = np.array([[51, 55], [14, 19], [0, 4]])
print(X)
X = X.flatten()
print(X)
print(X[np.array([0, 2, 4])])  # 인덱스가 0, 2, 4인 원소를 얻음
print(X > 15)
