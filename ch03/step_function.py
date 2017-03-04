# coding: utf-8
import numpy as np
import matplotlib.pylab as plt


def step_function_test(x):
    y = x > 0  # bool값이 저장됨
    return y.astype(np.int)  # astype() 변환함수임 int로 바꾸고 싶다


result = step_function_test(np.array([1.0, 2.0]))
print(result)


def step_function(x):
    return np.array(x > 0, dtype=np.int)  # dtype이란? integer type으로 인식하겠다


X = np.arange(-5.0, 5.0, 0.1)
Y = step_function(X)
plt.plot(X, Y)
plt.ylim(-0.1, 1.1)  # y축의 범위 지정
plt.show()

# test
# 참고: http://stackoverflow.com/questions/9457037/what-does-dtype-do
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
print(x)

# reshape
x = x.reshape((2, 5))  # x.shape = (2, 5) 같음
print(x)

x = np.array([1, 2, 3], dtype=np.int)

print('The original array')
print(x)

print('\n...Viewed as unsigned 8-bit integers (notice the length change!)')
y = x.view(np.uint8)
print(y)

print('\n...Doing the same thing by setting the dtype')
x.dtype = np.uint8
print(x)

print('\n...And we can set the dtype again and go back to the original.')
x.dtype = np.int
print(x)
