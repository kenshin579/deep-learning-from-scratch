# coding: utf-8
import os
import sys

print(os.pardir)
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
from dataset.mnist import load_mnist
from PIL import Image


def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()


# (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)
(x_train, t_train), (x_test, t_test) = load_mnist(flatten=False, normalize=False)
# print("x_train: ", x_train)
print("x_train.size: ", x_train.shape) # (60000, 1, 28, 28)
print("t_train.size: ", t_train.shape) # (60000,)

print("x_test.size: ", x_test.shape) # (10000, 1, 28, 28)
print("t_test.size: ", t_test.shape) # (10000,)

img = x_train[0]
label = t_train[0]
print(label)  # 5

print(img.shape)  # (784,)
img = img.reshape(28, 28)  # 형상을 원래 이미지의 크기로 변형
print(img.shape)  # (28, 28)

# img_show(img)
