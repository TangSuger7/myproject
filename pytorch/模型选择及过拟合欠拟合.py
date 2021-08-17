# -*- coding: UTF-8 -*-
import math
import numpy as np
import torch
from torch import  nn
from d2l import  torch as d2l

# 用三阶多项式生成训练和测试数据的标签
max_degree = 20
n_train, n_test = 100,100
true_w = np.zeros((max_degree))
true_w[0:4] = np.array([5,1.2,-3.4,5.6])

features = np.random.normal_(size=(n_train+n_test,1))
np.random.shuffle(features)
poly_features = np.power(features,np.arange(max_degree).reshape(-1,1))
for i in range(max_degree):
    poly_features[:,i] /= math.gamma(i+1)
labels = np.dot(poly_features,true_w)
labels += np.random.normal(scale=0.1,size=labels.shape)