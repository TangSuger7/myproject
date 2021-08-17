# -*- coding: UTF-8 -*-
import torch
from torch import nn
from torchvision import datasets,transforms
from d2l import torch as d2l
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
batch_size = 60000
torch.cuda.empty_cache()
print(torch.cuda.is_available())  # 查看cuda是否可用

print(torch.cuda.device_count())  # 返回GPU数目
print(torch.cuda.get_device_name(0))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if __name__ == '__main__':
    train_iter = torch.utils.data.DataLoader(
        datasets.FashionMNIST('../data', train=True, download=False,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch_size, shuffle=True,num_workers=0,pin_memory=True)
    test_iter = torch.utils.data.DataLoader(
        datasets.FashionMNIST('../data', train=False, download=False,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch_size, shuffle=True,num_workers=0,pin_memory=True)
    # train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    num_inputs, num_outputs, num_hiddens = 784, 10, 256
    W1 = nn.Parameter(torch.randn(num_inputs, num_hiddens, requires_grad=True))
    b1 = nn.Parameter(torch.zeros(num_hiddens, requires_grad=True))
    W2 = nn.Parameter(torch.randn(num_hiddens, num_outputs, requires_grad=True))
    b2 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))
    params = [W1, b1, W2, b2]


    def relu(X):
        return torch.max(X, torch.zeros_like(X))
    def net(X):
        X = X.reshape((-1, num_inputs))
        H = relu(X @ W1 + b1)
        return (H @ W2 + b2)
    num_epochs, lr = 10, 0.1
    loss = nn.CrossEntropyLoss()
    updater = torch.optim.SGD(params, lr=lr)
    d2l.train_ch3(net,train_iter,test_iter,loss,num_epochs,updater)
    d2l.predict_ch3(net,test_iter,6)
