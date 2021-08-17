# -*- coding: UTF-8 -*-
import torch
import torchvision
from torch.utils import data
from torchvision import transforms
from d2l import  torch as d2l
from IPython import  display
class Accumulator:
    def __init__(self,n):
        self.data = [0.] *n

    def add(self,*args):
        self.data = [a+float(b) for a,b in zip(self.data,args)]
    def reset(self):
        self.data = [0.] *len(self.data)
    def __getitem__(self,idx): #该方法通过idx作为key，返回实例对象中利用该key的结果
        return self.data[idx]





x = torch.arange(12)
x.shape
print(x.numel())
x = x.reshape(3,4)
y = torch.tensor([[2,1,4,3],[1,2,3,4],[4,3,2,1]])

# x+y x-y x*y x/y
# torch.cat((x,y),dim=0)  # 合并
x.sum()
x[-1]

# 内存分配问题
before = id(y)
y = y+x  # +=操作不会分配新内存，+操作会分配新内存
y = x.clone()
print(id(y) == before)
# 数据转换
A = x.numpy()
B = torch.tensor(A)
# 将大小为1的张量转换为Python标量
# x.item() ,int(x),float(x)

# --------------------------------------------
# 数据预处理
# import os
# os.makedirs(os.path.join('..','data'),exist_ok=True)
# data_file = os.path.join('..','data','house_tiny.csv')
# with open(data_file,'w') as f:
#     f.write('NumRooms,Alley,Price\n')
#     f.write('NA,Pave,127500\n')
#     f.write('2,NA,106000\n')
#     f.write('4,NA,178100\n')
#     f.write('NA,NA,140000\n')
# -------------------------------------
# import os
# data_file = os.path.join('..','data','house_tiny.csv')
# import pandas as pd
# data = pd.read_csv(data_file)
# print(data)
# ## 处理缺失数据
# inputs, outputs = data.iloc[:,0:2],data.iloc[:,2]
# inputs = inputs.fillna(inputs.mean())  # 对于数值的用均值代替
# inputs = pd.get_dummies(inputs,dummy_na=True)
# x,y = torch.tensor(inputs.values),torch.tensor(outputs.values)

# --------------------------
# 线性代数
# A = torch.arange(20).reshape(5,4)
# X = torch.arange(24).reshape(2,3,4) #行是最后一维，列是倒数第二维，依次...
#
# A*A # 要求两个矩阵维度相同，乘法规则是哈达玛积
# A@A.T  # 矩阵乘法
# X.sum(axis=0,keepdims=True) #计算后保持轴数不变，目的是为了使用广播机制时维度相同
# # torch.dot(x,y) 相当于按元素乘法 x*y,再进行求和表示点积
# # torch.mv(A,x) 相当于矩阵向量积Ax，其中x为行向量
# # torch.mm(A,B) 矩阵相乘
# # torch.norm(u) 范数

# ----------------------------
# 矩阵运算
# 自动求导
## 链式法则
### 正向累积
##### 需要存储中间结果，计算复杂度O(n)，内存复杂度O(n)
### 反向传递
##### 去除不需要的枝，计算复杂度O(n)，内存复杂度为O(1)

# x = torch.arange(4.0)
# x.requires_grad_(True)# 设定梯度保存位置，等价于x = torch.arange(4.0,requires_grad=True)
#
# y = 2 * torch.dot(x, x)
# y.backward()
# print(x.grad==4*x)
# x.grad.zero_() # 默认情况下，pytorch会累计梯度，需要清楚之前的值
# y = x.sum()
# y.backward()
# print(x.grad)

# ----------------------------------
# 线性回归从零实现
# import random
# import torch
# from d2l import torch as d2l
# import matplotlib.pyplot as plt
# def synthetic_data(w,b,num_examples):
#     x = torch.normal(0,1,(num_examples,len(w)))
#     y = torch.matmul(x,w) + b
#     y += torch.normal(0,0.01,y.shape)
#     return x,y.reshape((-1,1))
# true_w = torch.tensor([2,-3.4])
# true_b = 4.2
# features, labels = synthetic_data(true_w,true_b,1000)
#
# # plt.scatter(features[:,1],labels)
# def data_iter(batch_size,features,labels):
#     num_examples = len(features)
#     indices = list(range(num_examples))
#     random.shuffle(indices)
#     for i in range(0,num_examples,batch_size):
#         batch_indices = torch.tensor(indices[i:min(i+batch_size,num_examples)])
#         yield features[batch_indices],labels[batch_indices]
#
# batch_size = 10
# for X,y in data_iter(batch_size,features,labels):
#     print(X,'\n',y)
#     # break
#
# w = torch.normal(0,0.01,size=(2,1),requires_grad=True)
# b = torch.zeros(1, requires_grad=True)
#
# def linreg(X,w,b): # 定义线性模型
#     return torch.matmul(X,w)+b
# def squared_loss(y_hat,y):#定义损失函数
#     return (y_hat-y.reshape(y_hat.shape))**2/2
# def sgd(params,lr,batch_size):#小批量随机梯度下降
#     with torch.no_grad(): # 更新，不参与运算
#         for param in params:
#             param -= lr * param.grad / batch_size
#             param.grad.zero_()
# lr = 0.03
# num_epochs = 3
# net = linreg
# loss = squared_loss
# for epoch in range(num_epochs): #训练过程
#     for X,y in data_iter(batch_size,features,labels):
#         l = loss(net(X,w,b),y)
#         l.sum().backward()
#         sgd([w,b],lr,batch_size)
#     with torch.no_grad():
#         train_l = loss(net(features,w,b),labels)
#         print(f'epoch {epoch+1},loss {float(train_l.mean())}')
# print(f'w的估计误差：{true_w-w.reshape(true_w.shape)}')
# print(f'b的估计误差：{true_b-b}')

# ------------------------
# 线性回归的简洁实现
# import numpy
# import torch
# from torch.utils import  data
# from d2l import  torch as d2l
#
# true_w = torch.tensor([2,-3.4])
# true_b = 4.2
# features, labels = d2l.synthetic_data(true_w, true_b, 1000)
# def load_array(data_arrays,batch_size,is_train=True):
#     dataset = data.TensorDataset(*data_arrays) # args 表示任何多个无名参数，它本质是一个 tuple，元组的变量为args
#     # **kwargs 表示关键字参数，它本质上是一个 dict,字典为kwargs
#     return data.DataLoader(dataset,batch_size,shuffle=is_train)
# batch_size=10
# data_iter = load_array((features,labels),batch_size)
# # next(iter(data_iter)) # iter(list) 生成迭代器，使用next访问，元素全部访问后再使用next会报错
#
# from torch import  nn
# net = nn.Sequential(nn.Linear(2,1))# list of layers
# net[0].weight.data.normal_(0,0.01)
# net[0].bias.data.fill_(0)
# loss = nn.MSELoss()
#
# trainer = torch.optim.SGD(net.parameters(),lr=0.03)
# num_epochs = 3
# for epoch in range(num_epochs):
#     for X,y in data_iter:
#         l = loss(net(X),y)
#         trainer.zero_grad() # 防止累加
#         l.backward()
#         trainer.step()
#     l = loss(net(features),labels)
#     print(f'epoch {epoch+1},loss {l:f}')
# ----------------------------
## 分类和回归的区别
### 回归估计一个连续值，跟真实值的区别作为损失
### 分类预测一个离散类别，输出是预测为第i类的置信度
### 从回归到多类分类，使用均方误差，最大值最为预测softmax函数
###  交叉熵常用来衡量两个概率区别，其梯度是真实概率和预测概率的区别，即l(y,yh)=-logYhy    求导后为softmax(o(i-yi
## 损失函数
### L2 Loss 均方损失
### L1 Loss 绝对值损失
### Huber's 鲁棒损失，当预测试和真实值比较近的时候为均方损失，比较远的时候为绝对值损失
# --------------------------------
# 图像分类数据集

# class Accumulator:
#     def __init__(self,n):
#         self.data = [0.] *n
#
#     def add(self,*args):
#         self.data = [a+float(b) for a,b in zip(self.data,args)]
#     def reset(self):
#         self.data = [0.] *len(self.data)
#     def __getitem__(self,idx): #该方法通过idx作为key，返回实例对象中利用该key的结果
#         return self.data[idx]
# d2l.use_svg_display()
# trans = transforms.ToTensor()
# mnist_train = torchvision.datasets.FashionMNIST(root='../data',train=True,
#                                                 transform=trans,download=True)
# mnist_test = torchvision.datasets.FashionMNIST(root='../data',train=False,
#                                                transform=trans,download=True)
# batch_size = 256
# def get_dataloader_workers():
#     return 0
# train_iter = data.DataLoader(mnist_train,batch_size,shuffle=True)
# test_iter = data.DataLoader(mnist_test,batch_size,shuffle=True)
# test_iter = data.DataLoader(mnist_test,batch_size,shuffle=True)
#
# num_inputs = 784
# num_outputs = 10
#
# w = torch.normal(0,0.01,size=(num_inputs,num_outputs),requires_grad=True)
# b = torch.zeros(num_outputs,requires_grad=True)
#
# def softmax(X):
#     X_exp = torch.exp(X)
#     # partition = X_exp.sum(-1,keepdim=True)
#     partition = X_exp.sum(1,keepdim=True)
#     return X_exp/partition
# def net(X):
#     return softmax(torch.matmul(X.reshape((-1,w.shape[0])),w)+b)
# def cross_entropy(y_hat,y):# 实现交叉熵损失函数
#     return  -torch.log(y_hat[range(len(y_hat)),y])#取出第i个样本真实结果对应标签位置处预测值的交叉熵计算结果
# def accuary(y_hat,y):#利用预测结果得到准确率
#     if len(y_hat.shape)>1 and y_hat.shape[1] >1:
#         y_hat = y_hat.argmax(axis=1)#每一行最大值，找到最大值对应的序号
#     cmp = y_hat.type(y.dtype) == y # 把y_hat转成y数据类型作比较
#     return float(cmp.type(y.dtype).sum())
# def evaluate_accuary(net,data_iter):#评估任意模型
#     if isinstance(net,torch.nn.Module):
#         net.eval()#将模型设置为评估模式
#     metric = Accumulator(2) #，预测总数
#     for X,y in data_iter:
#         metric.add(accuary(net(X),y),y.numel()) # numel 元素个数
#     return metric[0]/metric[1]
#
# # print(evaluate_accuary(net,test_iter))
# def train_epoch_ch3(net,train_iter,loss,updater):# 开始训练,一次迭代
#     if isinstance(net,torch.nn.Module):#手动的，或者是nn模型
#         net.train()
#         print('n')
#     metric = Accumulator(3)
#     for X,y in train_iter:
#         y_hat = net(X)
#         l = loss(y_hat,y)
#         if isinstance(updater,torch.optim.Optimizer):
#             updater.zero_grad() # 梯度设为0
#             l.backward() # 计算梯度
#             updater.step() # 参数自更新
#             metric.add(
#                 float(l) * len(y),accuary(y_hat,y),
#                 y.size().numel()) # 累加器参数
#         else:
#             l.sum().backward() # 自己实现的l为向量
#             updater(X.shape[0])# 更新
#             metric.add(float(l.sum()),accuary(y_hat,y),y.numel())
# #     return metric[0]/metric[2],metric[1]/metric[2] # 损失，loss累加/总样本数，以及准确样本数/总样本数
# class Animator:
#     def __init__(self,xlabel=None,ylabel=None,legend=None,xlim=None,
#                  ylim=None,xscale='linear',yscale='linear',
#                  fmts=('-','m--','g-','r:'),nrows=1,ncols=1,
#                  figsize=(3.5,2.5)):
#         if legend is None:
#             legend =[]
#         d2l.use_svg_display()
#         self.fig,self.axes = d2l.plt.subplots(nrows,ncols,figsize=figsize)
#         if nrows*ncols==1:
#             self.axes = [self.axes,]
#         self.config_axes = lambda :d2l.set_axes(self.axes[0],
#                                                 xlabel,ylabel,xlim,ylim,
#                                                 xscale,yscale,legend)
#         self.X,self.Y,self.fmts = None,None,fmts
#     def add(self,x,y):
#         if not hasattr(y,"__len__"):# 判断y是否只有一个值
#             y = [y]
#         n = len(y)
#         if not hasattr(x,"__len__"):
#             x = [x]*n  #
#         if not self.X:
#             self.X = [[] for _ in range(n)]
#         if not self.Y:
#             self.Y = [[] for _ in range(n)]
#         for i,(a,b) in enumerate(zip(x,y)):
#             if a is not None and b is not None:
#                 self.X[i].append(a)
#                 self.Y[i].append(b)
#         self.axes[0].cla()
#         for x,y,fmt in zip(self.X,self.Y,self.fmts):
#             self.axes[0].plot(x,y,fmt)
#         self.config_axes()
#         display.display(self.fig)
#         display.clear_output(wait=True)
        # display
# def train_ch3(net,train_iter,test_iter,loss,num_epochs,updater):
#     animator = Animator(xlabel='epoch',xlim=(1,num_epochs),ylim=[0.3,0.9],
#                         legend=['train loss','train acc','test acc'])
#     for epoch in range(num_epochs):
#         train_metrics = train_epoch_ch3(net,train_iter,loss,updater)
#         test_acc = evaluate_accuary(net,test_iter)
#         animator.add(epoch+1,train_metrics+(test_acc,))
#     train_loss,train_acc = train_metrics
#     assert train_loss < 0.5, train_loss
#     assert train_acc <= 1 and train_acc > 0.7, train_acc
#     assert test_acc <= 1 and test_acc > 0.7, test_acc
# lr = 0.1
# def updater(batch_size):
#     return d2l.sgd([w,b],lr,batch_size)
# num_epochs=10
# train_ch3(net,train_iter,test_iter,cross_entropy,num_epochs,updater)

#  ---------------------------------
# softmax简洁实现
import torch
from torch import nn
from d2l import torch as d2l

batch_size = 256
def predict_ch3(net,test_iter,n=6):
    for X,y in test_iter:
        break
    trues = d2l.get_fashion_mnist_labels(y)
    preds = d2l.get_fashion_mnist_labels(net(X).argmax(axis=1))
    titles = [true + '\n' + pred for true,pred in zip(trues,preds)]
    d2l.show_images(X[0:n].reshape((n,28,28)),1,n,titles=titles[0:n])
if __name__ == '__main__':
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

    net = nn.Sequential(nn.Flatten(),nn.Linear(784,10))

    def init_weights(m):
        if type(m)==nn.Linear:
            nn.init.normal_(m.weight,std=0.1)
    net.apply(init_weights)#按照每一层去初始化参数
    loss =nn.CrossEntropyLoss()
    trainer = torch.optim.SGD(net.parameters(),lr=0.1)
    num_epochs = 10
    d2l.train_epoch_ch3(net,train_iter,loss,trainer)
    predict_ch3(net,test_iter)

# Q&A
#