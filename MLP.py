import os
import random

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as Data
from sklearn import preprocessing
from sklearn.preprocessing import normalize
from torch.autograd import Variable


# The ground truth of our dataset
def train_data():
    labels = ((4, 1, 3, 2, 0, 4, 1, 3, 2, 0, 4, 1, 3, 2, 0),
              (2, 1, 3, 0, 4, 4, 0, 3, 2, 1, 3, 4, 1, 2, 0),
              (2, 1, 3, 0, 4, 4, 0, 3, 2, 1, 3, 4, 1, 2, 0))

    for dir in range(1, 11):
        for i in range(3):
            for j in range(10):
                yield np.load(os.path.join('data', str(dir), str(i), 'DE_' + str(j) + '.npy')), labels[i][j]


def test_data():
    labels = ((4, 1, 3, 2, 0, 4, 1, 3, 2, 0, 4, 1, 3, 2, 0),
              (2, 1, 3, 0, 4, 4, 0, 3, 2, 1, 3, 4, 1, 2, 0),
              (2, 1, 3, 0, 4, 4, 0, 3, 2, 1, 3, 4, 1, 2, 0))

    for dir in range(1, 11):
        for i in range(3):
            for j in range(10, 15):
                yield np.load(os.path.join('data', str(dir), str(i), 'DE_' + str(j) + '.npy')), labels[i][j]


def reshape(data):
    for i in range(data.shape[2]):
        yield data[:, :, i].reshape(310)


def load_data():
    x_train = []
    y_train = []

    for (data, l) in train_data():
        for d in reshape(data):
            x_train.append(d)
            y_train.append(l)

    x_train = normalize(np.array(x_train))
    y_train = np.array(y_train)

    x_test = []
    y_test = []

    for (data, l) in test_data():
        for d in reshape(data):
            x_test.append(d)
            y_test.append(l)

    x_test = normalize(np.array(x_test))
    y_test = np.array(y_test)

    print('Successfully loaded')
    return x_train, y_train, x_test, y_test

#固定随机数种子，以保证结果可重复
torch.manual_seed(2019)
random.seed(2019)
np.random.seed(2019)
class Mydataset(Data.Dataset):
    def __init__(self, X, Y, transform = None):
        self.label = Y
        self.transform = transform
        if transform:
            self.data = transform(X)
        else:
            self.data = X
        self.len = self.data.shape[0]
    def __len__(self):
        return self.len
    def __getitem__(self, index):        #返回特征和标签的元组
        data = self.data[index,:]
        data_as_tensor = torch.from_numpy(data).float()             #特征必须是float类型的
        # print(self.label.shape)
        label = self.label[index]
        # label_as_tensor = torch.from_numpy(label)            #特征是否需要转换成张量
        return data_as_tensor,label

class Preprocess(object):
    def __call__(self,sample):
        return preprocessing.scale(sample)
x_train, y_train, x_test, y_test = load_data()
pre = Preprocess()
train_dataset = Mydataset(x_train,y_train,transform = pre)
test_dataset = Mydataset(x_test,y_test,transform = pre)
# train_dataset = Mydataset(x_train,y_train)
# test_dataset = Mydataset(x_test,y_test)
#pytorch只能以batch方式载入数据，有时还需要打乱数据，这些可以用dataloader函数完成
train_loader = Data.DataLoader(train_dataset,batch_size=5,shuffle=True)
test_loader = Data.DataLoader(test_dataset,batch_size=5,shuffle=True)
# transform = Mydataset([1,2],[3,4],3)
# print(type(transform))
# if(transform):
#     print("哈哈")
class Net(torch.nn.Module):
    def __init__(self,n_feature,n_hidden,n_output):
        super(Net,self).__init__()
        self.hidden = torch.nn.Linear(n_feature,n_hidden)  #隐藏层线性输出
        self.out = torch.nn.Linear(n_hidden,n_output)    #输出层线性输出

    def forward(self,x):
        x = F.sigmoid(self.hidden(x))
        x = F.sigmoid(self.out(x))                              #是否要有激活函数
        return x
#
net = Net(n_feature=310,n_hidden=128,n_output = 5)
# #print(net)
# #optimizer是训练的工具
optimizer = torch.optim.SGD(net.parameters(),lr=0.02)
# optimizer = torch.optim.Adam(net.parameters(),lr = 0.02)
loss_func = torch.nn.CrossEntropyLoss()
for epoch in range(200):
    for i,data in enumerate(train_loader):
        inputs, labels = data
        inputs = Variable(inputs)
        labels = Variable(labels).long()            #Label必须要是Long类型的
        outputs = net(inputs)
        loss = loss_func(outputs,labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

correct = 0
total = 0
pred = np.array([])
testlabel = np.array([])
for data in test_loader:
    features,labels = data
    labels = labels.long()    #Label必须要是Long类型的
    outputs = net(Variable(features))
    _, predicted = torch.max(outputs.data, 1)
    temp = predicted.numpy()
    pred = np.append(pred,temp)
    total += labels.size(0)
    temp = labels.numpy()
    testlabel = np.append(testlabel,temp)
    correct += (predicted == labels).sum()
np.save("test",testlabel)
np.save("pred",pred)
print("准确率为: %f%%" %(100 * float(correct)/float(total)))

