# coding: utf-8

# In[1]:


import os
import time

import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import normalize


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


if __name__ == '__main__':
    x_train, y_train, x_test, y_test = load_data()

# In[2]:


# 对训练集预测并输出结果
tic = time.clock()
gnb = GaussianNB()
y_pred = gnb.fit(x_train, y_train).predict(x_train)
toc = time.clock()
print("Number of mislabeled points out of a total %d points : %d"
      % (y_train.shape[0], (y_train != y_pred).sum()))

a = (1 - (y_train != y_pred).sum() / x_train.shape[0])
print("The accuracy rate for train set data is : %f" % a)
print("The time used is : %f" % (toc - tic))

# In[3]:


# 对测试集集预测并输出结果
tic = time.clock()
gnb = GaussianNB()
y_pred = gnb.fit(x_train, y_train).predict(x_test)
toc = time.clock()
print("Number of mislabeled points out of a total %d points : %d"
      % (y_test.shape[0], (y_test != y_pred).sum()))
a = (1 - (y_test != y_pred).sum() / x_test.shape[0])
print("The accuracy rate for train set data is : %f" % a)
print("The time used is : %f" % (toc - tic))

# In[4]:


np.save('NB_Gaussian_pred', y_pred)

# In[5]:


from sklearn.naive_bayes import BernoulliNB

# 对测试集集预测并输出结果
tic = time.clock()
gnb = BernoulliNB()
y_pred = gnb.fit(x_train, y_train).predict(x_test)
toc = time.clock()
print("Number of mislabeled points out of a total %d points : %d"
      % (y_test.shape[0], (y_test != y_pred).sum()))
a = (1 - (y_test != y_pred).sum() / x_test.shape[0])
print("The accuracy rate for train set data is : %f" % a)
print("The time used is : %f" % (toc - tic))
np.save('NB_BernoulliNB_pred', y_pred)

# In[6]:


np.save('y_test', y_test)
