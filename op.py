import numpy as np
from variable import *
from utils.im2col import im2col

class Op(object):
    '''
    ----------- |                                                         | -------------
         Z_n	|      Op(Wn1,Z_n)        Op(Wn2,Hn1)       Op(Wn3,Hn2)   |   a_{n+1}
                |  ----------------> Hn1 ------------> Hn2 ------------>  |
       dF/dz_n	|  <------------- dF/dHn1 <--------- dF/dHn2 <----------- |   dF/a_{n+1}
    -----------	|                                                         | -------------
        Layer_n                                                              Layer_{n+1}

    X : [samples, features], Op or Layer
    '''
    def __init__(self, operator, X1 = None, X2 = None):
        self.X1 = X1
        self.X2 = X2
        self.operator = operator


    def forward(self):
        if self.X2 != None:
            self.H = self.operator.forward(self.X1, self.X2)
        else:
            self.H = self.operator.forward(self.X1)
        self.output = self.H
        self.value = self.H.value
        print('output:{}'.format(self.value.shape))

    def backward(self, nextop):
        self.nextop = nextop
        if self.X2 != None:
            self.X1.D, self.X2.D = self.operator.backward(self.nextop)
            self.X1.D, self.X2.D = np.mean(self.X1.D, axis=0), np.mean(self.X2.D, axis=0)
            print('dX1:{}, dx2.D:{}'.format(self.X1.D.shape, self.X2.D.shape))
        else:
            self.X1.D = self.operator.backward(self.nextop)
            self.X1.D = np.mean(self.X1.D, axis=0)
            # self.X1.D = self.dX1 * self.nextop.D

class Add(object):
    def __init__(self):
        pass
    def forward(self, X1, X2):
        self.X1, self.X2 = X1, X2
        return Variable(self.X1.value + self.X2.value, lr=0)
    def backward(self):
        return 1, 1

class Identity(object):
    def __init__(self):
        pass
    def forward(self, X):
        self.X1 = X
        return self.X1
    def backward(self):
        return 1

class Dot(object):
    def __init__(self):
        pass
    def forward(self, X1, X2):
        self.X1, self.X2 = X1, X2
        return Variable(np.dot(self.X1.value, self.X2.value), lr=0)
    def backward(self, nextop):
        return np.dot(nextop.D, self.X2.value.T), np.dot(self.X1.value.T, nextop.D)
        # return self.X2.value, self.X1.value.T

class Flatten(object):
    def __init__(self):
        pass
    def forward(self, X):
        self.X1 = X
        return Variable(np.reshape(self.X1.value, (self.X1.value.shape[0], -1)), lr=0)
    def backward(self, nextop):
        return np.reshape(nextop.D, (self.X1.value.shape))


class Conv2d(object):
    def __init__(self, padding = 'same', stride = 1, pad = 0):
        self.padding = padding
        self.stride = stride
        self.pad = pad
    def forward(self, X1, X2):
        self.X = self.X1 = X1
        self.filter = self.X2 = X2
        N, C, H, W = self.X.value.shape
        self.filter_h, self.filter_w, self.filter_c, self.filter_c2 = self.filter.value.shape
        if self.padding == 'valid':
            self.padH, self.padW = self.pad, self.pad
        elif self.padding == 'same':
            self.padH = ((H - 1) * (self.stride - 1) + self.filter_h - 1) // 2
            self.padW = ((W - 1) * (self.stride - 1) + self.filter_w - 1) // 2
        else:
            raise ValueError('self.padding value error')
        out_h = (H + 2 * self.padH - self.filter_h) // self.stride + 1
        out_w = (W + 2 * self.padW - self.filter_w) // self.stride + 1
        colX = im2col(self.X.value, self.filter.value, self.stride, self.padH, self.padW)
        colFilter = np.reshape(self.filter.value, [-1, self.filter.value.shape[3]])
        y = np.dot(colX, colFilter)
        y = np.transpose(y, [0, 2, 1])
        y = y.reshape([y.shape[0], y.shape[1], out_h, out_w])
        return Variable(y, lr=0)

    def backward(self, nextop):
        pass













