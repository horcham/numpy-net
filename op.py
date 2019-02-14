import numpy as np
from variable import *
from utils.im_col import im2col, col2im

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

    def __repr__(self):
        return self.operator.name

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
            # self.X1.D, self.X2.D = np.mean(self.X1.D, axis=0, keepdims=True), np.mean(self.X2.D, axis=0, keepdims=True)
            self.D = self.X1.D
            print('D=dX1:{}, dx2:{}'.format(self.X1.D.shape, self.X2.D.shape))
        else:
            self.X1.D = self.operator.backward(self.nextop)
            # self.X1.D = np.mean(self.X1.D, axis=0, keepdims=True)
            self.D = self.X1.D
            print('D=dX1:{}'.format(self.X1.D.shape))
            # self.X1.D = self.dX1 * self.nextop.D

class Add(object):
    def __init__(self):
        self.name = 'Add'
    def forward(self, X1, X2):
        self.X1, self.X2 = X1, X2
        return Variable(self.X1.value + self.X2.value, lr=0)
    def backward(self, nextop):
        return nextop.D, nextop.D

class Identity(object):
    def __init__(self):
        self.name = 'Identity'
    def forward(self, X):
        self.X1 = X
        return self.X1
    def backward(self, nextop):
        return nextop.D

class Dot(object):
    def __init__(self):
        self.name = 'Dot'
    def forward(self, X1, X2):
        self.X1, self.X2 = X1, X2
        return Variable(np.dot(self.X1.value, self.X2.value), lr=0)
    def backward(self, nextop):
        # meanX1value = np.mean(self.X1.value, axis=0, keepdims=True)
        # return np.dot(nextop.D, self.X2.value.T), np.dot(meanX1value.T, nextop.D)
        return np.dot(nextop.D, self.X2.value.T), np.dot(self.X1.value.T, nextop.D)

class Flatten(object):
    def __init__(self):
        self.name = 'Flatten'
    def forward(self, X):
        self.X1 = X
        return Variable(np.reshape(self.X1.value, (self.X1.value.shape[0], -1)), lr=0)
    def backward(self, nextop):
        # return np.reshape(nextop.D, (1, self.X1.value.shape[1], self.X1.value.shape[2], self.X1.value.shape[3]))
        return np.reshape(nextop.D, self.X1.value.shape)


class Conv2d(object):
    def __init__(self, padding = 'same', stride = 1, pad = 0):
        self.padding = padding
        self.stride = stride
        self.pad = pad
        self.name = 'Conv2d'
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
        self.out_h = (H + 2 * self.padH - self.filter_h) // self.stride + 1
        self.out_w = (W + 2 * self.padW - self.filter_w) // self.stride + 1
        colX = im2col(self.X.value, self.filter.value.shape[0], self.filter.value.shape[1], self.stride, self.padH, self.padW)
        colFilter = np.reshape(self.filter.value, [-1, self.filter.value.shape[3]])
        y = np.dot(colX, colFilter)
        y = np.transpose(y, [0, 2, 1])
        y = y.reshape([y.shape[0], y.shape[1], self.out_h, self.out_w])
        return Variable(y, lr=0)

    def backward(self, nextop):
        _nextD = nextop.D
        _nextD_reshape = _nextD.transpose(1,2,3,0).reshape([self.filter.value.shape[3], -1])
        _colX = im2col(self.X.value, self.filter.value.shape[1], self.filter.value.shape[0], self.stride, self.padH, self.padW)
        _colX = _colX.reshape([_colX.shape[0]*_colX.shape[1], _colX.shape[2]]).T
        _DW = np.dot(_nextD_reshape, _colX.T)
        _DW = _DW.reshape(self.filter.value.shape)
        W_reshape = self.filter.value.transpose(3,0,1,2).reshape(self.filter_c2, -1)
        _DX_col = np.dot(W_reshape.T, _nextD_reshape)
        _DX_col = _DX_col.T.reshape([self.X1.value.shape[0], _DX_col.shape[1]/self.X1.value.shape[0], _DX_col.shape[0]])
        _DX = col2im(_DX_col, self.filter_h, self.filter_w, self.X1.value.shape)
        return _DX, _DW


class MaxPooling(object):
    def __init__(self, filter_h, filter_w, stride = 2, pad = 0):
        self.name = 'Maxpooling'
        self.filter_h = filter_h
        self.filter_w = filter_w
        self.stride = stride
        self.padH = self.padW = pad
    def forward(self, X):
        self.X = X
        print(X.value.shape)
        N, C, H, W = self.X.value.shape
        self.out_h = (H + 2 * self.padH - self.filter_h) // self.stride + 1
        self.out_w = (W + 2 * self.padW - self.filter_w) // self.stride + 1
        X_col = im2col(self.X.value, filter_h=self.filter_h, filter_w=self.filter_h, \
                       stride=self.stride, padH=self.padH, padW=self.padW)
        print(X_col.shape)
        X_col = X_col.transpose([0, 2, 1])
        X_col = X_col.reshape([X_col.shape[0], C, X_col.shape[1] / C, X_col.shape[2]])
        X_col = X_col.reshape([X_col.shape[0] * X_col.shape[1], X_col.shape[2], X_col.shape[3]])
        X_col = X_col.transpose([0, 2, 1]).reshape([X_col.shape[0] * X_col.shape[2], X_col.shape[1]])
        self.X_col = X_col
        print(X_col.shape)
        self.max_idx = np.argmax(X_col, axis=1)
        out = X_col[range(X_col.shape[0]), self.max_idx]
        out = out.reshape([N, C, self.out_h, self.out_w])
        print(out.shape)
        return Variable(out, lr=0)
    def backward(self, nextop):
        N, C, H, W = self.X.value.shape
        _nextD = nextop.D
        _nextD = _nextD.transpose(2,3,0,1).ravel()
        print(_nextD.shape)
        _DX_col = np.zeros(self.X_col.shape)
        _DX_col[range(self.X_col.shape[0]), self.max_idx] = _nextD
        print(_DX_col.shape)
        _DX_col = _DX_col.reshape([N, _DX_col.shape[0]/N, _DX_col.shape[1]])
        print(_DX_col.shape)
        _DX_col = _DX_col.reshape([N, _DX_col.shape[1]/C, C, _DX_col.shape[2]])
        print(_DX_col.shape)
        _DX_col = _DX_col.reshape([N, _DX_col.shape[1], C*_DX_col.shape[3]])
        print(_DX_col.shape)
        _DX = col2im(_DX_col, self.filter_h, self.filter_w, self.X.value.shape, stride=self.stride)
        print(_DX.shape)
        return _DX












