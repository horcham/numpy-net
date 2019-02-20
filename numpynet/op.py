import numpy as np
# import minpy.numpy as np
from .variable import *
from .im_col import im2col, col2im

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
        self.name = operator.name

    def __repr__(self):
        return self.operator.name

    def forward(self, if_train=True):
        if self.X2 != None:
            self.H = self.operator.forward(self.X1, self.X2, if_train)
        else:
            self.H = self.operator.forward(self.X1, if_train)
        self.output = self.H
        self.value = self.H.value
        # print('output:{}'.format(self.value.shape))

    def backward(self, nextop):
        self.nextop = nextop
        self.operator.backward(self.nextop)
        self.D = self.X1.D

class Add(object):
    def __init__(self):
        self.name = 'Add'
    def forward(self, X1, X2, if_train):
        self.X1, self.X2 = X1, X2
        return Variable(self.X1.value + self.X2.value, lr=0)
    def backward(self, nextop):
        self.X1.D = nextop.D
        self.X2.D = nextop.D

class Identity(object):
    def __init__(self):
        self.name = 'Identity'
    def forward(self, X, if_train):
        self.X1 = X
        return self.X1
    def backward(self, nextop):
        self.X1.D = nextop.D

class Dot(object):
    def __init__(self):
        self.name = 'Dot'
    def forward(self, X1, X2, if_train):
        self.X1, self.w, self.b = X1, X2[0], X2[1]
        return Variable(np.dot(self.X1.value, self.w.value) + self.b.value.T, lr=0)
    def backward(self, nextop):
        self.X1.D = np.dot(nextop.D, self.w.value.T)
        self.w.D = np.dot(self.X1.value.T, nextop.D)
        self.b.D = np.mean(nextop.D, axis=0, keepdims=True).T

class Flatten(object):
    def __init__(self):
        self.name = 'Flatten'
    def forward(self, X, if_train):
        self.X1 = X
        return Variable(np.reshape(self.X1.value, (self.X1.value.shape[0], -1)), lr=0)
    def backward(self, nextop):
        self.X1.D = np.reshape(nextop.D, self.X1.value.shape)


class Conv2d(object):
    def __init__(self, padding = 'same', stride = 1, pad = 0):
        self.padding = padding
        self.stride = stride
        self.pad = pad
        self.name = 'Conv2d'
    def forward(self, X1, X2, if_train):
        self.X = self.X1 = X1
        self.filter = self.X2_0 = X2[0]
        self.b = self.X2_0 = X2[1]
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
        cols = y.shape[1]
        y = np.reshape(y, [N*cols, self.filter_c2]) + self.b.value.T
        y = np.reshape(y, [N, cols, self.filter_c2])
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

        self.X1.D = _DX
        self.filter.D = _DW
        self.b.D = np.sum(_nextD, axis=(0, 2, 3)).reshape(self.filter_c2, -1)


class MaxPooling(object):
    def __init__(self, filter_h, filter_w, stride = 2, pad = 0):
        self.name = 'Maxpooling'
        self.filter_h = filter_h
        self.filter_w = filter_w
        self.stride = stride
        self.padH = self.padW = pad
    def forward(self, X, if_train):
        self.X = X
        N, C, H, W = self.X.value.shape
        self.out_h = (H + 2 * self.padH - self.filter_h) // self.stride + 1
        self.out_w = (W + 2 * self.padW - self.filter_w) // self.stride + 1
        X_col = im2col(self.X.value, filter_h=self.filter_h, filter_w=self.filter_h, \
                       stride=self.stride, padH=self.padH, padW=self.padW)
        X_col = X_col.transpose([0, 2, 1])
        X_col = X_col.reshape([X_col.shape[0], C, X_col.shape[1] / C, X_col.shape[2]])
        X_col = X_col.reshape([X_col.shape[0] * X_col.shape[1], X_col.shape[2], X_col.shape[3]])
        X_col = X_col.transpose([0, 2, 1]).reshape([X_col.shape[0] * X_col.shape[2], X_col.shape[1]])
        self.X_col = X_col
        self.max_idx = np.argmax(X_col, axis=1)
        out = X_col[range(X_col.shape[0]), self.max_idx]
        out = out.reshape([N, C, self.out_h, self.out_w])
        return Variable(out, lr=0)

    def backward(self, nextop):
        N, C, H, W = self.X.value.shape
        _nextD = nextop.D
        _nextD = _nextD.transpose(2,3,0,1).ravel()
        _DX_col = np.zeros(self.X_col.shape)
        _DX_col[range(self.X_col.shape[0]), self.max_idx] = _nextD
        _DX_col = _DX_col.reshape([N, _DX_col.shape[0]/N, _DX_col.shape[1]])
        _DX_col = _DX_col.reshape([N, _DX_col.shape[1]/C, C, _DX_col.shape[2]])
        _DX_col = _DX_col.reshape([N, _DX_col.shape[1], C*_DX_col.shape[3]])
        _DX = col2im(_DX_col, self.filter_h, self.filter_w, self.X.value.shape, stride=self.stride)
        self.X.D = _DX


class Dropout(object):
    def __init__(self, p):
        self.name = 'Dropout'
        self.p = p
    def forward(self, X, if_train):
        self.X1 = X
        if if_train:
            self.u1 = np.random.binomial(1, self.p, X.value.shape) / self.p
            return Variable(X.value * self.u1, lr=0)
        else:
            return X
    def backward(self, nextop):
        _nextD = nextop.D
        self.X1.D = _nextD * self.u1

class BatchNorm(object):
    def __init__(self, gamma, beta):
        self.name = 'BatchNorm'
        self.gamma = gamma
        self.beta = beta
        self.bn_mu = 0
        self.bn_var = 0
        self.N = 0

    def forward(self, X, if_train):
        if if_train:
            self.X = X
            self.mu = np.mean(X.value, axis=0)
            self.var = np.var(X.value, axis=0)
            self.X_norm = (X.value - self.mu)/ np.sqrt(self.var + 1e-7)
            self.out = self.gamma.value * self.X_norm + self.beta.value

            # self.bn_mu = self.mu * 0.1 + self.bn_mu * 0.9
            # self.bn_var = self.var * 0.1 + self.bn_var * 0.9
            self.bn_mu += self.mu
            self.bn_var += self.var
            self.N += 1

            return Variable(self.out, lr=0)
        else:
            self.X = X
            self.mu = np.mean(X.value, axis=0)
            # self.var = np.var(X.value, axis=0)
            # self.bn_mu = self.mu * 0.1 + self.bn_mu * 0.9
            # self.bn_var = self.var * 0.1 + self.bn_var * 0.9
            self.bn_mu_temp = self.mu * 0.1 + self.bn_mu/self.N * 0.9
            self.bn_var_temp = self.var * 0.1 + self.bn_var/self.N * 0.9
            self.out = (self.X.value - self.bn_mu_temp) / np.sqrt(self.bn_var_temp + 1e-7)
            self.out = self.gamma.value * self.out + self.beta.value
            return Variable(self.out, lr=0)

    def backward(self, nextop):
        _nextD = nextop.D
        X_mu = self.X.value - self.mu
        std_inv = 1.0 / np.sqrt(self.var + 1e-8)
        dhatX = _nextD * self.gamma.value
        dvar = np.sum(dhatX * X_mu, axis=0) * -0.5 * std_inv**3
        dmu = np.sum(dhatX * -1 * std_inv, axis=0) + dvar * np.mean(-2.0 * X_mu, axis=0)
        dX = (dhatX * std_inv) + (dvar * 2 * X_mu / self.X.value.shape[0]) + (dmu / self.X.value.shape[0])
        dgamma = np.sum(_nextD * dhatX, axis=0)
        dbeta = np.sum(_nextD, axis=0)
        self.gamma.D = dgamma
        self.beta.D = dbeta
        self.X.D = dX

        # dvar = np.sum(dhatX * (self.X.value - self.mu), axis=0) * -1.0/2 * (self.var + 1e-7)**(-3.0/2)
        # dmu = np.sum(dhatX * (-1.0 / np.sqrt(self.var + 1e-7)), axis=0) + \
        #       dvar * np.mean(-2 * (self.X.value - self.mu), axis=0)
        # dX = dhatX * 1.0 / np.sqrt(self.var + 1e-7) + \
        #      dvar * 2.0 * (self.X.value - self.mu) / self.X.value.shape[0] + \
        #      dmu * 1.0 / self.X.value.shape[0]
        # dgamma = np.sum(_nextD * dhatX, axis=0)
        # dbeta = np.sum(_nextD, axis=0)
        # self.gamma.D = dgamma
        # self.beta.D = dbeta
        # self.X.D = dX













