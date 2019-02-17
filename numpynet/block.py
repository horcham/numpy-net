import numpy as np
from .op import *
from .variable import *
from .layer import *
import copy

class Block(object):
    '''
         |    |--------------- X or WX -------------------|     |
         |    |                                           |     |
    --------> X --> Op1(X) --> Op2() --> ... --> Opn() -- + ------->
         |                                                      |
         |                      Block                           |
    '''

    def __init__(self, X1, operators, w = None):
        '''
        Parameters:
        -----------
        operators : list[op1,...opn]
        X1 : Variable
        X2 : list[op1xs, ... opnxs], op1xs: list[op1_x1, ... op1_xn], opi_xj: Variable
        W  : Variable, if dims do not match
        '''
        self.X1 = X1
        self.operators = operators
        self.w = w
    def forward(self, if_train=True):
        self.output = copy.copy(self.X1)
        for i in range(len(self.operators)):
            _op = self.operators[i]
            self.output = _op.forward(if_train)
        if self.w == None:
            self.output += self.X1
        else:
            self.output += np.dot(self.w, self.X1)
        self.value = self.output.value

    def backward(self, nextop):
        self.nextop = nextop
        self.D = copy.copy(nextop)
        for i in range(len(self.operators)-1, -1, -1):
            _op = self.operators[i]
            _op.backward(self.D)
        if self.w == None:
            self.D += 1
        else:
            self.D += np.dot(self.nextop, self.w.value.T)



class ResBlock(object):
    def __init__(self, X1=None, X2s=None, scps=None):
        self.X1 = X1
        self.X2s = X2s
        self.scps = scps  # short parameters

        self.conv0 = Op(Conv2d(), X1, X2s[0]['w'])
        self.bn0 = Op(BatchNorm(X2s[1]['gamma'], X2s[1]['beta']), self.conv0)
        self.act0 = Layer(ReluActivator(), self.bn0)
        self.conv1 = Op(Conv2d(), self.act0, X2s[2]['w'])
        self.bn1 = Op(BatchNorm(X2s[3]['gamma'], X2s[3]['beta']), self.conv0)
        self.operators = [self.conv0, self.bn0, self.act0, self.conv1, self.bn1]

        if scps != None:
            self.w, self.gamma, self.beta = scps[0], scps[1], scps[2]
            self.sc_conv0 = Op(Conv2d(), self.X1, self.w)
            self.sc_bn = Op(BatchNorm(self.gamma, self.beta), self.sc_conv0)

    def forward(self, if_train=True):
        self.output = copy.copy(self.X1)
        self.shortcut = copy.copy(self.X1)
        for i in range(len(self.operators)):
            _op = self.operators[i]
            self.output = _op.forward(if_train)

        if self.scps == None:
            self.output.value += self.X1.value
        else:
            self.sc_conv0.forward(if_train)
            self.sc_bn.forward(if_train)
            self.output.value += self.sc_bn.value

        self.value = self.output.value

    def backward(self, nextop):
        self.nextop = nextop
        self.D = copy.copy(nextop)
        for i in range(len(self.operators)-1, -1, -1):
            _op = self.operators[i]
            _op.backward(self.D)

        if self.scps == None:
            self.Dsc = 1
        else:
            self.sc_bn.backward(self.nextop)
            self.sc_conv0.backward(self.sc_bn.D)
            self.Dsc = self.sc_conv0.D

        self.D += self.Dsc

























