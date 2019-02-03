import numpy as np
from variable import *

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

	def backward(self, nextop):
		self.nextop = nextop
		if self.X2 != None:
			self.X1.D, self.X2.D = self.operator.backward(self.nextop)
			# print('dX1:{}, nextop.D:{}'.format(self.X1.D.shape, self.nextop.D.shape))
		else:
			self.X1.D = self.operator.backward(self.nextop)
			# self.X1.D = self.dX1 * self.nextop.D

class Add(object):
	def __init__(self):
		pass
	def forward(self, X1, X2):
		return Variable(X1.value + X2.value, lr=0)
	def backward(self):
		return 1, 1

class Identity(object):
	def __init__(self):
		pass
	def forward(self, X):
		return X
	def backward(self):
		return 1

class Dot(object):
	def __int__(self):
		pass
	def forward(self, X1, X2):
		self.X1, self.X2 = X1, X2
		return Variable(np.dot(X1.value, X2.value), lr=0)
	def backward(self, nextop):
		return np.dot(nextop.D, self.X2.value.T), np.dot(self.X1.value.T, nextop.D)
		# return self.X2.value, self.X1.value.T


