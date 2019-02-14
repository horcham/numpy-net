# coding:utf-8
import numpy as np
from variable import *
from utils.im_col import im2col, col2im

class Layer(object):
	'''
	       a_n =
	    op(z_{n-1})     |------------|  act   |-------------|
	------------------->|     a_n    | -----> |     Z_n     | ---------> op(z_n)
						|            |        |             |
	<-------------------|   dF/da_n  | <----- |   dF/dz_n   | <---------  dF/dop(z_n)

	a : [samples, features], Op or Layer
	'''
	def __init__(self, activator, a):
		self.a = a
		self.activator = activator

	def __repr__(self):
		return self.activator.name

	def forward(self):
		self.Z = self.activator.forward(self.a)
		self.output = self.Z
		self.value = self.Z.value

	def backward(self, nextop):
		self.nextop = nextop
		self.Pz = self.nextop.X1.D    # dF/dZ
		self.Pa = self.activator.backward(self.Z.value) * self.Pz   # dF/da = dF/dZ * dZ/da
		# self.D = np.mean(self.Pa, axis=0, keepdims=True)
		self.D = self.Pa
		print('D:{}'.format(self.D.shape))


# rule激活器
class ReluActivator(object):
	def __init__(self):
		self.name = 'Relu'

	def forward(self, X):    # 前向计算，计算输出
		return Variable(np.maximum(0, X.value), lr=0)

	def backward(self, output):  # 后向计算，计算导数
		output[output<0] = 0
		output[output>0] = 1
		return output
		# return 1 if output > 0 else 0

# IdentityActivator激活器.f(x)=x
class IdentityActivator(object):
	def __init__(self):
		self.name = 'Identity'

	def forward(self, X):   # 前向计算，计算输出
		return X

	def backward(self, output):   # 后向计算，计算导数
		return 1

#Sigmoid激活器
class SigmoidActivator(object):
	def __init__(self):
		self.name = 'Sigmoid'

	def forward(self, X):
		return Variable(1.0 / (1.0 + np.exp(-(X.value))), lr=0)

	def backward(self, output):
		# return output * (1 - output)
		return np.multiply(output, (1 - output))  # 对应元素相乘

# tanh激活器
class TanhActivator(object):
	def __init__(self):
		self.name = 'Tanh'

	def forward(self, X):
		return Variable(2.0 / (1.0 + np.exp(-2 * (X.value))) - 1.0, lr=0)

	def backward(self, output):
		return 1 - output * output




