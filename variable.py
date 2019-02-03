import numpy as np


class Variable(object):
	def __init__(self, initial, lr):
		self.value = initial
		self.lr = lr
		self.D = 0

	def update(self, optimizer):
		self.value = optimizer.update(self.value, self.D, self.lr)

