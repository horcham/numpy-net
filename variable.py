import numpy as np


class Variable(object):
	def __init__(self, initial, lr):
		self.value = initial
		self.lr = lr
		self.D = 0
		self.DList = []
		self.m = 0
		self.V = 0

	def update(self, optimizer):
		optimizer.update(self)

class Placeholder(object):
	def __init__(self):
		pass

