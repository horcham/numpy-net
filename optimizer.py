import numpy as np

class SGDOptimizer(object):
	def __init__(self):
		pass
	def update(self, W, dW, lr):
		return W + lr * dW

