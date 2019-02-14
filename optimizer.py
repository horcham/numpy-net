import numpy as np

class SGDOptimizer(object):
	def __init__(self):
		self.name = 'SGDOptimizer'
	def update(self, var):
		var.value = var.value - var.lr * var.D

class MomentumOptimizer(object):
	def __init__(self, beta1=0.9):
		self.name = 'MomentumOptimizer'
		self.beta1 = beta1
	def update(self, var):
		var.m = self.beta1 * var.m + (1 - self.beta1) * var.D
		var.value = var.value - var.lr * var.m

class AdaGramOptimizer(object):
	def __init__(self):
		self.name = 'AdaGramOptimizer'
	def update(self, var):
		var.V = np.sum(np.square(np.array(var.DList)), axis=0)
		var.value = var.value - var.lr * var.D / (np.sqrt(var.V)+1e-6)

class AdaDeltaOptimizer(object):
	def __init__(self, beta2=0.999):
		self.name = 'AdaDeltaOptimizer'
		self.beta2 = beta2
	def update(self, var):
		var.V = self.beta2 * var.V + (1 - self.beta2) * np.square(var.D)
		var.value = var.value - var.lr * var.D / (np.sqrt(var.V)+1e-6)

class RMSPropOptimizer(AdaDeltaOptimizer):
	def __init__(self, beta2=0.999):
		AdaDeltaOptimizer.__init__(self, beta2)
		self.name = 'RMSPropOptimizer'

class AdamOptimizer(object):
	def __init__(self, beta1=0.9, beta2=0.999):
		self.name = 'AdamOptimizer'
		self.beta1, self.beta2 = beta1, beta2
	def update(self, var):
		var.m = self.beta1 * var.m + (1 - self.beta1) * var.D
		var.V = self.beta2 * var.V + (1 - self.beta2) * np.square(var.D)
		var.value = var.value - var.lr * var.m / (np.sqrt(var.V)+1e-6)
