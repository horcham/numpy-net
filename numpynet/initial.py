import numpy as np

def UniformInit(shape, low=-1, high=1):
	return np.random.uniform(low, high, shape)

def NormalInit(shape, loc=0, scale=1.0):
	return np.random.normal(loc, scale, shape)

