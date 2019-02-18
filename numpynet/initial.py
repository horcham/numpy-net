# import numpy as np
import minpy.numpy as np

def UniformInit(shape, low=-0.5, high=0.5):
	return np.random.uniform(low, high, shape)

def NormalInit(shape, loc=0, scale=1.0):
	return np.random.normal(loc, scale, shape)

