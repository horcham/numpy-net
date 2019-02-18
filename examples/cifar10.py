import numpy as np
import cPickle
import cv2
import numpynet as nn
import sys
sys.path.append('../models')
from LeNet import LeNet
from VGG16 import VGG16
from ResNet18 import ResNet18

def readdata():
	tr_X, tr_Y = [], []
	te_X, te_Y = [], []
	for file in ['./cifar_data/data_batch_1', './cifar_data/data_batch_2', \
	             './cifar_data/data_batch_3', './cifar_data/data_batch_4', \
	             './cifar_data/data_batch_5']:
		with open(file, 'rb') as f:
			dict = cPickle.load(f)
			data = dict['data']
			data = data.reshape([data.shape[0], 3, 32, 32])
			tr_X.append(data)
			label = dict['labels']
			tr_Y += label

	with open('./cifar_data/test_batch', 'rb') as f:
		dict = cPickle.load(f)
		data = dict['data']
		data = data.reshape([data.shape[0], 3, 32, 32])
		te_X.append(data)
		label = dict['labels']
		te_Y += label
	tr_X, te_X = np.vstack(tr_X), np.vstack(te_X)
	tr_X, te_X = tr_X.astype(np.float32) - 127.5 , te_X.astype(np.float32) - 127.5
	tr_Y, te_Y = np.array(tr_Y), np.array(te_Y)
	return tr_X, tr_Y, te_X, te_Y

X_train, Y_train, X_test, Y_test = readdata()
print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)

X_train = nn.Variable(X_train, lr=0)

Y_train = np.expand_dims(Y_train, axis=1)
Y_train = nn.onehot(Y_train)
Y_train = nn.Variable(Y_train, lr=0)

X_test = nn.Variable(X_test, lr=0)

Y_test = np.expand_dims(Y_test, axis=1)
Y_test = nn.onehot(Y_test)
Y_test = nn.Variable(Y_test, lr=0)

# lenet = LeNet(X_train.value.shape, 512, 10)
# lenet.train(X_train, Y_train, X_test, Y_test, epochs=10, batchsize=10)

res18 = ResNet18(X_train.value.shape, 2048, 10)
res18.train(X_train, Y_train, X_test, Y_test, epochs=10, batchsize=10)





