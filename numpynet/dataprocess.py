import numpy as np
# import minpy.numpy as np
from .variable import *
from PIL import Image

def onehot(label):
    '''
    convert label to one-hot encoding

    Parameters:
    -----------
    label: [samples, 1]
        example: [[2], [3], [4], [1], [2],... ]
    '''
    _label = np.unique(label)
    _labeldict = {}
    for i in range(_label.shape[0]):
        _labeldict[_label[i]] = i
    Y = np.zeros([label.shape[0], _label.shape[0]])
    for i in range(label.shape[0]):
        Y[i, np.int(_labeldict[label[i, 0]])] = 1
    return Y

def miniBatch(X, Y, batch_size=10):
    X_value, Y_value = X.value, Y.value
    Batchs = []
    batch_nums = X_value.shape[0] / batch_size
    for i in range(batch_nums):
        Batchs.append([Variable(X_value[i*batch_size:(i+1)*batch_size], lr=0), Variable(Y_value[i*batch_size:(i+1)*batch_size], lr=0)])
        # print(Batchs[i][0].value.shape, Batchs[i][1].value.shape)
    return Batchs


def scaleallimage(X, shape):
    '''
    resize X into shape

    X: [n, c, h, w] of data, np.array
    shape: new [h, w] of image
    '''
    new_X = []
    for i in range(X.shape[0]):
        x = X[i]
        if len(X.shape) == 4:
            x = x.transpose([1, 2, 0])
        x = Image.fromarray(x)
        x = x.resize(shape)
        x = np.array(x)
        if len(X.shape) == 4:
            x = x.transpose([2, 0, 1])
        x = np.expand_dims(x, axis=0)
        new_X.append(x)
    return np.vstack(new_X)

def randomshuffle(X, Y):
    index = np.random.choice(len(X), len(X), replace=False)
    return X[index], Y[index]