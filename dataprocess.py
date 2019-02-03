import numpy as np

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
    for i in range(len(_label)):
        _labeldict[_label[i]] = i
    Y = np.zeros([len(label), len(_label)])
    for i in range(len(label)):
        Y[i, _labeldict[label[i, 0]]] = 1
    return Y