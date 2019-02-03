import numpy as np

class Loss(object):
    '''
    ----------- |                    F = loss
    op or layer | --------> output ==========  label
    -----------	| <-----------------  d F/d output
    '''
    def __init__(self, lossfunc):
        self.lossfunc = lossfunc

    def __repr__(self):
        return "loss:{}".format(self.lossfunc.loss)

    def forward(self, output, Y):
        self.X1 = self.output = output
        self.X2 = self.Y = Y
        self.loss = self.lossfunc.forward(output, Y)
        self.F = self.loss

    def backward(self):
        self.dF = self.lossfunc.backward()
        self.X1.D = self.dF


class MSE(object):
    def __init__(self):
        pass
    def forward(self, output, Y):
        self.output = output
        self.Y = Y
        self.loss = 1.0/2 * np.sum(np.dot((output.value - Y.value), (output.value - Y.value).T))
        return self.loss
    def backward(self):
        return self.Y.value - self.output.value




