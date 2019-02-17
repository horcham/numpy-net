import numpy as np
import numpynet as nn

class ResNet18(object):
    def __init__(self, imagesize, flatten_size, num_labels, lr=1e-4):
        self.N, self.C, self.W, self.H = imagesize
        self.num_labels = num_labels
        self.lr = lr

        self.graph = nn.Graph()
        self.X = nn.Placeholder()

        W0 = nn.Variable(nn.UniformInit([3, 3, self.C, 64]), lr=lr)
        self.graph.add_var(W0)
        b0 = nn.Variable(nn.UniformInit([64, 1]), lr=lr)
        self.graph.add_var(b0)

