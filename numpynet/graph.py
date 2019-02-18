import numpy as np
# import minpy.numpy as np

import copy

class Graph(object):
    def __init__(self):
        self.graph = []
        self.ops = []
        self.layers = []
        self.variable = []
        self.block = []

    def add_var(self, var):
        self.variable.append(var)

    def add_vars(self, varlist):
        self.variable += varlist

    def add_op(self, op):
        self.graph.append(op)
        self.ops.append(op)

    def add_ops(self, oplist):
        self.graph += oplist
        self.ops += oplist

    def add_layer(self, layer):
        self.graph.append(layer)
        self.layers.append(layer)

    def add_layers(self, layerlist):
        self.graph += layerlist
        self.layers += layerlist

    def add_block(self, block):
        self.graph.append(block)
        self.layers.append(block)

    def add_blocks(self, blocklist):
        self.graph += blocklist
        self.layers += blocklist

    def add_loss(self, loss):
        self.loss = loss

    def add_optimizer(self, optimizer):
        self.optimizer = optimizer

    def forward(self, X):
        for i, g in enumerate(self.graph):
            if i == 0:
                self.graph[i].X1 = X
            # print(i, self.graph[i], self.graph[i].X1.value.shape)
            g.forward()
            # print(g.value[0])
        self.output = self.graph[-1].output

    def calc_loss(self, Y):
        self.lossvalue = self.loss.forward(self.output, Y)

    def backward(self):
        self.loss.backward()
        self.back_graph = copy.copy(self.graph)
        self.back_graph.append(self.loss)
        for i in range(len(self.back_graph)-1, -1, -1):
            # print(i, self.back_graph[i])
            if(i == len(self.back_graph)-1):
                self.back_graph[i].backward()
            else:
                self.back_graph[i].backward(self.back_graph[i+1])

    def update(self):
        for var in self.variable:
            var.update(self.optimizer)

    def predict(self, X):
        for i, g in enumerate(self.graph):
            if i == 0:
                self.graph[i].X1 = X
            # print(i, self.graph[i])
            g.forward(if_train=False)
        self.output = self.graph[-1].output
        self.output = self.loss.predict(self.output)
        return self.output

    def accuracy(self, batch_te):
        acc = 0
        totals = 0
        for batch_x, batch_y in batch_te:
            y_hat = self.predict(batch_x)
            y = batch_y.value

            acc += np.sum(np.argmax(y_hat, axis=1) == np.argmax(y, axis=1))
            totals += len(y)
        return acc * 1.0 / totals




