import numpy as np
import copy

class Graph(object):
    def __init__(self):
        self.graph = []
        self.ops = []
        self.layers = []
        self.variable = []

    def add_var(self, var):
        self.variable.append(var)

    def add_op(self, op):
        self.graph.append(op)
        self.ops.append(op)

    def add_layer(self, layer):
        self.graph.append(layer)
        self.layers.append(layer)

    def add_loss(self, loss):
        self.loss = loss

    def add_optimizer(self, optimizer):
        self.optimizer = optimizer

    def forward(self):
        for g in self.graph:
            g.forward()
        self.output = self.graph[-1].output

    def calc_loss(self, Y):
        self.lossvalue = self.loss.forward(self.output, Y)

    def backward(self):
        self.loss.backward()
        self.back_graph = copy.copy(self.graph)
        self.back_graph.append(self.loss)
        for i in range(len(self.back_graph)-1, -1, -1):
            print(i, self.back_graph[i])
            if(i == len(self.back_graph)-1):
                self.back_graph[i].backward()
            else:
                self.back_graph[i].backward(self.back_graph[i+1])

    def update(self):
        for var in self.variable:
            var.update(self.optimizer)





