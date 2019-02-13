import numpy as np
import sys
sys.path.append('../')
from op import *
from layer import *
from variable import *
from loss import *
from initial import *
from optimizer import *
from graph import *
from dataprocess import *

if __name__ == '__main__':
    graph = Graph()
    X = Variable(UniformInit([1000, 50]), lr=0)
    # Y = Variable(UniformInit([1000, 3]), lr=0)
    Y = Variable(onehot(np.random.choice(['a', 'b'], [1000, 1])), lr=0)
    print(Y.value.shape)

    W0 = Variable(UniformInit([50, 30]), lr=0.01)
    graph.add_var(W0)
    W1 = Variable(UniformInit([30, 10]), lr=0.01)
    graph.add_var(W1)
    W2 = Variable(UniformInit([10, 5]), lr=0.01)
    graph.add_var(W2)
    W3 = Variable(UniformInit([5, 2]), lr=0.01)
    graph.add_var(W3)

    add0 = Op(Dot(), X, W0)
    graph.add_op(add0)
    act0 = Layer(SigmoidActivator(), add0)
    graph.add_layer(act0)

    add1 = Op(Dot(), act0, W1)
    graph.add_op(add1)
    act1 = Layer(SigmoidActivator(), add1)
    graph.add_layer(act1)

    add2 = Op(Dot(), act1, W2)
    graph.add_op(add2)
    act2 = Layer(SigmoidActivator(), add2)
    graph.add_layer(act2)

    add3 = Op(Dot(), act2, W3)
    graph.add_op(add3)
    # act3 = Layer(SigmoidActivator(), add3)
    # graph.add_layer(act3)

    # graph.add_loss(Loss(MSE()))
    graph.add_loss(Loss(Softmax()))
    graph.add_optimizer(SGDOptimizer())


    for t in range(5):
        print(t)
        graph.forward()
        graph.calc_loss(Y)
        print(graph.loss)
        graph.backward()
        graph.update()




