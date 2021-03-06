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
    input = Variable(UniformInit([100, 3, 50, 50]), lr=0)
    output = Variable(onehot(np.random.choice(['a', 'b'], [100, 1])), lr=0)

    graph = Graph()
    X = Placeholder()
    W0 = Variable(UniformInit([3, 3, 3, 10]), lr=0.01)
    graph.add_var(W0)
    W1 = Variable(UniformInit([3, 3, 10, 20]), lr=0.01)
    graph.add_var(W1)
    W2 = Variable(UniformInit([3, 3, 20, 30]), lr=0.01)
    graph.add_var(W2)
    W3 = Variable(UniformInit([3, 3, 30, 40]), lr=0.01)
    graph.add_var(W3)
    W4 = Variable(UniformInit([1080, 10]), lr=0.01)
    graph.add_var(W4)
    W5 = Variable(UniformInit([10, 2]), lr=0.01)
    graph.add_var(W5)
    g0, bt0 = Variable(np.random.random(), lr=0.01), Variable(np.random.random(), lr=0.01)
    graph.add_var(g0)
    graph.add_var(bt0)
    g1, bt1 = Variable(np.random.random(), lr=0.01), Variable(np.random.random(), lr=0.01)
    graph.add_var(g1)
    graph.add_var(bt1)
    g2, bt2 = Variable(np.random.random(), lr=0.01), Variable(np.random.random(), lr=0.01)
    graph.add_var(g2)
    graph.add_var(bt2)

    conv0 = Op(Conv2d(), X, W0)
    graph.add_op(conv0)
    bn0 = Op(BatchNorm(g0, bt0), conv0)
    graph.add_op(bn0)
    act0 = Layer(ReluActivator(), bn0)
    graph.add_layer(act0)

    pool0 = Op(MaxPooling(2,2), act0)
    graph.add_op(pool0)

    conv1 = Op(Conv2d(), pool0, W1)
    graph.add_op(conv1)
    bn1 = Op(BatchNorm(g1, bt1), conv1)
    graph.add_op(bn1)
    act1 = Layer(ReluActivator(), bn1)
    graph.add_layer(act1)

    pool1 = Op(MaxPooling(2,2), act1)
    graph.add_op(pool1)

    conv2 = Op(Conv2d(), pool1, W2)
    graph.add_op(conv2)
    bn2 = Op(BatchNorm(g2, bt2), conv2)
    graph.add_op(bn2)
    act2 = Layer(ReluActivator(), bn2)
    graph.add_layer(act2)

    pool2 = Op(MaxPooling(2,2), act2)
    graph.add_op(pool2)

    fla1 = Op(Flatten(), pool2)
    graph.add_op(fla1)

    FC1 = Op(Dot(), fla1, W4)
    graph.add_op(FC1)
    act4 = Layer(SigmoidActivator(), FC1)
    graph.add_layer(act4)

    FC2 = Op(Dot(), act4, W5)
    graph.add_op(FC2)
    act5 = Layer(SigmoidActivator(), FC2)
    graph.add_layer(act5)

    graph.add_loss(Loss(Softmax()))
    graph.add_optimizer(AdamOptimizer())


    batch = miniBatch(input, output, batch_size=10)
    for epoch in range(10):
        print(epoch)
        for batch_x, batch_y in batch:
            graph.forward(batch_x)
            graph.calc_loss(batch_y)
            graph.backward()
            graph.update()
        print(graph.loss)




