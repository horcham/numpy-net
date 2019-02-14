# numpy-net



## About numpy-net

Numpy-net is a neural networks framework implemented by numpy. Its coding style is like Gluon of mxnet. It can do derivation automatically via definition of operation's backward function.  Now this framework supports BP network and CNN.

---

## Support

- Initial
  - Uniform
  - Normal

- Operation
  - Add
  - Identity
  - Dot
  - Flatten
  - Conv2d
  - Maxpooling
- Activator
  - Relu
  - Identity
  - Sigmoid
  - Tanh
- Loss Function
  - MSE
  - Softmax
- Optimizer
  - SGD
  - Momentum
  - AdaGram
  - AdaDelta / RMSprop
  - Adam

---

## Quick Start

### Requirements

- python2.7
- numpy 1.11.3

Get numpy-net

```python
git clone https://github.com/horcham/numpy-net.git
cd numpy-net
```

There are some demos, `exam/main_BP` is a implement of BP, `exam/main_conv` is a implement of CNN, you can run them and have fun:

```
cd exam
python main_BP.py  
python main_conv.py
```

---

## Tutorials

### Graph

`Graph` is a network definition. During `Graph` 's definition, `Variable`,`Operation`,`Layer`, `Loss Function`, `Optimizer` are just symbol, we can add them into graph.

```python
graph = Graph()		    # Graph initial

X = Variable(UniformInit([1000, 50]), lr=0)								# input data
Y = Variable(onehot(np.random.choice(['a', 'b'], [1000, 1])), lr=0)     # output data, contain two labels

W0 = Variable(UniformInit([50, 30]), lr=0.01)
graph.add_var(W0)       # Add Variable into graph
W1 = Variable(UniformInit([30, 2]), lr=0.01)
graph.add_var(W1)

FC0 = Op(Dot(), X, W0)
graph.add_op(FC0)     # Add Operation into graph

act0 = Layer(SigmoidActivator(), FC0)
graph.add_layer(act0)   # Add Layer into graph

FC1 = Op(Dot(), act0, W1)
graph.add_op(FC1)

graph.add_loss(Loss(Softmax()))  	# add Loss function
graph.add_optimizer(SGDOptimizer())	# add Optimizer
```

After the definition, we can train the net

```python
graph.forward()			# netword forward
graph.calc_loss(Y)		# use label Y and calculate loss
graph.backward()		# netword backward and calculate gradient
graph.update()			# update the variable in graph.add_var by optimizer
```



### Variables

`Variable` is similar to `numpy.array`, but it is a class which also contains other attributes like lr(learning rate), D(gradient). Any input and variables should be convert to `Variable` before feeding into network. 

```python
graph = Graph()

X = Variable(X, lr=0)	# if X is not trainable, lr=0
w = Variable(w, lr=1)	# if w is trainable, lr=1, and add it into graph
graph.add_var(w)
```



