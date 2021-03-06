# numpynet



## About numpynet

Numpynet is a neural networks framework implemented by numpy. 
Its coding style is like Gluon of mxnet. It can do derivation 
automatically via definition of operation's backward function.  
Now this framework supports BP network and CNN.

---
## Performance

|dataset|model|learning rate|epoch|accuracy|
|-------|----:|------------:|----:|-------:|
| MNIST |LeNet|  1e-3       |30   |0.9711  |

---
## Support Component

- Variable:  
  - `Variable(np.array, lr=0)`
  - `Placeholder()`
 
- Initial
  - Uniform:  `Uniform(shape)`
  - Normal:  `Normal(shape)`

- Operation
  - Add: `__init__()`, `forward(X1, X2)`
  - Identity `__init__()`, `forward(X1)`
  - Dot  `__init__()`, `forward(X1, X2)`
  - Flatten  `__init__()`, `forward(X1)`
  - Conv2d   `__init__(padding, stride, pad)`, `forward(X1, X2)` 
  - Maxpooling  `__init__(filter_h, filter_w, stride, pad)`, `forward(X1)`
  - Dropout `__init__(p)`, `forward(X1)`
  - BatchNorm `__init__(gamma, beta)`, `forward(X1)`
- Block
  - ResBlock  `__init__(X1, X2, scps)`, `forward()`
- Activator
  - Relu   `__init__()`, `forward(X1)` 
  - Identity   `__init__()`, `forward(X1)` 
  - Sigmoid   `__init__()`, `forward(X1)` 
  - Tanh   `__init__()`, `forward(X1)` 
- Loss Function
  - MSE   `__init__()` 
  - Softmax   `__init__()`
- Optimizer
  - SGD    `__init__()`    
  - Momentum   `__init__(beta1=0.9)`
  - AdaGram   `__init__()`
  - AdaDelta / RMSprop   `__init__(beta2=0.999)`
  - Adam   `__init__(beta1=0.9, beta2=0.999)`
- Models
  - LeNet 
  - AlexNet(without LRN)
  - VGG16
  - ResNet18
- Examples
  - mnist
  - cifar-10
---

## Quick Start

---
### Requirements

- python 2.7
- numpy 1.11.3

Get numpynet

```python
git clone https://github.com/horcham/numpy-net.git
python setup.py install
```
and you can import `numpynet` and play
```python
import numpynet as nn
```

There are some demos, `examples/mnist.py` is a demo that solving digital 
classification by Lenet. you can run them and have fun:

```
cd examples
python mnist.py  
```

---

## Tutorials

---
### Graph

`Graph` is a network definition. During `Graph` 's definition, `Variable`,`Operation`,`Layer`, `Loss Function`, `Optimizer` are just symbol, we can add them into graph.

```python
import numpynet as nn

input = nn.Variable(nn.UniformInit([1000, 50]), lr=0)		# input data
output = nn.Variable(nn.onehot(np.random.choice(['a', 'b'], [1000, 1])), lr=0)     # output data, contain two labels

graph = nn.Graph()		    # Graph initial

X = nn.Placeholder()       # Add Placeholder
W0 = nn.Variable(nn.UniformInit([50, 30]), lr=0.01)
graph.add_var(W0)       # Add Variable into graph
W1 = nn.Variable(nn.UniformInit([30, 2]), lr=0.01)
graph.add_var(W1)

FC0 = nn.Op(nn.Dot(), X, W0)
graph.add_op(FC0)     # Add Operation into graph

act0 = nn.Layer(nn.SigmoidActivator(), FC0)
graph.add_layer(act0)   # Add Layer into graph

FC1 = nn.Op(nn.Dot(), act0, W1)
graph.add_op(FC1)

graph.add_loss(nn.Loss(nn.Softmax()))  	# add Loss function
graph.add_optimizer(nn.SGDOptimizer())	# add Optimizer
```

After the definition, we can train the net

```python
graph.forward(input)			# netword forward
graph.calc_loss(output)		# use label Y and calculate loss
graph.backward()		# netword backward and calculate gradient
graph.update()			# update the variable in graph.add_var by optimizer
```


---
### Variables

`Variable` is similar to `numpy.array`, but it is a class which also 
contains other attributes like lr(learning rate), D(gradient). Any input 
and variables should be convert to `Variable` before feeding into network. 

```python
graph = Graph()

X = Variable(X, lr=0)	# if X is not trainable, lr=0
w = Variable(w, lr=1)	# if w is trainable, lr=1, and add it into graph
graph.add_var(w)
```

### Placeholder
When definding the graph, we use `Placeholder` to represent input data.
```python
X = Placeholder()       # Add Placeholder
W0 = Variable(UniformInit([50, 30]), lr=0.01)
FC0 = Op(Dot(), X, W0)
graph.add_op(FC0)     # Add Operation into graph
```
After definition, the graph begin to train, input data(`Variable`) and output data(`Variable`) 
were feed into graph by
```python
graph.forward(input)			# netword forward
graph.calc_loss(output)		# use label Y and calculate loss
```
and `Placeholder` is replaced by `Variable`

---
### Operation

`Operation` is similar to operations of numpy. For example, class Dot is 
similar to numpy.dot, but it also contains backward funtions, which is used 
to calculate gradient.

During `graph`'s definition, `Operation` is defined as symbol. 

How to define an `Operation`?
```python
op_handler = Op(operation(), X, W)  # operation which need two inputs
op_handler2 = Op(operation(), X)    # operation which need one inputs
```

`operation()` : `Dot()`,`Add`,`Conv2d` and so on

`X` : First input, `Variable`, which is not trainable

`W` : Second input, `Variable`, which is trainable

 For example
```python
FC0 = Op(Dot(), X, W0)  # Define operation: Dot
graph.add_op(FC0)     # Add Operation into graph
```

`Operation` calculates when `graph` begins to forward and backward.
```python
graph.forward()  # operation begins forward
graph.backward() # operation begins backward
```

#### Dropout
`Dropout` is an `Op`, we always add it before fully connected operation. We only need
to define `Dropout` in graph:
```python
add0 = Op(Dot(), X, W0)
graph.add_op(add0)
dp0 = Op(Dropout(0.3), add0)
graph.add_op(dp0)
act0 = Layer(SigmoidActivator(), dp0)
graph.add_layer(act0)
```


#### BatchNorm
`BatchNorm` is an `Op`, we always add it before `Layer`. Before define `BatchNorm`,
we should first define `gamma` and `beta` as trainable `Variable`, and add them into
 graph by `graph.add_var`:
```python
g0, bt0 = Variable(np.random.random(), lr=0.01), Variable(np.random.random(), lr=0.01)
graph.add_var(g0)
graph.add_var(bt0)
```

and define `BarchNorm` in graph:
```python
conv0 = Op(Conv2d(), X, W0)
graph.add_op(conv0)
bn0 = Op(BatchNorm(g0, bt0), conv0)
graph.add_op(bn0)
act0 = Layer(ReluActivator(), bn0)
graph.add_layer(act0)
```
After definition of graph, when `graph.backward` called, `gamma` and `beta` will be trained

---
### Block
Add `Block` to graph, now `Block` just support `ResBlock`(Block in ResNet-18), the follow shows
its construction
```
     |    |-------------> X or (conv --> BN)  --------------|     |
     |    |                                                 v     |
--------> X --> conv --> BN --> Relu --> conv --> BN -----> + ------->
     |                                                            |
     |                      Block                                 |
```
When the dimention of BN's output is different from X, the shortcut switch to `conv --> BN`, 
otherwise `X`. 

To define a `ResBlock`,  you call
```python
block = nn.ResBlock(X1, x2, sc)
```
`X1` is the output of previous `op`/`block`/`layer`. `X2` is dict, it contains the parameters
of `conv` and `BN` in main branch. `sc` contains the parameters of
shortcut, it is also dict.

For example, when shortcut is `X`
```python
# Block 1
W1_1 = nn.Variable(nn.UniformInit([3, 3, 64, 64]), lr=lr)
b1_1 = nn.Variable(nn.UniformInit([64, 1]), lr=lr)
gamma1_1 = nn.Variable(nn.UniformInit([1, 64, self.W, self.H]), lr=lr)
beta1_1 = nn.Variable(nn.UniformInit([1, 64, self.W, self.H]), lr=lr)
W1_2 = nn.Variable(nn.UniformInit([3, 3, 64, 64]), lr=lr)
b1_2 = nn.Variable(nn.UniformInit([64, 1]), lr=lr)
gamma1_2 = nn.Variable(nn.UniformInit([1, 64, self.W, self.H]), lr=lr)
beta1_2 = nn.Variable(nn.UniformInit([1, 64, self.W, self.H]), lr=lr)
self.graph.add_vars([W1_1, b1_1, gamma1_1, beta1_1, \
                     W1_2, b1_2, gamma1_2, beta1_2])
pamas1 = {'w1': W1_1, 'b1': b1_1, \  # the first conv in main branch
          'gamma1': gamma1_1, 'beta1': beta1_1, \  # the first BN in main branch
          'w2': W1_2, 'b2': b1_2, \  # the second conv in main branch
          'gamma2': gamma1_2, 'beta2': beta1_2}    # the second BN in main branch
          
B1 = nn.ResBlock(act0, pamas1)
self.graph.add_block(B1)
``` 

When shortcut comes to `conv --> BN`
```python
# Block 3
W3_1 = nn.Variable(nn.UniformInit([3, 3, 64, 128]), lr=lr)
b3_1 = nn.Variable(nn.UniformInit([128, 1]), lr=lr)
gamma3_1 = nn.Variable(nn.UniformInit([1, 128, self.W/2, self.H/2]), lr=lr)
beta3_1 = nn.Variable(nn.UniformInit([1, 128, self.W/2, self.H/2]), lr=lr)
W3_2 = nn.Variable(nn.UniformInit([3, 3, 128, 128]), lr=lr)
b3_2 = nn.Variable(nn.UniformInit([128, 1]), lr=lr)
gamma3_2 = nn.Variable(nn.UniformInit([1, 128, self.W/2, self.H/2]), lr=lr)
beta3_2 = nn.Variable(nn.UniformInit([1, 128, self.W/2, self.H/2]), lr=lr)

w_sc3 = nn.Variable(nn.UniformInit([3, 3, 64, 128]), lr=lr)
b_sc3 = nn.Variable(nn.UniformInit([128, 1]), lr=lr)
gamma_sc3 = nn.Variable(nn.UniformInit([1, 128, self.W/2, self.H/2]), lr=lr)
beta3_sc3 = nn.Variable(nn.UniformInit([1, 128, self.W/2, self.H/2]), lr=lr)
self.graph.add_vars([W3_1, b3_1, gamma3_1, beta3_1, \
                     W3_2, b3_2, gamma3_2, beta3_2, \
                     w_sc3, b_sc3, gamma_sc3, beta3_sc3])
pamas3 = {'w1': W3_1, 'b1': b3_1, \   # the first conv in main branch
          'gamma1': gamma3_1, 'beta1': beta3_1, \  # the first BN in main branch
          'w2': W3_2, 'b2': b3_2, \   # the second conv in main branch
          'gamma2': gamma3_2, 'beta2': beta3_2}    # the second BN in main branch
sc3 = {'w': w_sc3, 'b': b_sc3, \      # conv in shortcut
        'gamma': gamma_sc3, 'beta': beta3_sc3}     # BN in shortcut
B3 = nn.ResBlock(pool2, pamas3, sc3)
self.graph.add_block(B3)
```


---
### Layer
Add activations to graph

During `Layer`'s definition, `Layer` is defined as symbol. 
How to define an `Layer`?
```python
layer_handler = Layer(activator(), X)  
```
`X` : `Variable`, which is not trainable


 For example
```python
act0 = Layer(SigmoidActivator(), add0) # Define Layer: Sigmoid activation
graph.add_layer(act0)   # Add Layer into graph
```

`Layer` calculates when `graph` begins to forward and backward.
```python
graph.forward()  # Layer begins forward
graph.backward() # Layer begins backward
```

---
### Loss Function
Add Loss Function to graph, it will calculate loss, and begin 
calculate gradient.

During `Loss`'s definition, `Loss` is defined as symbol. 

How to define an `Loss`?
```python
loss_handler = Loss(loss_funtion())  
```
`loss_function()` can be `MSE()`, `Softmax`

For example
```python
graph.add_loss(Loss(Softmax()))
```

`Loss` calculates after graph forward, call
```python
graph.calc_loss(Y)
```
to calculate loss. `Y` is labels of data, `Variable`

After calculating loss, call 
```python
graph.backward()
``` 
to do backward.

---
### Optimizer
Add Optimizer to graph, it will update trainable `Variable`
after backward.

How to define an `Loss`?
```python
Optimizer_handler = Optimizer()  
```
`Optimizer()` can be `SGDOptimizer`, `MomentumOptimizer` and so on

For example
```python
graph.add_optimizer(AdamOptimizer())
```

After backward, trainable `Variable` needs to update according 
its gradient
```python
graph.update()
```

## More
If you want to define `Operation` or `Layer`, you only need to
define how this `Operation` or `Layer` forward and backward.Be careful
that the first input of `Operation` must be untrainable `Variable` and 
the second input must be trainable

Meanwhile, it is easy for you to define `Optimizer`, `Loss`



























