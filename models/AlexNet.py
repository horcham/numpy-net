import numpynet as nn

class AlexNet(object):
	def __init__(self, imagesize, flatten_size, num_labels, lr=1e-3):
		self.N, self.C, self.W, self.H = imagesize
		self.num_labels = num_labels
		self.lr = lr

		self.graph = nn.Graph()
		self.X = nn.Placeholder()

		W0 = nn.Variable(nn.UniformInit([11, 11, self.C, 96]), lr=lr)
		b0 = nn.Variable(nn.UniformInit([96, 1]), lr=lr)
		self.graph.add_vars([W0, b0])
		conv0 = nn.Op(nn.Conv2d(padding='SAME', stride=4), self.X, [W0, b0])
		self.graph.add_op(conv0)
		act0 = nn.Layer(nn.ReluActivator(), conv0)
		self.graph.add_layer(act0)
		pool0 = nn.Op(nn.MaxPooling(filter_h=3, filter_w=3, stride=2), act0)
		self.graph.add_op(pool0)

		W1 = nn.Variable(nn.UniformInit([5, 5, 96, 256]), lr=lr)
		b1 = nn.Variable(nn.UniformInit([256, 1]), lr=lr)
		self.graph.add_vars([W1, b1])
		conv1 = nn.Op(nn.Conv2d(padding='SAME', stride=1), pool0, [W1, b1])
		self.graph.add_op(conv1)
		act1 = nn.Layer(nn.ReluActivator(), conv1)
		self.graph.add_layer(act1)
		pool1 = nn.Op(nn.MaxPooling(filter_h=3, filter_w=3, stride=2), act1)
		self.graph.add_op(pool1)

		W2 = nn.Variable(nn.UniformInit([5, 5, 256, 384]), lr=lr)
		b2 = nn.Variable(nn.UniformInit([384, 1]), lr=lr)
		self.graph.add_vars([W2, b2])
		conv2 = nn.Op(nn.Conv2d(padding='SAME', stride=1), pool1, [W2, b2])
		self.graph.add_op(conv2)
		act2 = nn.Layer(nn.ReluActivator(), conv2)
		self.graph.add_layer(act2)

		W3 = nn.Variable(nn.UniformInit([5, 5, 384, 384]), lr=lr)
		b3 = nn.Variable(nn.UniformInit([384, 1]), lr=lr)
		self.graph.add_vars([W3, b3])
		conv3 = nn.Op(nn.Conv2d(padding='SAME', stride=1), act2, [W3, b3])
		self.graph.add_op(conv3)
		act3 = nn.Layer(nn.ReluActivator(), conv3)
		self.graph.add_layer(act3)

		W4 = nn.Variable(nn.UniformInit([5, 5, 384, 256]), lr=lr)
		b4 = nn.Variable(nn.UniformInit([256, 1]), lr=lr)
		self.graph.add_vars([W4, b4])
		conv4 = nn.Op(nn.Conv2d(padding='SAME', stride=1), act3, [W4, b4])
		self.graph.add_op(conv4)
		act4 = nn.Layer(nn.ReluActivator(), conv4)
		self.graph.add_layer(act4)

		pool5 = nn.Op(nn.MaxPooling(filter_h=3, filter_w=3, stride=2), act4)
		self.graph.add_op(pool2)

		fla = nn.Op(nn.Flatten(), pool5)
		self.graph.add_op(fla)


		WFC0 = nn.Variable(nn.UniformInit([flatten_size, 4096]), lr=lr)
		bFC0 = nn.Variable(nn.UniformInit([4096, 1]), lr=lr)
		FC0 = nn.Op(nn.Dot(), fla, [WFC0, bFC0])
		self.graph.add_op(FC0)
		Fact0 = nn.Layer(nn.ReluActivator(), FC0)
		self.graph.add_layer(Fact0)
		dp0 = nn.Op(nn.Dropout(0.3), Fact0)
		self.graph.add_op(dp0)

		WFC1 = nn.Variable(nn.UniformInit([4096, 4096]), lr=lr)
		bFC1 = nn.Variable(nn.UniformInit([4096, 1]), lr=lr)
		FC1 = nn.Op(nn.Dot(), dp0, [WFC1, bFC1])
		self.graph.add_op(FC1)
		Fact1 = nn.Layer(nn.ReluActivator(), FC1)
		self.graph.add_layer(Fact1)
		dp1 = nn.Op(nn.Dropout(0.3), Fact1)
		self.graph.add_op(dp1)

		WFC2 = nn.Variable(nn.UniformInit([4096, 1000]), lr=lr)
		bFC2 = nn.Variable(nn.UniformInit([1000, 1]), lr=lr)
		FC2 = nn.Op(nn.Dot(), dp1, [WFC2, bFC2])
		self.graph.add_op(FC2)
		Fact2 = nn.Layer(nn.ReluActivator(), FC2)
		self.graph.add_layer(Fact2)
		dp2 = nn.Op(nn.Dropout(0.3), Fact2)
		self.graph.add_op(dp2)

		WFC3 = nn.Variable(nn.UniformInit([1000, num_labels]), lr=lr)
		bFC3 = nn.Variable(nn.UniformInit([num_labels, 1]), lr=lr)
		FC3 = nn.Op(nn.Dot(), dp2, [WFC3, bFC3])
		self.graph.add_op(FC3)
		Fact3 = nn.Layer(nn.ReluActivator(), FC3)
		self.graph.add_layer(Fact3)
		dp3 = nn.Op(nn.Dropout(0.3), Fact3)
		self.graph.add_op(dp3)

		self.graph.add_loss(nn.Loss(nn.Softmax()))
		self.graph.add_optimizer((nn.AdamOptimizer()))


	def train(self, X_train, Y_train, X_test, Y_test, \
	          epochs, batchsize=100):
		for epoch in range(epochs):
			batch_tr = nn.miniBatch(X_train, Y_train, batchsize)
			batch_te = nn.miniBatch(X_test, Y_test, batch_size=100)
			for i, (batch_x, batch_y) in enumerate(batch_tr):
				self.graph.forward(batch_x)
				self.graph.calc_loss(batch_y)
				self.graph.backward()
				self.graph.update()
				if i % 50 == 0:
					print('epoch:{}/{}, batch:{}/{}, train loss:{}'.format(epoch, epochs, i, len(batch_tr), self.graph.loss))
			accuracy = self.graph.accuracy(batch_te)
			print('epoch:{}, accuracy:{}'.format(epoch, accuracy))


