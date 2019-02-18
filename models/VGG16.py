import numpynet as nn

class VGG16(object):
    def __init__(self, imagesize, flatten_size, num_labels, lr=1e-4):

        self.N, self.C, self.W, self.H = imagesize
        self.num_labels = num_labels
        self.lr = lr

        self.graph = nn.Graph()
        self.X = nn.Placeholder()

        W0 = nn.Variable(nn.UniformInit([3, 3, self.C, 64]), lr=self.lr)
        self.graph.add_var(W0)
        b0 = nn.Variable(nn.UniformInit([64, 1]), lr=lr)
        self.graph.add_var(b0)
        conv0 = nn.Op(nn.Conv2d(padding='same'), self.X, [W0, b0])
        self.graph.add_op(conv0)
        act0 = nn.Layer(nn.ReluActivator(), conv0)
        self.graph.add_layer(act0)

        W1 = nn.Variable(nn.UniformInit([3, 3, 64, 64]), lr=self.lr)
        self.graph.add_var(W1)
        b1 = nn.Variable(nn.UniformInit([64, 1]), lr=lr)
        self.graph.add_var(b1)
        conv1 = nn.Op(nn.Conv2d(padding='same'), act0, [W1, b1])
        self.graph.add_op(conv1)
        act1 = nn.Layer(nn.ReluActivator(), conv1)
        self.graph.add_layer(act1)

        pool1 = nn.Op(nn.MaxPooling(filter_h=2, filter_w=2, stride=2), act1)
        self.graph.add_op(pool1)

        W2 = nn.Variable(nn.UniformInit([3, 3, 64, 128]), lr=self.lr)
        self.graph.add_var(W2)
        b2 = nn.Variable(nn.UniformInit([128, 1]), lr=lr)
        self.graph.add_var(b2)
        conv2 = nn.Op(nn.Conv2d(padding='same'), pool1, [W2, b2])
        self.graph.add_op(conv2)
        act2 = nn.Layer(nn.ReluActivator(), conv2)
        self.graph.add_layer(act2)

        W3 = nn.Variable(nn.UniformInit([3, 3, 128, 128]), lr=self.lr)
        self.graph.add_var(W3)
        b3 = nn.Variable(nn.UniformInit([128, 1]), lr=lr)
        self.graph.add_var(b3)
        conv3 = nn.Op(nn.Conv2d(padding='same'), act2, [W3, b3])
        self.graph.add_op(conv3)
        act3 = nn.Layer(nn.ReluActivator(), conv3)
        self.graph.add_layer(act3)

        pool3 = nn.Op(nn.MaxPooling(filter_h=2, filter_w=2, stride=2), act3)
        self.graph.add_op(pool3)

        W4 = nn.Variable(nn.UniformInit([3, 3, 128, 256]), lr=self.lr)
        self.graph.add_var(W4)
        b4 = nn.Variable(nn.UniformInit([256, 1]), lr=lr)
        self.graph.add_var(b4)
        conv4 = nn.Op(nn.Conv2d(padding='same'), pool3, [W4, b4])
        self.graph.add_op(conv4)
        act4 = nn.Layer(nn.ReluActivator(), conv4)
        self.graph.add_layer(act4)

        W5 = nn.Variable(nn.UniformInit([3, 3, 256, 256]), lr=self.lr)
        self.graph.add_var(W5)
        b5 = nn.Variable(nn.UniformInit([256, 1]), lr=lr)
        self.graph.add_var(b5)
        conv5 = nn.Op(nn.Conv2d(padding='same'), act4, [W5, b5])
        self.graph.add_op(conv5)
        act5 = nn.Layer(nn.ReluActivator(), conv5)
        self.graph.add_layer(act5)

        W6 = nn.Variable(nn.UniformInit([3, 3, 256, 256]), lr=self.lr)
        self.graph.add_var(W6)
        b6 = nn.Variable(nn.UniformInit([256, 1]), lr=lr)
        self.graph.add_var(b6)
        conv6 = nn.Op(nn.Conv2d(padding='same'), act5, [W6, b6])
        self.graph.add_op(conv6)
        act6 = nn.Layer(nn.ReluActivator(), conv6)
        self.graph.add_layer(act6)

        W7 = nn.Variable(nn.UniformInit([3, 3, 256, 256]), lr=self.lr)
        self.graph.add_var(W7)
        b7 = nn.Variable(nn.UniformInit([256, 1]), lr=lr)
        self.graph.add_var(b7)
        conv7 = nn.Op(nn.Conv2d(padding='same'), act6, [W7, b7])
        self.graph.add_op(conv7)
        act7 = nn.Layer(nn.ReluActivator(), conv7)
        self.graph.add_layer(act7)

        pool8 = nn.Op(nn.MaxPooling(filter_h=2, filter_w=2, stride=2), act7)
        self.graph.add_op(pool8)

        W9 = nn.Variable(nn.UniformInit([3, 3, 256, 512]), lr=self.lr)
        self.graph.add_var(W9)
        b9 = nn.Variable(nn.UniformInit([512, 1]), lr=lr)
        self.graph.add_var(b9)
        conv9 = nn.Op(nn.Conv2d(padding='same'), pool8, [W9, b9])
        self.graph.add_op(conv9)
        act9 = nn.Layer(nn.ReluActivator(), conv9)
        self.graph.add_layer(act9)

        W10 = nn.Variable(nn.UniformInit([3, 3, 512, 512]), lr=self.lr)
        self.graph.add_var(W10)
        b10 = nn.Variable(nn.UniformInit([512, 1]), lr=lr)
        self.graph.add_var(b10)
        conv10 = nn.Op(nn.Conv2d(padding='same'), act9, [W10, b10])
        self.graph.add_op(conv10)
        act10 = nn.Layer(nn.ReluActivator(), conv10)
        self.graph.add_layer(act10)

        W11 = nn.Variable(nn.UniformInit([3, 3, 512, 512]), lr=self.lr)
        self.graph.add_var(W11)
        b11 = nn.Variable(nn.UniformInit([512, 1]), lr=lr)
        self.graph.add_var(b11)
        conv11 = nn.Op(nn.Conv2d(padding='same'), act10, [W11, b11])
        self.graph.add_op(conv11)
        act11 = nn.Layer(nn.ReluActivator(), conv11)
        self.graph.add_layer(act11)

        pool12 = nn.Op(nn.MaxPooling(filter_h=2, filter_w=2, stride=2), act11)
        self.graph.add_op(pool12)

        W13 = nn.Variable(nn.UniformInit([3, 3, 512, 512]), lr=self.lr)
        self.graph.add_var(W13)
        b13 = nn.Variable(nn.UniformInit([512, 1]), lr=lr)
        self.graph.add_var(b13)
        conv13 = nn.Op(nn.Conv2d(padding='same'), pool12, [W13, b13])
        self.graph.add_op(conv13)
        act13 = nn.Layer(nn.ReluActivator(), conv13)
        self.graph.add_layer(act13)

        W14 = nn.Variable(nn.UniformInit([3, 3, 512, 512]), lr=self.lr)
        self.graph.add_var(W14)
        b14 = nn.Variable(nn.UniformInit([512, 1]), lr=lr)
        self.graph.add_var(b14)
        conv14 = nn.Op(nn.Conv2d(padding='same'), act13, [W14, b14])
        self.graph.add_op(conv14)
        act14 = nn.Layer(nn.ReluActivator(), conv14)
        self.graph.add_layer(act14)

        W15 = nn.Variable(nn.UniformInit([3, 3, 512, 512]), lr=self.lr)
        self.graph.add_var(W15)
        b15 = nn.Variable(nn.UniformInit([512, 1]), lr=lr)
        self.graph.add_var(b15)
        conv15 = nn.Op(nn.Conv2d(padding='same'), act13, [W15, b15])
        self.graph.add_op(conv15)
        act15 = nn.Layer(nn.ReluActivator(), conv15)
        self.graph.add_layer(act15)

        pool16 = nn.Op(nn.MaxPooling(filter_h=2, filter_w=2, stride=2), act15)
        self.graph.add_op(pool16)

        fla = nn.Op(nn.Flatten(), pool16)
        self.graph.add_op(fla)

        WFC0 = nn.Variable(nn.UniformInit([flatten_size, 4096]), lr=lr)
        self.graph.add_var(WFC0)
        bFC0 = nn.Variable(nn.UniformInit([4096, 1]), lr=lr)
        self.graph.add_var(bFC0)
        FC0 = nn.Op(nn.Dot(), fla, [WFC0, bFC0])
        self.graph.add_op(FC0)
        Fact0 = nn.Layer(nn.ReluActivator(), FC0)
        self.graph.add_layer(Fact0)

        WFC1 = nn.Variable(nn.UniformInit([4096, 4096]), lr=lr)
        self.graph.add_var(WFC1)
        bFC1 = nn.Variable(nn.UniformInit([4096, 1]), lr=lr)
        self.graph.add_var(bFC1)
        FC1 = nn.Op(nn.Dot(), Fact0, [WFC1, bFC1])
        self.graph.add_op(FC1)
        Fact1 = nn.Layer(nn.ReluActivator(), FC1)
        self.graph.add_layer(Fact1)

        WFC2 = nn.Variable(nn.UniformInit([4096, num_labels]), lr=lr)
        self.graph.add_var(WFC2)
        bFC2 = nn.Variable(nn.UniformInit([num_labels, 1]), lr=lr)
        self.graph.add_var(bFC2)
        FC2 = nn.Op(nn.Dot(), Fact1, [WFC2, bFC2])
        self.graph.add_op(FC2)
        # Fact2 = nn.Layer(nn.ReluActivator(), FC2)
        # self.graph.add_layer(Fact2)

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






