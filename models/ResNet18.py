import numpynet as nn

class ResNet18(object):
    def __init__(self, imagesize, flatten_size, num_labels, lr=1e-3):
        self.N, self.C, self.W, self.H = imagesize
        self.num_labels = num_labels
        self.lr = lr

        self.graph = nn.Graph()
        X = nn.Placeholder()

        W0 = nn.Variable(nn.UniformInit([3, 3, self.C, 64]), lr=lr)
        b0 = nn.Variable(nn.UniformInit([64, 1]), lr=lr)
        gamma0 = nn.Variable(nn.UniformInit([1, 64, self.W, self.H]), lr=lr)
        beta0 = nn.Variable(nn.UniformInit([1, 64, self.W, self.H]), lr=lr)
        self.graph.add_vars([W0, b0, gamma0, beta0])
        conv0 = nn.Op(nn.Conv2d(), X, [W0, b0])
        bn0 = nn.Op(nn.BatchNorm(gamma0, beta0), conv0)
        act0 = nn.Layer(nn.ReluActivator(), bn0)
        self.graph.add_ops([conv0, bn0])
        self.graph.add_layer(act0)

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
        pamas1 = {'w1': W1_1, 'b1': b1_1, \
                  'gamma1': gamma1_1, 'beta1': beta1_1, \
                  'w2': W1_2, 'b2': b1_2, \
                  'gamma2': gamma1_2, 'beta2': beta1_2}
        B1 = nn.ResBlock(act0, pamas1)
        self.graph.add_block(B1)

        # Block 2
        W2_1 = nn.Variable(nn.UniformInit([3, 3, 64, 64]), lr=lr)
        b2_1 = nn.Variable(nn.UniformInit([64, 1]), lr=lr)
        gamma2_1 = nn.Variable(nn.UniformInit([1, 64, self.W, self.H]), lr=lr)
        beta2_1 = nn.Variable(nn.UniformInit([1, 64, self.W, self.H]), lr=lr)
        W2_2 = nn.Variable(nn.UniformInit([3, 3, 64, 64]), lr=lr)
        b2_2 = nn.Variable(nn.UniformInit([64, 1]), lr=lr)
        gamma2_2 = nn.Variable(nn.UniformInit([1, 64, self.W, self.H]), lr=lr)
        beta2_2 = nn.Variable(nn.UniformInit([1, 64, self.W, self.H]), lr=lr)
        self.graph.add_vars([W2_1, b2_1, gamma2_1, beta2_1, \
                             W2_2, b2_2, gamma2_2, beta2_2])
        pamas2 = {'w1': W2_1, 'b1': b2_1, \
                  'gamma1': gamma2_1, 'beta1': beta2_1, \
                  'w2': W2_2, 'b2': b2_2, \
                  'gamma2': gamma2_2, 'beta2': beta2_2}
        B2 = nn.ResBlock(B1, pamas2)
        self.graph.add_block(B2)

        # pool2
        pool2 = nn.Op(nn.MaxPooling(2,2), B2)
        self.graph.add_layer(pool2)

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
        pamas3 = {'w1': W3_1, 'b1': b3_1, \
                  'gamma1': gamma3_1, 'beta1': beta3_1, \
                  'w2': W3_2, 'b2': b3_2, \
                  'gamma2': gamma3_2, 'beta2': beta3_2}
        sc3 = {'w': w_sc3, 'b': b_sc3, 'gamma': gamma_sc3, 'beta': beta3_sc3}
        B3 = nn.ResBlock(pool2, pamas3, sc3)
        self.graph.add_block(B3)

        # Block 4
        W4_1 = nn.Variable(nn.UniformInit([3, 3, 128, 128]), lr=lr)
        b4_1 = nn.Variable(nn.UniformInit([128, 1]), lr=lr)
        gamma4_1 = nn.Variable(nn.UniformInit([1, 128, self.W/2, self.H/2]), lr=lr)
        beta4_1 = nn.Variable(nn.UniformInit([1, 128, self.W/2, self.H/2]), lr=lr)
        W4_2 = nn.Variable(nn.UniformInit([3, 3, 128, 128]), lr=lr)
        b4_2 = nn.Variable(nn.UniformInit([128, 1]), lr=lr)
        gamma4_2 = nn.Variable(nn.UniformInit([1, 128, self.W/2, self.H/2]), lr=lr)
        beta4_2 = nn.Variable(nn.UniformInit([1, 128, self.W/2, self.H/2]), lr=lr)
        self.graph.add_vars([W4_1, b4_1, gamma4_1, beta4_1, \
                             W4_2, b4_2, gamma4_2, beta4_2])
        pamas4 = {'w1': W4_1, 'b1': b4_1, \
                  'gamma1': gamma4_1, 'beta1': beta4_1, \
                  'w2': W4_2, 'b2': b4_2, \
                  'gamma2': gamma4_2, 'beta2': beta4_2}
        B4 = nn.ResBlock(B3, pamas4)
        self.graph.add_block(B4)

        # pool4
        pool4 = nn.Op(nn.MaxPooling(2,2), B4)
        self.graph.add_layer(pool4)

        # Block 5
        W5_1 = nn.Variable(nn.UniformInit([3, 3, 128, 256]), lr=lr)
        b5_1 = nn.Variable(nn.UniformInit([256, 1]), lr=lr)
        gamma5_1 = nn.Variable(nn.UniformInit([1, 256, self.W/4, self.H/4]), lr=lr)
        beta5_1 = nn.Variable(nn.UniformInit([1, 256, self.W/4, self.H/4]), lr=lr)
        W5_2 = nn.Variable(nn.UniformInit([3, 3, 256, 256]), lr=lr)
        b5_2 = nn.Variable(nn.UniformInit([256, 1]), lr=lr)
        gamma5_2 = nn.Variable(nn.UniformInit([1, 256, self.W/4, self.H/4]), lr=lr)
        beta5_2 = nn.Variable(nn.UniformInit([1, 256, self.W/4, self.H/4]), lr=lr)

        w_sc5 = nn.Variable(nn.UniformInit([3, 3, 128, 256]), lr=lr)
        b_sc5 = nn.Variable(nn.UniformInit([256, 1]), lr=lr)
        gamma_sc5 = nn.Variable(nn.UniformInit([1, 256, self.W/4, self.H/4]), lr=lr)
        beta3_sc5 = nn.Variable(nn.UniformInit([1, 256, self.W/4, self.H/4]), lr=lr)
        self.graph.add_vars([W5_1, b5_1, gamma5_1, beta5_1, \
                             W5_2, b5_2, gamma5_2, beta5_2, \
                             w_sc5, b_sc5, gamma_sc5, beta3_sc5])
        pamas5 = {'w1': W5_1, 'b1': b5_1, \
                  'gamma1': gamma5_1, 'beta1': beta5_1, \
                  'w2': W5_2, 'b2': b5_2, \
                  'gamma2': gamma5_2, 'beta2': beta5_2}
        sc5 = {'w': w_sc5, 'b': b_sc5, 'gamma': gamma_sc5, 'beta': beta3_sc5}
        B5 = nn.ResBlock(pool4, pamas5, sc5)
        self.graph.add_block(B5)

        # Block 6
        W6_1 = nn.Variable(nn.UniformInit([3, 3, 256, 256]), lr=lr)
        b6_1 = nn.Variable(nn.UniformInit([256, 1]), lr=lr)
        gamma6_1 = nn.Variable(nn.UniformInit([1, 256, self.W/4, self.H/4]), lr=lr)
        beta6_1 = nn.Variable(nn.UniformInit([1, 256, self.W/4, self.H/4]), lr=lr)
        W6_2 = nn.Variable(nn.UniformInit([3, 3, 256, 256]), lr=lr)
        b6_2 = nn.Variable(nn.UniformInit([256, 1]), lr=lr)
        gamma6_2 = nn.Variable(nn.UniformInit([1, 256, self.W/4, self.H/4]), lr=lr)
        beta6_2 = nn.Variable(nn.UniformInit([1, 256, self.W/4, self.H/4]), lr=lr)
        self.graph.add_vars([W6_1, b6_1, gamma6_1, beta6_1, \
                             W6_2, b6_2, gamma6_2, beta6_2])
        pamas6 = {'w1': W6_1, 'b1': b6_1, \
                  'gamma1': gamma6_1, 'beta1': beta6_1, \
                  'w2': W6_2, 'b2': b6_2, \
                  'gamma2': gamma6_2, 'beta2': beta6_2}
        B6 = nn.ResBlock(B5, pamas6)
        self.graph.add_block(B6)

        # pool6
        pool6 = nn.Op(nn.MaxPooling(2,2), B6)
        self.graph.add_layer(pool6)

        # Block 7
        W7_1 = nn.Variable(nn.UniformInit([3, 3, 256, 512]), lr=lr)
        b7_1 = nn.Variable(nn.UniformInit([512, 1]), lr=lr)
        gamma7_1 = nn.Variable(nn.UniformInit([1, 512, self.W/8, self.H/8]), lr=lr)
        beta7_1 = nn.Variable(nn.UniformInit([1, 512, self.W/8, self.H/8]), lr=lr)
        W7_2 = nn.Variable(nn.UniformInit([3, 3, 512, 512]), lr=lr)
        b7_2 = nn.Variable(nn.UniformInit([512, 1]), lr=lr)
        gamma7_2 = nn.Variable(nn.UniformInit([1, 512, self.W/8, self.H/8]), lr=lr)
        beta7_2 = nn.Variable(nn.UniformInit([1, 512, self.W/8, self.H/8]), lr=lr)

        w_sc7 = nn.Variable(nn.UniformInit([3, 3, 256, 512]), lr=lr)
        b_sc7 = nn.Variable(nn.UniformInit([512, 1]), lr=lr)
        gamma_sc7 = nn.Variable(nn.UniformInit([1, 512, self.W/8, self.H/8]), lr=lr)
        beta3_sc7 = nn.Variable(nn.UniformInit([1, 512, self.W/8, self.H/8]), lr=lr)
        self.graph.add_vars([W7_1, b7_1, gamma7_1, beta7_1, \
                             W7_2, b7_2, gamma7_2, beta7_2, \
                             w_sc7, b_sc7, gamma_sc7, beta3_sc7])
        pamas7 = {'w1': W7_1, 'b1': b7_1, \
                  'gamma1': gamma7_1, 'beta1': beta7_1, \
                  'w2': W7_2, 'b2': b7_2, \
                  'gamma2': gamma7_2, 'beta2': beta7_2}
        sc7 = {'w': w_sc7, 'b': b_sc7, 'gamma': gamma_sc7, 'beta': beta3_sc7}
        B7 = nn.ResBlock(pool6, pamas7, sc7)
        self.graph.add_block(B7)

        # Block 8
        W8_1 = nn.Variable(nn.UniformInit([3, 3, 512, 512]), lr=lr)
        b8_1 = nn.Variable(nn.UniformInit([512, 1]), lr=lr)
        gamma8_1 = nn.Variable(nn.UniformInit([1, 512, self.W/8, self.H/8]), lr=lr)
        beta8_1 = nn.Variable(nn.UniformInit([1, 512, self.W/8, self.H/8]), lr=lr)
        W8_2 = nn.Variable(nn.UniformInit([3, 3, 512, 512]), lr=lr)
        b8_2 = nn.Variable(nn.UniformInit([512, 1]), lr=lr)
        gamma8_2 = nn.Variable(nn.UniformInit([1, 512, self.W/8, self.H/8]), lr=lr)
        beta8_2 = nn.Variable(nn.UniformInit([1, 512, self.W/8, self.H/8]), lr=lr)
        self.graph.add_vars([W8_1, b8_1, gamma8_1, beta8_1, \
                             W8_2, b8_2, gamma8_2, beta8_2])
        pamas8 = {'w1': W8_1, 'b1': b8_1, \
                  'gamma1': gamma8_1, 'beta1': beta8_1, \
                  'w2': W8_2, 'b2': b8_2, \
                  'gamma2': gamma8_2, 'beta2': beta8_2}
        B8 = nn.ResBlock(B7, pamas8)
        self.graph.add_block(B8)

        # pool9
        pool9 = nn.Op(nn.MaxPooling(2,2), B8)
        self.graph.add_layer(pool9)

        # flatten
        fla = nn.Op(nn.Flatten(), pool9)
        self.graph.add_op(fla)

        # Fc10
        WFC10 = nn.Variable(nn.UniformInit([flatten_size, num_labels]), lr=lr)
        self.graph.add_var(WFC10)
        bFC10 = nn.Variable(nn.UniformInit([num_labels, 1]), lr=lr)
        self.graph.add_var(bFC10)
        FC10 = nn.Op(nn.Dot(), fla, [WFC10, bFC10])
        self.graph.add_op(FC10)

        # loss and optimizer
        self.graph.add_loss(nn.Loss(nn.Softmax()))
        self.graph.add_optimizer((nn.AdamOptimizer()))

    def train(self, X_train, Y_train, X_test, Y_test, \
              epochs, batchsize=10):
        for epoch in range(epochs):
            batch_tr = nn.miniBatch(X_train, Y_train, batchsize)
            batch_te = nn.miniBatch(X_test, Y_test, batch_size=batchsize)
            for i, (batch_x, batch_y) in enumerate(batch_tr):
                self.graph.forward(batch_x)
                self.graph.calc_loss(batch_y)
                self.graph.backward()
                self.graph.update()
                if i % 1 == 0:
                    print('epoch:{}/{}, batch:{}/{}, train loss:{}'.format(epoch, epochs, i, len(batch_tr), self.graph.loss))
                if i % 20 == 0:
                    accuracy = self.graph.accuracy(batch_te, batchs=10)
                    print('epoch:{}, accuracy:{}'.format(epoch, accuracy))
                if i % 1000 == 0 and i != 0:
                    accuracy = self.graph.accuracy(batch_te)
                    print('epoch:{}, accuracy:{}'.format(epoch, accuracy))


