#-*- coding:utf-8 -*-
import chainer
import chainer.functions as F
import chainer.links as L


class MLP(chainer.Chain):

    def __init__(self):
        super(MLP, self).__init__(conv1=L.Convolution2D(1, 10, (3,8), stride=(3,8)),
            conv2=L.Convolution2D(None, 8, (2,2), stride=1),
            conv3=L.Convolution2D(None, 6, (2,2), stride=1),

            deconv2 = L.Deconvolution2D(None,8,3,stride=1),
            deconv1 = L.Deconvolution2D(None,3,(3,8),stride=(3,8)))
                       

    def __call__(self, x):
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))

        h = F.relu(self.deconv2(h))
        h = F.relu(self.deconv1(h))
        return h
