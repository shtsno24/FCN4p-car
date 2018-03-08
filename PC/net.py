#-*- coding:utf-8 -*-
import chainer
import chainer.functions as F
import chainer.links as L


class MLP(chainer.Chain):

    def __init__(self):
        super(MLP, self).__init__(conv1=L.DepthwiseConvolution2D(1, 4, (3,8), stride=(3,8)),
            conv2=L.Convolution2D(None, 4, 4, stride=2),
            conv3=L.Convolution2D(None, 16, 3, stride=2),

            deconv3 = L.Deconvolution2D(None,8,5,stride=5),
            deconv1 = L.Deconvolution2D(None,1,(3,8),stride=(3,8)))
                       

    def __call__(self, x):
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))

        h = F.relu(self.deconv3(h))
        h = F.relu(self.deconv1(h))
        return h
