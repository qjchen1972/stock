# -*- coding: utf-8 -*-
from __future__ import division
import torch
from torch import nn
import torch.nn.functional as F
from math import floor
from collections import OrderedDict
from math import sqrt

class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features,affine=True)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                        growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate,affine=True)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                        kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class DenseNet(nn.Module):
    r"""Densenet-BC model class, based on

    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    Args:

        growth_rate (int) - how many filters to add each layer (`k` in paper)

        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer

        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """

    def __init__(self, growth_rate=32, block_config=(6,12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=1000):
        super(DenseNet, self).__init__()

        # First convolution
	self.features = nn.Sequential(OrderedDict([
            		('conv0', nn.Conv2d(3, num_init_features, 
			kernel_size=7, stride=2,padding=3, bias=False)),
                        ('norm0', nn.BatchNorm2d(num_init_features)),
                        ('relu0', nn.ReLU(inplace=True)),
                        ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
			]))

        # Each denseblock
	num_features = num_init_features
	for i, num_layers in enumerate(block_config):
	    if num_layers == 0:continue
	    block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
	    self.features.add_module('denseblock%d' % (i + 1), block)
	    num_features = num_features + num_layers * growth_rate
	    if i != len(block_config) - 1:
	        trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
		self.features.add_module('transition%d' % (i + 1), trans)
		num_features = num_features // 2

        # Final batch norm
	self.features.add_module('norm5', nn.BatchNorm2d(num_features))
	self.features.add_module('relu5', nn.ReLU(inplace=True))

        self.vec = nn.Linear(num_features, 32)
        #self.feel = nn.Sequential(nn.Tanh(),nn.Linear(4, 4))
        self.feel = nn.Linear(4, 32)
	#self.classifier = nn.Sequential(nn.Linear(32, num_classes), nn.Sigmoid())
        self.classifier = nn.Linear(64, num_classes)
        
    def forward(self, x,feel):
        features = self.features(x)
	kernelSize = tuple(int(i) for i in features.size()[2:])
	out = F.avg_pool2d(features, kernel_size=kernelSize, stride=1).view(features.size(0), -1)
        out = self.vec(out)
        out = F.softmax(out,1)

        fe = self.feel(feel)
        fe = F.softmax(fe,1)

        #out = out + fe
        #out = F.softmax(out,1)
        out =  torch.cat((out, fe), 1)
	out = self.classifier(out)
	return out



