# coding=utf-8
# ================================================================
#
#   Author      : LuoDeng
#   Created date: 2020-12-10
#   Description : 自定义层
#
# ================================================================

import tensorflow as tf
import tensorflow.keras.layers as layers
from tensorflow.keras import Model
from tensorflow.python.keras.layers import Layer


METHOD_BICUBIC = 'BICUBIC'
METHOD_NEAREST_NEIGHBOR = 'NEAREST_NEIGHBOR'
METHOD_BILINEAR = 'BILINEAR'
METHOD_AREA = 'AREA'


class Conv2dUnit(object):
    """
    Basic Convolution Unit
    """
    def __init__(self, filters, kernel_size, strides=1, padding='valid', use_bias=False, bn=True, activation='relu') -> None:
        super(Conv2dUnit, self).__init__()
        self.conv = layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            use_bias=use_bias,
            activation='linear'
        )
        self.bn = None
        self.activation = None
        if bn:
            self.bn = layers.BatchNormalization()
        if activation == 'relu':
            self.activation = layers.ReLU()
        elif activation == 'leaky_relu':
            self.activation = layers.LeakyReLU(alpha=0.1)
    
    def __call__(self, x):
        x = self.conv(x)
        if self.bn:
            x = self.bn(x)
        if self.activation:
            x = self.activation(x)
        return x


class Conv3x3(object):
    """
    3x3 Convolution with batchnorm, relu activation, same padding
    """
    def __init__(self, filter2, strides, use_dcn):
        super(Conv3x3, self).__init__()
        if use_dcn:
            self.conv2d_unit = None
        else:
            self.conv2d_unit = Conv2dUnit(filter2, 3, strides=strides, padding='same', use_bias=False, bn=0, act=None)
        self.bn = layers.BatchNormalization()
        self.act = layers.ReLU()
    
    def __call__(self, x):
        x = self.conv2d_unit(x)
        x = self.bn
        x = self.act(x)
        return x


class InstanceNormalization(Layer):
    """
    Instance Normalization, output shape(N, H, W, C)
    """
    def __init__(self, epsilon=1e-9, **kwargs):
        super(InstanceNormalization, self).__init__()
        self.epsilon = epsilon
    
    def build(self, input_shape):
        super(InstanceNormalization, self).build(input_shape)
        pass



class Resize(Model):
    def __init__(self, h, w, method):
        super(Resize, self).__init__()
        self.h = h
        self.w = w
        self.method = method
    def __call__(self, x):
        m = tf.image.ResizeMethod.BILINEAR
        if self.method == METHOD_BILINEAR:
            m = tf.image.ResizeMethod.BILINEAR
        elif self.method == METHOD_BICUBIC:
            m = tf.image.ResizeMethod.BICUBIC
        elif self.method == METHOD_NEAREST_NEIGHBOR:
            m = tf.image.ResizeMethod.NEAREST_NEIGHBOR
        elif self.method == METHOD_AREA:
            m = tf.image.ResizeMethod.AREA
        a = tf.image.resize(x, [self.h, self.w], method=m)
        return a
