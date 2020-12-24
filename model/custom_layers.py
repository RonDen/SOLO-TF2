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
from tensorflow.python.keras.backend import var
from tensorflow.python.keras.layers import Layer


METHOD_BICUBIC = 'BICUBIC'
METHOD_NEAREST_NEIGHBOR = 'NEAREST_NEIGHBOR'
METHOD_BILINEAR = 'BILINEAR'
METHOD_AREA = 'AREA'


class Conv2dUnit(Model):
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
            self.conv2d_unit = Conv2dUnit(filter2, 3, strides=strides, padding='same', use_bias=False, bn=False, activation=None)
        self.bn = layers.BatchNormalization()
        self.act = layers.ReLU()
    
    def __call__(self, x):
        x = self.conv2d_unit(x)
        x = self.bn(x)
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
        shape = (input_shape[-1], )
        self.gamma = self.add_weight(shape=shape, initializer='ones', name='gamma')
        self.beta = self.add_weight(shape=shape, initializer='zeros', name='beta')
    
    def compute_output_shape(self, input_shape):
        return input_shape
    
    def call(self, x):
        N, H, W, C = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]
        # 把同一group的元素融合到一起。IN是GN的特例，当num_groups为c时。
        x_reshape = tf.reshape(x, (N, H * W, C))
        mean = tf.reduce_mean(x_reshape, axis=1, keepdims=True)
        t = tf.square(x_reshape - mean)
        variance = tf.reduce_mean(t, axis=1, keepdims=True)
        std = tf.sqrt(variance + self.epsilon)
        outputs = (x_reshape - mean) / std
        outputs = self.gamma * outputs + self.beta
        outputs = tf.reshape(outputs, (N, H, W, C))
        return outputs


class GroupNormalization(Layer):
    """
    GroupNormalization, output shape (N, H, W, C)
    """
    def __init__(self, num_groups, epsilon=1e-9, **kwargs):
        super(GroupNormalization, self).__init__()
        self.epsilon = epsilon
        self.num_groups = num_groups

    def build(self, input_shape):
        super(GroupNormalization, self).build(input_shape)
        shape = (input_shape[-1], )
        self.gamma = self.add_weight(shape=shape, initializer='ones', name='gamma')
        self.beta = self.add_weight(shape=shape, initializer='zeros', name='beta')
    
    def compute_output_shape(self, input_shape):
        return input_shape
    
    def call(self, x):
        N, H, W, C = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]
        # 把同一group的元素融合到一起。IN是GN的特例，当num_groups为c时。
        x_reshape = tf.reshape(x, (N, -1, self.num_groups))
        mean = tf.reduce_mean(x_reshape, axis=1, keepdims=True)
        t = tf.square(x_reshape - mean)
        variance = tf.reduce_mean(t, axis=1, keepdims=True)
        std = tf.sqrt(variance + self.epsilon)
        outputs = (x_reshape - mean) / std
        outputs = tf.reshape(outputs, (N, H * W, C))
        outputs = self.gamma * outputs + self.beta
        outputs = tf.reshape(outputs, (N, H, W, C))
        return outputs

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


if __name__ == "__main__":
    data1 = tf.random.uniform(shape=(1, 416, 416, 3), minval=-1, maxval=1)
    conv1 = Conv3x3(64, 2, False)
    conv2 = Conv3x3(32, 2, False)

    gn1 = GroupNormalization(4)
    in1 = InstanceNormalization()

    out1 = conv1(data1)
    out1 = gn1(out1)
    out2 = conv2(out1)
    out2 = in1(out2)
    
    print(out2.shape)
