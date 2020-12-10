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


class Conv2dUnit(object):
    def __init__(self, filters, kernel_size, strides=1, padding='valid', use_bias=False, bn=1, activation='relu') -> None:
        super(Conv2dUnit, self).__init__()
        self.conv = layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,

            
        )
    
    def __call__(self):
        pass