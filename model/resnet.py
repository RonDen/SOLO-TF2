# coding=utf-8
# ================================================================
#
#   Author      : LuoDeng
#   Created date: 2020-12-24 22:02:33
#   Description : ResNet Backbone，R50 and R101 Implemented
#
# ================================================================

import tensorflow as tf
import tensorflow.keras.layers as layers
from tensorflow.python.keras import activations
from model.custom_layers import Conv2dUnit, Conv3x3


class ConvBlock(object):
    '''
    官方SOLO仓库中，下采样是在中间的3x3卷积层进行，不同于keras的resnet。
    与IdentityBlock的区别是在跳连中加了一个1x1卷积
    一个Block内部有3个卷积层，在一个3x3卷积之前加一个1x1卷积，在条连相加之后再激活
    '''
    def __init__(self, filters, use_dcn=False, stride=2):
        super().__init__()
        super(ConvBlock, self).__init__()
        filters1, filters2, filters3 = filters

        # 1x1 conv
        # 3x3 conv bn relu
        # 1x1 conv
        self.conv1 = Conv2dUnit(filters1, 1, strides=1, padding='valid', use_bias=False, bn=True, activation='relu')
        self.conv2 = Conv3x3(filters2, stride, use_dcn)
        self.conv3 = Conv2dUnit(filters3, 1, strides=1, padding='valid', use_bias=False, bn=True, activation=None)

        self.conv4 = Conv2dUnit(filters3, 1, strides=stride, padding='valid', use_bias=False, bn=True, activation=None)
        self.act = layers.ReLU()

    def __call__(self, input_tensor):
        x = self.conv1(input_tensor)
        x = self.conv2(x)
        x = self.conv3(x)
        shortcut = self.conv4(input_tensor)
        x = layers.add([x, shortcut])
        x = self.act(x)
        return x


class IdentityBlock(object):
    """
    IdentityBlock
    经典的ResNet中的IdentityBlock，不过区别是跳连在Block内部
    一个Block内部有3个卷积层，在一个3x3卷积之前加一个1x1卷积，在条连相加之后再激活
    """
    def __init__(self, filters, use_dcn=False):
        super(IdentityBlock, self).__init__()
        filters1, filters2, filters3 = filters

        self.conv1 = Conv2dUnit(filters1, 1, strides=1, padding='valid', use_bias=False, bn=True, activation='relu')
        self.conv2 = Conv3x3(filters2, 1, use_dcn)
        self.conv3 = Conv2dUnit(filters3, 1, strides=1, padding='valid', use_bias=False, bn=True, activation=None)

        self.act = layers.ReLU()

    def __call__(self, input_tensor):
        x = self.conv1(input_tensor)
        x = self.conv2(x)
        x = self.conv3(x)
        x = layers.add([x, input_tensor])
        x = self.act(x)
        return x


class ResNet(object):
    def __init__(self, depth, use_dcn=False):
        super(ResNet, self).__init__()
        assert depth in [50, 101]
        # stage1
        # Image size down 2
        self.conv1 = Conv2dUnit(64, 7, strides=2, padding='same', use_bias=False, bn=True, activation='relu')
        # Image size down 2
        self.maxpool = layers.MaxPool2D(pool_size=3, strides=2, padding='same')

        # stage2
        self.stage2_0 = ConvBlock([64, 64, 256], stride=1)
        self.stage2_1 = IdentityBlock([64, 64, 256])
        self.stage2_2 = IdentityBlock([64, 64, 256])

        # stage3
        self.stage3_0 = ConvBlock([128, 128, 512], stride=1)
        self.stage3_1 = IdentityBlock([128, 128, 512], use_dcn=use_dcn)
        self.stage3_2 = IdentityBlock([128, 128, 512], use_dcn=use_dcn)
        self.stage3_3 = IdentityBlock([128, 128, 512], use_dcn=use_dcn)

        # stage4
        self.stage4_0 = ConvBlock([256, 256, 1024], use_dcn=use_dcn)
        k = 21
        if depth == 50:
            k = 4
        self.stage4_layers = []
        for i in range(k):
            ly = IdentityBlock([256, 256, 1024], use_dcn=use_dcn)
            self.stage4_layers.append(ly)
        self.stage4_last_layer = IdentityBlock([256, 256, 1024], use_dcn=use_dcn)

        # stage5
        self.stage5_0 = ConvBlock([512, 512, 2048], use_dcn=use_dcn)
        self.stage5_1 = IdentityBlock([512, 512, 2048], use_dcn=use_dcn)
        self.stage5_2 = IdentityBlock([512, 512, 2048], use_dcn=use_dcn)
    
    def __call__(self, input_tensor):
        # down 4x
        x = self.conv1(input_tensor)
        x = self.maxpool(x)

        # stage2
        x = self.stage2_0(x)
        x = self.stage2_1(x)
        s4 = self.stage2_2(x)
    
        # stage3
        x = self.stage3_0(s4)
        x = self.stage3_1(x)
        x = self.stage3_2(x)
        s8 = self.stage3_3(x)
    
        # stage4
        x = self.stage4_0(s8)
        for ly in self.stage4_layers:
            x = ly(x)
    
        s16 = self.stage4_last_layer(x)
        # stage5
        x = self.stage5_0(s16)
        x = self.stage5_1(x)
        s32 = self.stage5_2(x)
        return [s4, s8, s16, s32]



if __name__ == "__main__":
    x = tf.random.uniform(shape=(1, 416, 416, 3), minval=-1, maxval=1)

    def _test1():
        conv1 = Conv2dUnit(64, 7, strides=2, padding='same', use_bias=False, bn=True, activation='relu')
        maxpool = layers.MaxPool2D(pool_size=3, strides=2, padding='same')
        x = conv1(x)
        print(x.shape)
        x = maxpool(x)
        print(x.shape)
        
        stage2_0 = ConvBlock([64, 64, 256], stride=1)
        x = stage2_0(x)
        print(x.shape)
        stage2_1 = IdentityBlock([64, 64, 256])
        x = stage2_1(x)
        print(x.shape)
        stage2_2 = IdentityBlock([64, 64, 256])
        x = stage2_2(x)
        print(x.shape)

    def _test_resnet50():
        resnet50 = ResNet(50)
        s4, s8, s16, s32 = resnet50(x)
        print(s4.shape, s8.shape, s16.shape, s32.shape)

    def _test_resnet101():
        resnet101 = ResNet(101)
        s4, s8, s16, s32 = resnet101(x)
        print(s4.shape, s8.shape, s16.shape, s32.shape)
    
    print("-"*10, 'test1', "-"*10)
    _test1()
    print("-"*10, 'test resnet50', "-"*10)
    _test_resnet50()
    print("-"*10, 'test_resnet101', "-"*10)
    _test_resnet101()
