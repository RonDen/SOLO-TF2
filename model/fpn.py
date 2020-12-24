# coding=utf-8
# ================================================================
#
#   Author      : LuoDeng
#   Created date: 2020-12-24 22:55:44
#   Description : FPN Neck
#
# ================================================================

import tensorflow as tf
import tensorflow.keras.layers as layers
from model.custom_layers import Conv2dUnit


class FPN(object):
    def __init__(
        self,
        in_channels,
        out_channels,
        num_outs,
        start_level=0,
        end_level=-1,
        add_extra_convs=False,
        extra_conv_on_inputs=True,
        relu_before_extra_convs=False,
        no_norm_on_lateral=False,
        conv_cfg=None,
        norm_cfg=None,
        activation=None
        ):
        super(FPN, self).__init__()

        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.start_level = start_level
        self.end_level= end_level
        self.add_extra_convs = add_extra_convs
        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
            self.backbone_end_level = end_level
        
        # FPN部分有8个卷积层
        self.lateral_convs = []
        self.fpn_convs = []
        for i in range(self.start_level, self.backbone_end_level):
            lateral_conv = Conv2dUnit(out_channels, 1, strides=1, padding='valid', use_bias=True, bn=False, activation=None)
            fpn_conv = Conv2dUnit(out_channels, 3, strides=1, padding='same', use_bias=True, bn=False, activation=None)
            self.lateral_convs.append(lateral_conv)
            self.fpn_convs.append(fpn_conv)

    def __call__(self, xs):
        num_ins = len(xs)
        
        # build laterals
        laterals = []
        for i in range(num_ins):
            x = self.lateral_convs[i](xs[i + self.start_level])
            laterals.append(x)
        
        # build top down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, 1):
            x = layers.UpSampling2D(2)(laterals[i])
            laterals[i - 1] = layers.add([laterals[i - 1], x])
        

        # build outputs
        # part1: from original level
        outs = []
        for i in range(used_backbone_levels):
            x = self.fpn_convs[i](laterals[i])
            outs.append(x)
        # part2: add extra levels
        if self.num_outs > len(outs):
            # use maxpool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    # down 2x
                    x = layers.MaxPooling2D(pool_size=1, strides=2, padding='valid')(outs[-1])
                    outs.append(x)
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                pass
        return outs


if __name__ == "__main__":
    x = tf.random.uniform(shape=(1, 416, 416, 3), minval=-1, maxval=1)

    def _test1():
        from model.resnet import ResNet
        r50 = ResNet(50)
        [s4, s8, s16, s32] = r50(x)
        
    fpn = FPN()
    pass
