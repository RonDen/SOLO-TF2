#! /usr/bin/env python
# coding=utf-8
# ================================================================
#
#   Author      : LuoDeng
#   Created date: 2020-12-26 14:37:52
#   Description : 复制权重，将Pytorch模型权重转换为Keras模型可以读取的
#
# ================================================================
import torch
import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow.keras.models as models

from model.head import DecoupledSOLOHead
from model.fpn import FPN
from model.resnet import Resnet
from model.solo import SOLO

WEIGHT_ROOT = 'weights/'
NAME = 'Decoupled_SOLO_R50_1x'

def load_weights(path):
    """ Loads weights from a compressed save file. """
    # state_dict = torch.load(path)
    state_dict = torch.load(path, map_location=torch.device('cpu'))
    return state_dict

state_dict = load_weights(WEIGHT_ROOT + NAME + '.pth')

state_dict = state_dict['state_dict']

tracked_dic = {}
backbone_dic = {}
neck_dic = {}
bbox_head_dic = {}
others = {}
for key, value in state_dict.items():
    if 'tracked' in key:
        tracked_dic[key] = value.data.numpy()
        continue
    if 'backbone' in key:
        backbone_dic[key] = value.data.numpy()
        continue
    if 'neck' in key:
        neck_dic[key] = value.data.numpy()
        continue
    if 'bbox_head' in key:
        bbox_head_dic[key] = value.data.numpy()
        continue
    others[key] = value.data.numpy()


print('============================================================')


# Resnet50中，stage1有1个卷积层，其余4个stage分别有3、4、6、3个残差块。每个stage有1个conv_block，其余为identity_block。
# conv_block有4个卷积层，identity_block有3个卷积层，所以Resnet50有1+(1*4+2*3)+(1*4+3*3)+(1*4+5*3)+(1*4+2*3)=1+10+13+19+10=53个卷积层。
# 同理，Resnet101有1+(1*4+2*3)+(1*4+3*3)+(1*4+22*3)+(1*4+2*3)=1+10+13+70+10=104个卷积层。


backbone_map = {}

# stage1
backbone_map['conv2d_1'] = 'backbone.conv1'
backbone_map['batch_normalization_1'] = 'backbone.bn1'

conv_id = 2

# stage2
for block_id in range(3):
    for block_conv_id in range(3):
        backbone_map['conv2d_%d' % conv_id] = 'backbone.layer1.%d.conv%d' % (block_id, block_conv_id+1)
        backbone_map['batch_normalization_%d' % conv_id] = 'backbone.layer1.%d.bn%d' % (block_id, block_conv_id+1)
        conv_id += 1
    if block_id == 0:
        backbone_map['conv2d_%d' % conv_id] = 'backbone.layer1.0.downsample.0'
        backbone_map['batch_normalization_%d' % conv_id] = 'backbone.layer1.0.downsample.1'
        conv_id += 1

# stage3
for block_id in range(4):
    for block_conv_id in range(3):
        backbone_map['conv2d_%d' % conv_id] = 'backbone.layer2.%d.conv%d' % (block_id, block_conv_id+1)
        backbone_map['batch_normalization_%d' % conv_id] = 'backbone.layer2.%d.bn%d' % (block_id, block_conv_id+1)
        conv_id += 1
    if block_id == 0:
        backbone_map['conv2d_%d' % conv_id] = 'backbone.layer2.0.downsample.0'
        backbone_map['batch_normalization_%d' % conv_id] = 'backbone.layer2.0.downsample.1'
        conv_id += 1

# stage4
for block_id in range(6):
    for block_conv_id in range(3):
        backbone_map['conv2d_%d' % conv_id] = 'backbone.layer3.%d.conv%d' % (block_id, block_conv_id+1)
        backbone_map['batch_normalization_%d' % conv_id] = 'backbone.layer3.%d.bn%d' % (block_id, block_conv_id+1)
        conv_id += 1
    if block_id == 0:
        backbone_map['conv2d_%d' % conv_id] = 'backbone.layer3.0.downsample.0'
        backbone_map['batch_normalization_%d' % conv_id] = 'backbone.layer3.0.downsample.1'
        conv_id += 1

# stage5
for block_id in range(3):
    for block_conv_id in range(3):
        backbone_map['conv2d_%d' % conv_id] = 'backbone.layer4.%d.conv%d' % (block_id, block_conv_id+1)
        backbone_map['batch_normalization_%d' % conv_id] = 'backbone.layer4.%d.bn%d' % (block_id, block_conv_id+1)
        conv_id += 1
    if block_id == 0:
        backbone_map['conv2d_%d' % conv_id] = 'backbone.layer4.0.downsample.0'
        backbone_map['batch_normalization_%d' % conv_id] = 'backbone.layer4.0.downsample.1'
        conv_id += 1

# FPN部分有8个卷积层
neck_map = {}
for k in range(8):
    if k % 2 == 0:
        neck_map['conv2d_%d' % conv_id] = 'neck.lateral_convs.%d.conv' % (k//2)
    else:
        neck_map['conv2d_%d' % conv_id] = 'neck.fpn_convs.%d.conv' % ((k-1)//2)
    conv_id += 1

# head部分
gn_id = 1
bbox_head_map = {}
for k in range(7):   # 卷积没偏移
    bbox_head_map['conv2d_%d' % conv_id] = 'bbox_head.ins_convs_x.%d.conv' % k
    bbox_head_map['group_normalization_%d' % gn_id] = 'bbox_head.ins_convs_x.%d.gn' % k
    conv_id += 1
    gn_id += 1

    bbox_head_map['conv2d_%d' % conv_id] = 'bbox_head.ins_convs_y.%d.conv' % k
    bbox_head_map['group_normalization_%d' % gn_id] = 'bbox_head.ins_convs_y.%d.gn' % k
    conv_id += 1
    gn_id += 1

    bbox_head_map['conv2d_%d' % conv_id] = 'bbox_head.cate_convs.%d.conv' % k
    bbox_head_map['group_normalization_%d' % gn_id] = 'bbox_head.cate_convs.%d.gn' % k
    conv_id += 1
    gn_id += 1

# 卷积有偏移
for k in range(5):
    bbox_head_map['conv2d_%d' % conv_id] = 'bbox_head.dsolo_ins_list_x.%d' % k
    conv_id += 1
    bbox_head_map['conv2d_%d' % conv_id] = 'bbox_head.dsolo_ins_list_y.%d' % k
    conv_id += 1
bbox_head_map['conv2d_%d' % conv_id] = 'bbox_head.dsolo_cate'
conv_id += 1


def find(base_model, conv2d_name, batch_normalization_name):
    i1, i2 = -1, -1
    for i in range(len(base_model.layers)):
        if base_model.layers[i].name == conv2d_name:
            i1 = i
        if base_model.layers[i].name == batch_normalization_name:
            i2 = i
    return i1, i2

def backbone_copy(conv, bn, conv_name, bn_name):
    keyword1 = '%s.weight' % conv_name
    keyword2 = '%s.weight' % bn_name
    keyword3 = '%s.bias' % bn_name
    keyword4 = '%s.running_mean' % bn_name
    keyword5 = '%s.running_var' % bn_name
    for key in state_dict:
        value = state_dict[key].numpy()
        if keyword1 in key:
            w = value
        elif keyword2 in key:
            y = value
        elif keyword3 in key:
            b = value
        elif keyword4 in key:
            m = value
        elif keyword5 in key:
            v = value
    w = w.transpose(2, 3, 1, 0)
    conv.set_weights([w])
    bn.set_weights([y, b, m, v])

def neck_copy(conv, conv_name):
    keyword1 = '%s.weight' % conv_name
    keyword2 = '%s.bias' % conv_name
    for key in state_dict:
        value = state_dict[key].numpy()
        if keyword1 in key:
            w = value
        elif keyword2 in key:
            b = value
    w = w.transpose(2, 3, 1, 0)
    conv.set_weights([w, b])


def head_copy(conv, gn, conv_name, gn_name):
    keyword1 = '%s.weight' % conv_name
    keyword2 = '%s.weight' % gn_name
    keyword3 = '%s.bias' % gn_name
    for key in state_dict:
        value = state_dict[key].numpy()
        if keyword1 in key:
            w = value
        elif keyword2 in key:
            y = value
        elif keyword3 in key:
            b = value
    w = w.transpose(2, 3, 1, 0)
    conv.set_weights([w])
    gn.set_weights([y, b])



inputs = layers.Input(shape=(None, None, 3))
resnet = Resnet(50)
fpn = FPN(in_channels=[256, 512, 1024, 2048], out_channels=256, num_outs=5)
head = DecoupledSOLOHead()
solo = SOLO(resnet, fpn, head)
outs = solo(inputs, None, eval=False)
model = models.Model(inputs=inputs, outputs=outs)
model.summary()

tf.keras.utils.plot_model(model, to_file=NAME + '.png', show_shapes=True)

print('\nCopying...')
for i in range(1, 53+1, 1):
    i1, i2 = find(model, 'conv2d_%d' % i, 'batch_normalization_%d' % i)
    backbone_copy(model.layers[i1], model.layers[i2], backbone_map['conv2d_%d' % i], backbone_map['batch_normalization_%d' % i])

for i in range(54, 62, 1):
    i1, _ = find(model, 'conv2d_%d' % i, 'aaa')
    neck_copy(model.layers[i1], neck_map['conv2d_%d' % i])

gn_id = 1
for i in range(62, 83, 1):
    i1, i2 = find(model, 'conv2d_%d' % i, 'group_normalization_%d' % gn_id)
    head_copy(model.layers[i1], model.layers[i2], bbox_head_map['conv2d_%d' % i], bbox_head_map['group_normalization_%d' % gn_id])
    gn_id += 1

for i in range(83, 94, 1):
    i1, _ = find(model, 'conv2d_%d' % i, 'aaa')
    neck_copy(model.layers[i1], bbox_head_map['conv2d_%d' % i])


model.save(NAME + '.h5')
print('\nDone.')
