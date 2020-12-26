# coding=utf-8
# ================================================================
#
#   Author      : LuoDeng
#   Created date: 2020-12-24 22:55:44
#   Description : FPN Neck
#
# ================================================================

import tensorflow as tf
import numpy as np
import tensorflow.keras.layers as layers
from tensorflow.python.ops.control_flow_ops import group
from model.custom_layers import Resize, GroupNormalization
import tensorflow_addons as tfa

def concat_coord(x):
    ins_feat = x
    
    batch_size = tf.shape(input=x)[0]
    h = tf.shape(input=x)[1]
    w = tf.shape(input=x)[2]
    float_h, float_w = tf.cast(h, tf.float32), tf.cast(w, tf.float32)

    y_range = tf.range(float_h, dtype=tf.float32)     # [h, ]
    # conver to [-1, 1]
    y_range = 2.0 * y_range / (float_h - 1.0) - 1.0
    x_range = tf.range(float_w, dtype=tf.float32)     # [w, ]
    x_range = 2.0 * x_range / (float_w - 1.0) - 1.0
    x_range = x_range[tf.newaxis, :]   # [1, w]
    y_range = y_range[:, tf.newaxis]   # [h, 1]
    x = tf.tile(x_range, [h, 1])     # [h, w]
    y = tf.tile(y_range, [1, w])     # [h, w]

    x = x[tf.newaxis, :, :, tf.newaxis]   # [1, h, w, 1]
    y = y[tf.newaxis, :, :, tf.newaxis]   # [1, h, w, 1]
    x = tf.tile(x, [batch_size, 1, 1, 1])   # [N, h, w, 1]
    y = tf.tile(y, [batch_size, 1, 1, 1])   # [N, h, w, 1]

    ins_feat_x = tf.concat([ins_feat, x], axis=-1)   # [N, h, w, c+1]
    ins_feat_y = tf.concat([ins_feat, y], axis=-1)   # [N, h, w, c+1]

    return [ins_feat_x, ins_feat_y]


def points_nms(heat, kernel=2):
    # kernel must be 2
    hmax = layers.MaxPool2D(pool_size=(kernel, kernel), strides=(1, 1), padding='same')(heat)
    keep = tf.cast(tf.equal(hmax, heat), tf.float32)
    return keep * heat
    

def matrix_nms(seg_masks, cate_labels, cate_scores, kernel='gaussian', sigma=2.0, sum_masks=None):
    """Matrix NMS for multi-class masks.

    Args:
        seg_masks (Tensor): shape (n, h, w)   True、False组成的掩码
        cate_labels (Tensor): shape (n), mask labels in descending order
        cate_scores (Tensor): shape (n), mask scores in descending order
        kernel (str):  'linear' or 'gauss'
        sigma (float): std in gaussian method
        sum_masks (Tensor):  shape (n, )      n个物体的面积

    Returns:
        Tensor: cate_scores_update, tensors of shape (n)
    """
    n_samples = tf.shape(input=cate_labels)[0]   # 物体数
    seg_masks = tf.reshape(tf.cast(seg_masks, tf.float32), (n_samples, -1))   # [n, h*w]  掩码由True、False变成0、1
    # inter.
    inter_matrix = tf.matmul(seg_masks, seg_masks, transpose_b=True)   # [n, n] 自己乘以自己的转置。两两之间的交集面积。
    # union.
    sum_masks_x = tf.tile(sum_masks[tf.newaxis, :], [n_samples, 1])     # [n, n]  sum_masks重复了n行得到sum_masks_x
    # iou.
    iou_matrix = inter_matrix / (sum_masks_x + tf.transpose(a=sum_masks_x, perm=[1, 0]) - inter_matrix)
    rows = tf.range(0, n_samples, 1, 'int32')
    cols = tf.range(0, n_samples, 1, 'int32')
    rows = tf.tile(tf.reshape(rows, (1, -1)), [n_samples, 1])
    cols = tf.tile(tf.reshape(cols, (-1, 1)), [1, n_samples])
    tri_mask = tf.cast(rows > cols, 'float32')
    iou_matrix = tri_mask * iou_matrix   # [n, n]   只取上三角部分

    # label_specific matrix.
    cate_labels_x = tf.tile(cate_labels[tf.newaxis, :], [n_samples, 1])     # [n, n]  cate_labels重复了n行得到cate_labels_x
    label_matrix = tf.cast(tf.equal(cate_labels_x, tf.transpose(a=cate_labels_x, perm=[1, 0])), tf.float32)
    label_matrix = tri_mask * label_matrix   # [n, n]   只取上三角部分

    # IoU compensation
    compensate_iou = tf.reduce_max(input_tensor=iou_matrix * label_matrix, axis=0)
    compensate_iou = tf.tile(compensate_iou[tf.newaxis, :], [n_samples, 1])     # [n, n]
    compensate_iou = tf.transpose(a=compensate_iou, perm=[1, 0])      # [n, n]

    # IoU decay
    decay_iou = iou_matrix * label_matrix

    # # matrix nms
    if kernel == 'gaussian':
        decay_matrix = tf.exp(-1 * sigma * (decay_iou ** 2))
        compensate_matrix = tf.exp(-1 * sigma * (compensate_iou ** 2))
        decay_coefficient = tf.reduce_min(input_tensor=(decay_matrix / compensate_matrix), axis=0)
    elif kernel == 'linear':
        decay_matrix = (1-decay_iou)/(1-compensate_iou)
        decay_coefficient = tf.reduce_min(input_tensor=decay_matrix, axis=0)
    else:
        raise NotImplementedError

    # update the score.
    cate_scores_update = cate_scores * decay_coefficient
    return cate_scores_update



class DecoupledSOLOHead(object):
    def __init__(
        self,
        num_classes=80,
        in_channels=256,
        seg_feat_channels=256,
        stacked_convs=7,
        strides=[8, 8, 16, 32, 32],
        base_edge_list=(16, 32, 64, 128, 256),
        scale_ranges=((1, 96), (48, 192), (96, 384), (192, 768), (384, 2048)),
        sigma=0.2,
        num_grids=[40, 36, 24, 16, 12],
        cate_down_pos=0,
        with_deform=False,
        loss_ins=None,
        loss_cate=None
        ):
        self.num_classes = num_classes
        self.seg_num_grids = num_grids
        self.in_channels = in_channels
        self.seg_feat_channels = seg_feat_channels
        self.stacked_convs = stacked_convs
        self.strides = strides
        self.sigma = sigma
        self.cate_down_pos = cate_down_pos
        self.base_edge_list = base_edge_list
        self.scale_ranges = scale_ranges
        self.with_deform = with_deform

        self._init_layers()

    def _init_layers(self):
        norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)
        self.ins_convs_x = []
        self.ins_convs_y = []
        self.cate_convs = []

        # repeat 7 times
        for i in range(self.stacked_convs):
            conv2d_1 = layers.Conv2D(self.seg_feat_channels, 3, padding='same', strides=1, use_bias=False, data_format='channels_last')
            # gn_1 = GroupNormalization(num_groups=32)
            gn_1 = tfa.layers.GroupNormalization(groups=32, axis=3)
            relu_1 = layers.ReLU()
            self.ins_convs_x.append(conv2d_1)
            self.ins_convs_x.append(gn_1)
            self.ins_convs_x.append(relu_1)

            conv2d_2 = layers.Conv2D(self.seg_feat_channels, 3, padding='same', strides=1, use_bias=False, data_format='channels_last')
            # gn_2 = GroupNormalization(num_groups=32)
            gn_2 = tfa.layers.GroupNormalization(groups=32, axis=3)
            relu_2 = layers.ReLU()
            self.ins_convs_y.append(conv2d_2)
            self.ins_convs_y.append(gn_2)
            self.ins_convs_y.append(relu_2)

            conv2d_3 = layers.Conv2D(self.seg_feat_channels, 3, padding='same', strides=1, use_bias=False, data_format='channels_last')
            # gn_3 = GroupNormalization(num_groups=32)
            gn_3 = tfa.layers.GroupNormalization(groups=32, axis=3)
            relu_3 = layers.ReLU()
            self.cate_convs.append(conv2d_3)
            self.cate_convs.append(gn_3)
            self.cate_convs.append(relu_3)
        
        self.dsolo_ins_list_x = []
        self.dsolo_ins_list_y = []
        # final conv output channles [40, 36, 24, 16, 12]
        for seg_num_grid in self.seg_num_grids:
            conv2d_1 = layers.Conv2D(seg_num_grid, 3, padding='same', strides=1, use_bias=True)
            self.dsolo_ins_list_x.append(conv2d_1)
            conv2d_2 = layers.Conv2D(seg_num_grid, 3, padding='same', strides=1, use_bias=True)
            self.dsolo_ins_list_y.append(conv2d_2)
        # category
        self.dsolo_cate = layers.Conv2D(self.num_classes, 3, padding='same', strides=1, use_bias=True)
    
    def __call__(self, feats, cfg, eval):
        # DecoupledSOLOHead都是这样，一定有5个张量，5个张量的strides=[8, 8, 16, 32, 32]，所以先对首尾张量进行插值
        new_feats = [Resize(tf.shape(input=feats[1])[1], tf.shape(input=feats[1])[2], 'BILINEAR')(feats[0]),
                     feats[1],
                     feats[2],
                     feats[3],
                     Resize(tf.shape(input=feats[3])[1], tf.shape(input=feats[3])[2], 'BILINEAR')(feats[4])]
        featmap_sizes = [tf.shape(input=featmap)[1:3] for featmap in new_feats]
        upsampled_size = (featmap_sizes[0][0] * 2, featmap_sizes[0][1] * 2)   # stride=4

        ins_pred_x_list, ins_pred_y_list, cate_pred_list = [], [], []
        for idx in range(len(self.seg_num_grids)):
            ins_feat = new_feats[idx]   # 给掩码分支
            cate_feat = new_feats[idx]  # 给分类分支

            # ============ ins branch (掩码分支，特征图形状是[N, mask_h, mask_w, grid]) ============
            ins_feat_x, ins_feat_y = layers.Lambda(concat_coord)(ins_feat)   # [N, h, w, c+1]、 [N, h, w, c+1]

            for ins_layer_x, ins_layer_y in zip(self.ins_convs_x, self.ins_convs_y):
                ins_feat_x = ins_layer_x(ins_feat_x)   # [N, h, w, 256]
                ins_feat_y = ins_layer_y(ins_feat_y)   # [N, h, w, 256]
            
            ins_feat_x = layers.UpSampling2D(2, interpolation='bilinear')(ins_feat_x)   # [N, 2*h, 2*w, 256]
            ins_feat_y = layers.UpSampling2D(2, interpolation='bilinear')(ins_feat_y)   # [N, 2*h, 2*w, 256]

            ins_pred_x = self.dsolo_ins_list_x[idx](ins_feat_x)   # [N, 2*h, 2*w, grid]，即[N, mask_h, mask_w, grid]
            ins_pred_y = self.dsolo_ins_list_y[idx](ins_feat_y)   # [N, 2*h, 2*w, grid]，即[N, mask_h, mask_w, grid]
            # 若输入图片大小为416x416，那么new_feats里图片大小应该为[52, 52, 26, 13, 13]，因为strides=[8, 8, 16, 32, 32]。
            # 那么对应的ins_pred_x大小应该为[104, 104, 52, 26, 26]
            # 那么对应的ins_pred_y大小应该为[104, 104, 52, 26, 26]

            # ============ cate branch (分类分支，特征图形状是[N, grid, grid, num_classes=80]) ============
            for i, cate_layer in enumerate(self.cate_convs):
                if i == self.cate_down_pos:   # 第0次都要插值成seg_num_grid x seg_num_grid的大小。
                    seg_num_grid = self.seg_num_grids[idx]
                    cate_feat = Resize(seg_num_grid, seg_num_grid, 'BILINEAR')(cate_feat)
                cate_feat = cate_layer(cate_feat)

            cate_pred = self.dsolo_cate(cate_feat)   # 种类分支，通道数变成了80，[N, grid, grid, 80]

            # ============ 是否是预测状态 ============
            if eval:
                ins_pred_x = layers.Activation(tf.nn.sigmoid)(ins_pred_x)
                ins_pred_x = Resize(upsampled_size[0], upsampled_size[1], 'BILINEAR')(ins_pred_x)

                ins_pred_y = layers.Activation(tf.nn.sigmoid)(ins_pred_y)
                ins_pred_y = Resize(upsampled_size[0], upsampled_size[1], 'BILINEAR')(ins_pred_y)
                # 若输入图片大小为416x416，那么new_feats里图片大小应该为[52, 52, 26, 13, 13]，因为strides=[8, 8, 16, 32, 32]。
                # 那么此处的5个ins_pred_x大小应该为[104, 104, 104, 104, 104]；
                # 那么此处的5个ins_pred_y大小应该为[104, 104, 104, 104, 104]。即stride=4。训练时不会执行这里。
                cate_pred = layers.Activation(tf.nn.sigmoid)(cate_pred)
                cate_pred = layers.Lambda(points_nms)(cate_pred)
            ins_pred_x_list.append(ins_pred_x)
            ins_pred_y_list.append(ins_pred_y)
            cate_pred_list.append(cate_pred)
        if eval:
            num_layers = 5
            def output_layer(args):
                p = 0
                pred_mask_x = []
                for i in range(num_layers):
                    mask_x = args[p]   # 从小感受野 到 大感受野 （从多格子 到 少格子）
                    mask_x = tf.transpose(a=mask_x, perm=[0, 3, 1, 2])
                    pred_mask_x.append(mask_x)
                    p += 1
                pred_mask_x = tf.concat(pred_mask_x, axis=1)

                pred_mask_y = []
                for i in range(num_layers):
                    mask_y = args[p]   # 从小感受野 到 大感受野 （从多格子 到 少格子）
                    mask_y = tf.transpose(a=mask_y, perm=[0, 3, 1, 2])
                    pred_mask_y.append(mask_y)
                    p += 1
                pred_mask_y = tf.concat(pred_mask_y, axis=1)

                pred_cate = []
                for i in range(num_layers):
                    c = args[p]   # 从小感受野 到 大感受野 （从多格子 到 少格子）
                    c = tf.reshape(c, (1, -1, self.num_classes))
                    pred_cate.append(c)
                    p += 1
                pred_cate = tf.concat(pred_cate, axis=1)

                o = self.get_seg_single(pred_cate[0],
                       pred_mask_x[0],
                       pred_mask_y[0],
                       upsampled_size,
                       upsampled_size,
                       cfg)
                return o

            output = layers.Lambda(output_layer)([*ins_pred_x_list, *ins_pred_y_list, *cate_pred_list])
            return output
        return ins_pred_x_list + ins_pred_y_list + cate_pred_list

    def get_seg_single(
        self,
        cate_preds,
        seg_preds_x,
        seg_preds_y,
        featmap_size,
        ori_shape,
        cfg
        ):
        '''
        Args:
            cate_preds:    同一张图片5个输出层的输出汇合  [40*40+36*36+24*24+16*16+12*12, 80]
            seg_preds_x:   同一张图片5个输出层的输出汇合  [40+36+24+16+12, 104, 104]
            seg_preds_y:   同一张图片5个输出层的输出汇合  [40+36+24+16+12, 104, 104]
            featmap_size:  [s4, s4]        一维张量  1-D Tensor
            img_shape:     [800, 1216, 3]  一维张量  1-D Tensor
            ori_shape:     [427, 640, 3]   一维张量  1-D Tensor
            scale_factor:  800/427
            cfg:
            rescale:
            debug:

        Returns:
        '''
        # trans trans_diff.
        seg_num_grids = tf.zeros([len(self.seg_num_grids), ], tf.int32) + np.array(self.seg_num_grids)
        trans_size = tf.cumsum(tf.pow(seg_num_grids, 2))
        seg_size = tf.cumsum(seg_num_grids)    # [40, 40+36, 40+36+24, ...]

        trans_diff = []
        seg_diff = []
        num_grids = []
        strides = []
        n_stage = len(self.seg_num_grids)   # 5个输出层

        for ind_ in range(n_stage):
            if ind_ == 0:
                # 第0个输出层的分类分支在cate_preds中的偏移是0
                trans_diff_ = tf.zeros([self.seg_num_grids[ind_] ** 2, ], tf.int32)
                # 第0个输出层的掩码分支在seg_preds_x中的偏移是0
                seg_diff_ = tf.zeros([self.seg_num_grids[ind_] ** 2, ], tf.int32)
            else:
                # 第1个输出层的分类分支在cate_preds中的偏移是40*40，第2个输出层的分类分支在cate_preds中的偏移是40*40+36*36，...
                trans_diff_ = tf.zeros([self.seg_num_grids[ind_] ** 2, ], tf.int32) + trans_size[ind_ - 1]
                # 第0个输出层的掩码分支在seg_preds_x中的偏移是40，第0个输出层的掩码分支在seg_preds_x中的偏移是40+36，...
                seg_diff_ = tf.zeros([self.seg_num_grids[ind_] ** 2, ], tf.int32) + seg_size[ind_ - 1]
            # 第0个输出层的一行（或一列）的num_grids是40，第1个输出层的一行（或一列）的num_grids是36，...
            num_grids_ = tf.zeros([self.seg_num_grids[ind_] ** 2, ], tf.int32) + self.seg_num_grids[ind_]
            # 第0个输出层的stride是8，第1个输出层的stride是8，...
            strides_ = tf.zeros([self.seg_num_grids[ind_] ** 2, ], tf.float32) + float(self.strides[ind_])

            trans_diff.append(trans_diff_)
            seg_diff.append(seg_diff_)
            num_grids.append(num_grids_)
            strides.append(strides_)
        trans_diff = tf.concat(trans_diff, axis=0)   # [3872, ]
        seg_diff = tf.concat(seg_diff, axis=0)       # [3872, ]
        num_grids = tf.concat(num_grids, axis=0)     # [3872, ]
        strides = tf.concat(strides, axis=0)         # [3872, ]

        # process. 处理。
        inds = tf.where(cate_preds > cfg.score_thr)   # [[3623, 17], [3623, 60], [3639, 17], ...]   分数超过阈值的物体所在格子
        cate_scores = tf.gather_nd(cate_preds, inds)

        trans_diff = tf.gather(trans_diff, inds[:, 0])   # [3472, 3472, 3472, ...]   格子所在输出层的分类分支在cate_preds中的偏移
        seg_diff = tf.gather(seg_diff, inds[:, 0])       # [100, 100, 100, ...]      格子所在输出层的掩码分支在seg_preds_x中的偏移
        num_grids = tf.gather(num_grids, inds[:, 0])     # [16, 16, 16, ...]         格子所在输出层每一行有多少个格子
        strides = tf.gather(strides, inds[:, 0])         # [32, 32, 32, ...]         格子所在输出层的stride

        loc = tf.cast(inds[:, 0], tf.int32)
        y_inds = (loc - trans_diff) // num_grids   # 格子行号
        x_inds = (loc - trans_diff) % num_grids    # 格子列号
        y_inds += seg_diff   # 格子行号在seg_preds_y中的绝对位置
        x_inds += seg_diff   # 格子列号在seg_preds_x中的绝对位置

        cate_labels = inds[:, 1]   # 类别
        mask_x = tf.gather(seg_preds_x, x_inds)   # [11, s4, s4]
        mask_y = tf.gather(seg_preds_y, y_inds)   # [11, s4, s4]
        seg_masks_soft = mask_x * mask_y    # [11, s4, s4]  物体的mask，逐元素相乘得到
        seg_masks = seg_masks_soft > cfg.mask_thr
        sum_masks = tf.reduce_sum(input_tensor=tf.cast(seg_masks, tf.float32), axis=[1, 2])   # [11, ]  11个物体的面积
        keep = tf.compat.v1.where(sum_masks > strides)   # 面积大于这一层的stride才保留

        seg_masks_soft = tf.gather_nd(seg_masks_soft, keep)   # 用概率表示的掩码
        seg_masks = tf.gather_nd(seg_masks, keep)             # 用True、False表示的掩码
        cate_scores = tf.gather_nd(cate_scores, keep)    # 类别得分
        sum_masks = tf.gather_nd(sum_masks, keep)        # 面积
        cate_labels = tf.gather_nd(cate_labels, keep)    # 类别
        # mask scoring   是1的像素的 概率总和 占 面积（是1的像素数） 的比重
        seg_score = tf.reduce_sum(input_tensor=seg_masks_soft * tf.cast(seg_masks, tf.float32), axis=[1, 2]) / sum_masks
        cate_scores *= seg_score   # 类别得分乘上这个比重得到新的类别得分。因为有了mask scoring机制，所以分数一般比其它算法如yolact少。

        def exist_objs_1(cate_scores, seg_masks_soft, seg_masks, sum_masks, cate_labels):
            # sort and keep top nms_pre
            k = tf.shape(input=cate_scores)[0]
            _, sort_inds = tf.nn.top_k(cate_scores, k=k, sorted=True)   # [7, 5, 8, ...] 降序。最大值的下标，第2大值的下标，...
            sort_inds = sort_inds[:cfg.nms_pre]   # 最多cfg.nms_pre个。
            seg_masks_soft = tf.gather(seg_masks_soft, sort_inds)   # 按照分数降序
            seg_masks = tf.gather(seg_masks, sort_inds)             # 按照分数降序
            cate_scores = tf.gather(cate_scores, sort_inds)
            sum_masks = tf.gather(sum_masks, sort_inds)
            cate_labels = tf.gather(cate_labels, sort_inds)

            # Matrix NMS
            cate_scores = matrix_nms(seg_masks, cate_labels, cate_scores,
                                     kernel=cfg.kernel, sigma=cfg.sigma, sum_masks=sum_masks)

            keep = tf.compat.v1.where(cate_scores > cfg.update_thr)   # 大于第二个分数阈值才保留
            keep = tf.reshape(keep, (-1, ))
            seg_masks_soft = tf.gather(seg_masks_soft, keep)
            cate_scores = tf.gather(cate_scores, keep)
            cate_labels = tf.gather(cate_labels, keep)

            def exist_objs_2(cate_scores, seg_masks_soft, cate_labels):
                # sort and keep top_k
                k = tf.shape(input=cate_scores)[0]
                _, sort_inds = tf.nn.top_k(cate_scores, k=k, sorted=True)
                sort_inds = sort_inds[:cfg.max_per_img]
                seg_masks_soft = tf.gather(seg_masks_soft, sort_inds)
                cate_scores = tf.gather(cate_scores, sort_inds)
                cate_labels = tf.gather(cate_labels, sort_inds)

                # 插值前处理
                seg_masks_soft = tf.transpose(a=seg_masks_soft, perm=[1, 2, 0])
                seg_masks_soft = seg_masks_soft[tf.newaxis, :, :, :]

                # seg_masks_soft = tf.image.resize_images(seg_masks_soft, tf.convert_to_tensor([featmap_size[0] * 4, featmap_size[1] * 4]), method=tf.image.ResizeMethod.BILINEAR)
                # seg_masks = tf.image.resize_images(seg_masks_soft, tf.convert_to_tensor([ori_shape[0], ori_shape[1]]), method=tf.image.ResizeMethod.BILINEAR)

                seg_masks_soft = tf.image.resize(seg_masks_soft, [featmap_size[0] * 4, featmap_size[1] * 4], method=tf.image.ResizeMethod.BILINEAR)
                seg_masks = tf.image.resize(seg_masks_soft, [ori_shape[0], ori_shape[1]], method=tf.image.ResizeMethod.BILINEAR)

                # 插值后处理
                seg_masks = tf.transpose(a=seg_masks, perm=[0, 3, 1, 2])
                seg_masks = tf.cast(seg_masks > cfg.mask_thr, tf.float32)
                cate_labels = tf.reshape(cate_labels, (1, -1))
                cate_scores = tf.reshape(cate_scores, (1, -1))
                return seg_masks, cate_labels, cate_scores

            def no_objs_2():
                seg_masks = tf.zeros([1, 1, 1, 1], tf.float32) - 1.0
                cate_labels = tf.zeros([1, 1], tf.int64) - 1
                cate_scores = tf.zeros([1, 1], tf.float32) - 1.0
                return seg_masks, cate_labels, cate_scores
            
            # 是否有物体
            seg_masks, cate_labels, cate_scores = tf.cond(pred=tf.equal(tf.shape(input=cate_scores)[0], 0),
                                                          true_fn=no_objs_2,
                                                          false_fn=lambda: exist_objs_2(cate_scores, seg_masks_soft, cate_labels))
            return seg_masks, cate_labels, cate_scores
        
        def no_objs_1():
            seg_masks = tf.zeros([1, 1, 1, 1], tf.float32) - 1.0
            cate_labels = tf.zeros([1, 1], tf.int64) - 1
            cate_scores = tf.zeros([1, 1], tf.float32) - 1.0
            return seg_masks, cate_labels, cate_scores
        
        seg_masks, cate_labels, cate_scores = tf.cond(pred=tf.equal(tf.shape(input=cate_scores)[0], 0),
                                                      true_fn=no_objs_1,
                                                      false_fn=lambda: exist_objs_1(cate_scores, seg_masks_soft, seg_masks, sum_masks, cate_labels))
        return [seg_masks, cate_labels, cate_scores]
