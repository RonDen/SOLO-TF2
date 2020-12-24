# coding=utf-8
# ================================================================
#
#   Author      : LuoDeng
#   Created date: 2020-12-24 22:55:44
#   Description : FPN Neck
#
# ================================================================

def concat_coord(x):
    pass


def points_nms(heat, kernel=2):
    pass


def matrix_nms(heat, kernel=2):
    pass


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
    pass


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
        pass

    def _init_layers(self):
        pass
    
    def __call__(self, feats, cfg, eval):
        pass
    
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
        pass