import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow.keras.models as models

from config import DecoupledSOLO_R50_FPN_Config
from model.fpn import FPN
from model.resnet import ResNet
from model.solo import SOLO
from model.head import DecoupledSOLOHead
from loss.solo_loss import solo_loss
from tools.cocotools import get_classes


if __name__ == "__main__":
    cfg = DecoupledSOLO_R50_FPN_Config()
    num_layers = 5
    class_names = get_classes(cfg.classes_path)
    num_classes = len(class_names)
    batch_size = cfg.batch_size
    input_shape = (416, 416, 3)
    inputs = layers.Input(shape=input_shape, batch_size=cfg.batch_size)
    resnet = ResNet(50)
    fpn = FPN(in_channels=[256, 512, 1024, 2048], out_channels=256, num_outs=5)
    head = DecoupledSOLOHead()
    solo = SOLO(resnet, fpn, head)
    outs = solo(inputs, cfg, eval=False)
    model_body = models.Model(inputs=inputs, outputs=outs)

    batch_gt_objs_tensors = []
    batch_gt_clss_tensors = []
    batch_gt_masks_tensors = []
    batch_gt_pos_idx_tensors = []
    for lid in range(num_layers):
        sample_layer_gt_objs = layers.Input(name='layer%d_gt_objs' % (lid, ), shape=(None, None, 1), dtype='float32')
        sample_layer_gt_clss = layers.Input(name='layer%d_gt_clss' % (lid, ), shape=(None, None, num_classes), dtype='float32')
        sample_layer_gt_masks = layers.Input(name='layer%d_gt_masks' % (lid, ), shape=(None, None, None), dtype='float32')
        sample_layer_gt_pos_idx = layers.Input(name='layer%d_gt_pos_idx' % (lid, ), shape=(None, 3), dtype='int32')
        batch_gt_objs_tensors.append(sample_layer_gt_objs)
        batch_gt_clss_tensors.append(sample_layer_gt_clss)
        batch_gt_masks_tensors.append(sample_layer_gt_masks)
        batch_gt_pos_idx_tensors.append(sample_layer_gt_pos_idx)

    loss_list = layers.Lambda(solo_loss, name='solo_loss',
                           arguments={'batch_size': batch_size, 'num_layers': num_layers
                                      })([*model_body.output, *batch_gt_objs_tensors, *batch_gt_clss_tensors, *batch_gt_masks_tensors, *batch_gt_pos_idx_tensors])
    model = models.Model([model_body.input, *batch_gt_objs_tensors, *batch_gt_clss_tensors, *batch_gt_masks_tensors, *batch_gt_pos_idx_tensors], loss_list)
    model.summary()
    print("Begin Export")
    tf.keras.utils.plot_model(model=model, to_file='decoupled_solo_r50_fpn.png', show_shapes=True)
    print("Done!")
