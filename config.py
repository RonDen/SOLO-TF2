# coding=utf-8
# ================================================================
#
#   Author      : LuoDeng
#   Created date: 2020-12-10
#   Description : 配置文件。
#
# ================================================================


class TestConfig(object):
    def __init__(self):
        # self.nms_pre = 500
        # self.score_thr = 0.1
        # self.mask_thr = 0.5
        self.nms_pre = 500
        self.score_thr = 0.15
        self.mask_thr = 0.5
        self.update_thr = 0.1505   # 双分数阈值，这是第二个分数阈值，做第二次分数阈值过滤时使用。
        self.kernel = 'gaussian'
        self.sigma = 2.0
        self.max_per_img = 100


class DecoupledSOLO_R50_FPN_Config(object):
    """
    train.py里需要的配置
    """
    def __init__(self):
        # COCO数据集
        self.train_path = 'data/COCO/annotations/instances_train2017.json'
        # self.train_path = '../COCO/annotations/instances_val2017.json'
        self.val_path = 'data/COCO/annotations/instances_val2017.json'
        self.classes_path = 'data/coco_classes.txt'
        self.train_pre_path = 'data/COCO/train2017/'  # 训练集图片相对路径
        # self.train_pre_path = '../COCO/val2017/'  # 验证集图片相对路径
        self.val_pre_path = 'data/COCO/val2017/'  # 验证集图片相对路径

        # 模式。 0-从头训练，1-读取之前的模型继续训练（model_path可以是'yolov4.h5'、'./weights/step00001000.h5'这些。）
        self.pattern = 1
        self.lr = 0.00001
        self.batch_size = 2
        # 如果self.pattern = 1，需要指定self.model_path表示从哪个模型读取权重继续训练。
        self.model_path = './weights/step00007000.h5'

        # ========= 一些设置 =========
        # 每隔几步保存一次模型
        self.save_iter = 1000
        # 每隔几步计算一次eval集的mAP
        self.eval_iter = 5000000
        # 训练多少步
        self.max_iters = 800000


        # 验证
        # self.input_shape越大，精度会上升，但速度会下降。
        # self.input_shape = (320, 320)
        # self.input_shape = (416, 416)
        self.input_shape = (608, 608)
        # 验证时的分数阈值和nms_iou阈值
        self.conf_thresh = 0.001
        self.nms_thresh = 0.45
        # 是否画出验证集图片
        self.draw_image = False
        # 验证时的批大小
        self.eval_batch_size = 4


        # ============= 数据增强相关 =============
        self.with_mixup = False
        self.context = {'fields': ['image', 'gt_bbox', 'gt_class', 'gt_score']}
        # PadBox
        self.num_max_boxes = 70

        # test
        self.test_cfg = TestConfig()



class TrainConfig_2(object):
    """
    其它配置
    """
    def __init__(self):
        pass




