## 基于SOLO: Segmenting Objects by Locations的实例分割（Tensorflow2实现）

### 版本依赖

|软件 | 版本 |
|--|--|
| Linux | Ubuntu 18.04/20.04 |
| cuda | 11.1 |
| cudnn | 8.4 |
| tensorflow | 2.4.0 |
| tensorflow-addons | 可选（使用了GN） |

### 项目结构

```bash
$ tree .
.
├── config.py
├── data
│   ├── coco -> /home/<username>/data/coco/ # Symbolic link to the datasets
│   └── coco_classes.txt                    # class name
├── decoupled_solo_r50_fpn.png              # Network Structure Generated by pydot(use tensorflow utils)
├── inference.py                            # for infertence
├── loss                                    # define the loss
│   ├── __init__.py
│   └── solo_loss.py                        # solo loss defined here
├── model                                   # define the model
│   ├── custom_layers.py                    # Resize, GN, IN, Conv2dUnit, Conv3x3 ...
│   ├── fpn.py                              # FPN
│   ├── head.py                             # Decoupled SOLO Head
│   ├── __init__.py
│   ├── resnet.py                           # ResNet, support r50 and r101
│   └── solo.py                             # SOLO Network
├── network_summary.txt
├── plotmodel.py                            # used to plot the model structure
├── pytorch2keras.py                        # convert model weight from torch style to keras or tf
├── README.md
├── requirements.txt                        # requirements
├── temp                                    # temp folder, used for debug
│   ├── 0_0.jpg
│   └── 0.jpg
├── tools                                   # dataprocess tools, for coco dataset transform
│   ├── cocotools.py
│   ├── data_process.py
│   ├── __init__.py
│   └── transform.py
├── train.py                                # train
└── weights                                 # saved weights
    ├── Decoupled_SOLO_R50_1x.h5
    ├── step00015000.h5
    └── ...
```

### 训练效果

![tensorboard](./images/readme/tensorboard.png)

### 参考资料

[1] [SOLO: Segmenting Objects by Locations](https://arxiv.org/abs/1912.04488)

[2] [SOLOv2: Dynamic and Fast Instance Segmentation](https://arxiv.org/abs/2003.10152)

[3] [原作者代码仓库（Pytorch和MMdetection实现）](https://github.com/WXinlong/SOLO)

[4] [Keras-SOLO](https://github.com/miemie2013/Keras-SOLO)
