# coding=utf-8
# ================================================================
#
#   Author      : LuoDeng
#   Created date: 2020-12-24 22:55:44
#   Description : FPN Neck
#
# ================================================================


class SOLO(object):
    def __init__(self, backbone, neck, head):
        super(SOLO, self).__init__()
        self.backbone = backbone
        self.neck = neck
        self.head = head

    def __call__(self, x, cfg, eval):
        x = self.backbone(x)
        x = self.neck(x)
        if eval:
            x = self.head(x, cfg.test_cfg, eval)
        else:
            x = self.head(x, None, eval)
        return x