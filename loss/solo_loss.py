import tensorflow as tf

def dice_loss(pred_mask, gt_mask, gt_obj):
    a = tf.reduce_sum(pred_mask * gt_mask, axis=[1, 2])
    b = tf.reduce_sum(pred_mask * pred_mask, axis=[1, 2]) + 0.001
    c = tf.reduce_sum(gt_mask * gt_mask, axis=[1, 2]) + 0.001
    d = (2 * a) / (b + c)
    loss_mask_mask = tf.reshape(gt_obj, (-1, ))   # 掩码损失的掩码。
    return (1-d) * loss_mask_mask


def solo_loss2(args, batch_size, num_layers):
    p = 0

    ins_pred_x_list = []
    for i in range(num_layers):
        ins_pred_x_list.append(args[p])   # 从小感受野 到 大感受野 （从多格子 到 少格子）
        p += 1

    ins_pred_y_list = []
    for i in range(num_layers):
        ins_pred_y_list.append(args[p])   # 从小感受野 到 大感受野 （从多格子 到 少格子）
        p += 1

    cate_pred_list = []
    for i in range(num_layers):
        cate_pred_list.append(args[p])    # 从小感受野 到 大感受野 （从多格子 到 少格子）
        p += 1

    batch_gt_objs_tensors = []
    for i in range(num_layers):
        batch_gt_objs_tensors.append(args[p])
        p += 1

    batch_gt_clss_tensors = []
    for i in range(num_layers):
        batch_gt_clss_tensors.append(args[p])
        p += 1

    batch_gt_masks_tensors = []
    for i in range(num_layers):
        batch_gt_masks_tensors.append(args[p])
        p += 1

    batch_gt_pos_idx_tensors = []
    for i in range(num_layers):
        batch_gt_pos_idx_tensors.append(args[p])
        p += 1

    # ================= 计算损失 =================
    num_ins = 0.  # 记录这一批图片的正样本个数
    loss_clss, loss_masks = [], []
    for bid in range(batch_size):
        for lid in range(num_layers):
            # ================ 掩码损失 ======================
            pred_mask_x = ins_pred_x_list[lid][bid]
            pred_mask_y = ins_pred_y_list[lid][bid]
            pred_mask_x = tf.transpose(a=pred_mask_x, perm=[2, 0, 1])
            pred_mask_y = tf.transpose(a=pred_mask_y, perm=[2, 0, 1])

            gt_objs = batch_gt_objs_tensors[lid][bid]
            gt_masks = batch_gt_masks_tensors[lid][bid]
            pmidx = batch_gt_pos_idx_tensors[lid][bid]

            idx_sum = tf.reduce_sum(input_tensor=pmidx, axis=1)
            keep = tf.compat.v1.where(idx_sum > -1)
            keep = tf.reshape(keep, (-1, ))
            pmidx = tf.gather(pmidx, keep)

            yx_idx = pmidx[:, :2]
            y_idx = pmidx[:, 0]
            x_idx = pmidx[:, 1]
            m_idx = pmidx[:, 2]

            # 抽出来
            gt_obj = tf.gather_nd(gt_objs, yx_idx)
            mask_y = tf.gather(pred_mask_y, y_idx)
            mask_x = tf.gather(pred_mask_x, x_idx)
            gt_mask = tf.gather(gt_masks, m_idx)

            # 正样本数量
            num_ins += tf.reduce_sum(input_tensor=gt_obj)

            pred_mask = tf.sigmoid(mask_x) * tf.sigmoid(mask_y)
            loss_mask = dice_loss(pred_mask, gt_mask, gt_obj)
            loss_masks.append(loss_mask)


            # ================ 分类损失 ======================
            gamma = 2.0
            alpha = 0.25
            pred_conf = cate_pred_list[lid][bid]
            pred_conf = tf.sigmoid(pred_conf)
            gt_clss = batch_gt_clss_tensors[lid][bid]
            pos_loss = gt_clss * (0 - tf.math.log(pred_conf + 1e-9)) * tf.pow(1 - pred_conf, gamma) * alpha
            neg_loss = (1 - gt_clss) * (0 - tf.math.log(1 - pred_conf + 1e-9)) * tf.pow(pred_conf, gamma) * (1 - alpha)
            clss_loss = pos_loss + neg_loss
            clss_loss = tf.reduce_sum(input_tensor=clss_loss, axis=[0, 1])
            loss_clss.append(clss_loss)
    loss_masks = tf.concat(loss_masks, axis=0)
    ins_loss_weight = 3.0
    loss_masks = tf.reduce_sum(input_tensor=loss_masks) * ins_loss_weight
    loss_masks = loss_masks / (num_ins + 1e-9)   # 损失同原版SOLO，之所以不直接用tf.reduce_mean()，是因为多了一些0损失占位，分母并不等于num_ins。

    loss_clss = tf.concat(loss_clss, axis=0)
    clss_loss_weight = 1.0
    loss_clss = tf.reduce_sum(input_tensor=loss_clss) * clss_loss_weight
    loss_clss = loss_clss / (num_ins + 1e-9)

    return [loss_masks, loss_clss]


def solo_loss(args, batch_size, num_layers):
    p = 0

    ins_pred_x_list = []
    for i in range(num_layers):
        ins_pred_x_list.append(args[p])   # 从小感受野 到 大感受野 （从多格子 到 少格子）
        p += 1

    ins_pred_y_list = []
    for i in range(num_layers):
        ins_pred_y_list.append(args[p])   # 从小感受野 到 大感受野 （从多格子 到 少格子）
        p += 1

    cate_pred_list = []
    for i in range(num_layers):
        cate_pred_list.append(args[p])    # 从小感受野 到 大感受野 （从多格子 到 少格子）
        p += 1

    batch_gt_objs_tensors = []
    for i in range(num_layers):
        batch_gt_objs_tensors.append(args[p])
        p += 1

    batch_gt_clss_tensors = []
    for i in range(num_layers):
        batch_gt_clss_tensors.append(args[p])
        p += 1

    batch_gt_masks_tensors = []
    for i in range(num_layers):
        batch_gt_masks_tensors.append(args[p])
        p += 1

    batch_gt_pos_idx_tensors = []
    for i in range(num_layers):
        batch_gt_pos_idx_tensors.append(args[p])
        p += 1

    # ================= 计算损失 =================
    num_ins = 0.  # 记录这一批图片的正样本个数
    loss_clss, loss_masks = [], []
    for bid in range(batch_size):
        for lid in range(num_layers):
            # ================ 掩码损失 ======================
            pred_mask_x = ins_pred_x_list[lid][bid]
            pred_mask_y = ins_pred_y_list[lid][bid]
            pred_mask_x = tf.transpose(a=pred_mask_x, perm=[2, 0, 1])
            pred_mask_y = tf.transpose(a=pred_mask_y, perm=[2, 0, 1])

            gt_objs = batch_gt_objs_tensors[lid][bid]
            gt_masks = batch_gt_masks_tensors[lid][bid]
            pmidx = batch_gt_pos_idx_tensors[lid][bid]

            idx_sum = tf.reduce_sum(input_tensor=pmidx, axis=1)
            keep = tf.compat.v1.where(idx_sum > -1)
            keep = tf.reshape(keep, (-1, ))
            pmidx = tf.gather(pmidx, keep)

            yx_idx = pmidx[:, :2]
            y_idx = pmidx[:, 0]
            x_idx = pmidx[:, 1]
            m_idx = pmidx[:, 2]

            # 抽出来
            gt_obj = tf.gather_nd(gt_objs, yx_idx)
            mask_y = tf.gather(pred_mask_y, y_idx)
            mask_x = tf.gather(pred_mask_x, x_idx)
            gt_mask = tf.gather(gt_masks, m_idx)

            # 正样本数量
            num_ins += tf.reduce_sum(input_tensor=gt_obj)

            pred_mask = tf.sigmoid(mask_x) * tf.sigmoid(mask_y)
            loss_mask = dice_loss(pred_mask, gt_mask, gt_obj)
            loss_masks.append(loss_mask)


            # ================ 分类损失 ======================
            gamma = 2.0
            alpha = 0.25
            pred_conf = cate_pred_list[lid][bid]
            pred_conf = tf.sigmoid(pred_conf)
            gt_clss = batch_gt_clss_tensors[lid][bid]
            pos_loss = gt_clss * (0 - tf.math.log(pred_conf + 1e-9)) * tf.pow(1 - pred_conf, gamma) * alpha
            neg_loss = (1 - gt_clss) * (0 - tf.math.log(1 - pred_conf + 1e-9)) * tf.pow(pred_conf, gamma) * (1 - alpha)
            clss_loss = pos_loss + neg_loss
            clss_loss = tf.reduce_sum(input_tensor=clss_loss, axis=[0, 1])
            loss_clss.append(clss_loss)
    loss_masks = tf.concat(loss_masks, axis=0)
    ins_loss_weight = 3.0
    loss_masks = tf.reduce_sum(input_tensor=loss_masks) * ins_loss_weight
    loss_masks = loss_masks / (num_ins + 1e-9)   # 损失同原版SOLO，之所以不直接用tf.reduce_mean()，是因为多了一些0损失占位，分母并不等于num_ins。

    loss_clss = tf.concat(loss_clss, axis=0)
    clss_loss_weight = 1.0
    loss_clss = tf.reduce_sum(input_tensor=loss_clss) * clss_loss_weight
    loss_clss = loss_clss / (num_ins + 1e-9)

    return [loss_masks, loss_clss]
    # return loss_masks
