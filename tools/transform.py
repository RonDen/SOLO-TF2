#! /usr/bin/env python
# coding=utf-8
# ================================================================
#
#   Author      : LuoDeng
#   Created date: 2020-12-10
#   Description : 数据增强
#
# ================================================================
import cv2
import uuid
import numpy as np
from scipy import ndimage

try:
    from collections.abc import Sequence
except Exception:
    from collections import Sequence


class BboxError(ValueError):
    pass


class ImageError(ValueError):
    pass



class BaseOperator(object):
    def __init__(self, name=None):
        if name is None:
            name = self.__class__.__name__
        self._id = name + '_' + str(uuid.uuid4())[-6:]

    def __call__(self, sample, context=None):
        """ Process a sample.
        Args:
            sample (dict): a dict of sample, eg: {'image':xx, 'label': xxx}
            context (dict): info about this sample processing
        Returns:
            result (dict): a processed sample
        """
        return sample

    def __str__(self):
        return str(self._id)


class DecodeImage(BaseOperator):
    def __init__(self, to_rgb=True, with_mixup=False):
        """ Transform the image data to numpy format.
        对图片解码。最开始的一步。把图片读出来（rgb格式），加入到sample['image']。一维数组[h, w, 1]加入到sample['im_info']
        Args:
            to_rgb (bool): whether to convert BGR to RGB
            with_mixup (bool): whether or not to mixup image and gt_bbbox/gt_score
        """

        super(DecodeImage, self).__init__()
        self.to_rgb = to_rgb
        self.with_mixup = with_mixup
        if not isinstance(self.to_rgb, bool):
            raise TypeError("{}: input type is invalid.".format(self))
        if not isinstance(self.with_mixup, bool):
            raise TypeError("{}: input type is invalid.".format(self))

    def __call__(self, sample, context=None, coco=None):
        """ load image if 'im_file' field is not empty but 'image' is"""
        if 'image' not in sample:
            with open(sample['im_file'], 'rb') as f:
                sample['image'] = f.read()   # 增加一对键值对'image'。

        im = sample['image']
        data = np.frombuffer(im, dtype='uint8')
        im = cv2.imdecode(data, 1)  # BGR mode, but need RGB mode
        if self.to_rgb:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        sample['image'] = im

        if 'h' not in sample:
            sample['h'] = im.shape[0]
        if 'w' not in sample:
            sample['w'] = im.shape[1]
        # make default im_info with [h, w, 1]
        sample['im_info'] = np.array(   # 增加一对键值对'im_info'。
            [im.shape[0], im.shape[1], 1.], dtype=np.float32)
        # decode mixup image
        if self.with_mixup and 'mixup' in sample:
            self.__call__(sample['mixup'], context, coco)

        # 掩码图片。如果是训练集的话。
        if 'gt_poly' in sample:
            gt_poly = sample['gt_poly']
            assert len(gt_poly) == len(sample['gt_bbox']), "Poly Numbers Error."
            gt_mask = []
            if len(gt_poly) > 0:
                # 最初的方案，用cv2.fillPoly()画出真实掩码。因为有取整，发现有点偏差。
                # for obj_polys in gt_poly:
                #     mask = np.zeros((im.shape[0], im.shape[1]), dtype="uint8")
                #     for poly in obj_polys:  # coco数据集里，一个物体由多个多边形表示时（比如物体被挡住，不连通时）
                #         points = np.array(poly)
                #         points = np.reshape(points, (-1, 2))
                #         points = points.astype(np.int32)
                #         vertices = np.array([points], dtype=np.int32)
                #         cv2.fillPoly(mask, vertices, 1)   # 一定要是这个API而不是fillConvexPoly()，后者只能填充凸多边形而不支持凹多边形。
                #         mask = np.reshape(mask, (im.shape[0], im.shape[1], 1))
                #     gt_mask.append(mask)
                # gt_mask = np.concatenate(gt_mask, axis=-1)

                # 现在的方案，用annToMask()得到真实掩码
                anno_id = sample['anno_id']
                target = coco.loadAnns(anno_id)
                # mask是一个list，里面每一个元素是一个shape=(height*width,)的ndarray，1代表是这个注解注明的物体，0代表其他物体和背景。
                masks = [coco.annToMask(obj).reshape(-1) for obj in target]
                masks = np.vstack(masks)  # 设N=len(mask)=注解数，这一步将mask转换成一个ndarray，shape=(N, height*width)
                masks = masks.reshape(-1, im.shape[0], im.shape[1])  # shape=(N, height, width)
                gt_mask = masks.transpose(1, 2, 0)     # shape=(height, width, N)
            else:   # 对于没有gt的纯背景图，弄1个方便后面的增强跟随sample['image']
                gt_mask = np.zeros((im.shape[0], im.shape[1], 1), dtype=np.int32)
            sample['gt_mask'] = gt_mask
        return sample


class MixupImage(BaseOperator):
    def __init__(self, alpha=1.5, beta=1.5):
        """ Mixup image and gt_bbbox/gt_score
        Args:
            alpha (float): alpha parameter of beta distribute
            beta (float): beta parameter of beta distribute
        """
        super(MixupImage, self).__init__()
        self.alpha = alpha
        self.beta = beta
        if self.alpha <= 0.0:
            raise ValueError("alpha shold be positive in {}".format(self))
        if self.beta <= 0.0:
            raise ValueError("beta shold be positive in {}".format(self))

    def _mixup_img(self, img1, img2, factor):
        h = max(img1.shape[0], img2.shape[0])
        w = max(img1.shape[1], img2.shape[1])
        img = np.zeros((h, w, img1.shape[2]), 'float32')
        img[:img1.shape[0], :img1.shape[1], :] = img1.astype('float32') * factor
        img[:img2.shape[0], :img2.shape[1], :] += img2.astype('float32') * (1.0 - factor)
        return img.astype('uint8')

    def _concat_mask(self, mask1, mask2, gt_score1, gt_score2):
        h = max(mask1.shape[0], mask2.shape[0])
        w = max(mask1.shape[1], mask2.shape[1])
        expand_mask1 = np.zeros((h, w, mask1.shape[2]), 'float32')
        expand_mask2 = np.zeros((h, w, mask2.shape[2]), 'float32')
        expand_mask1[:mask1.shape[0], :mask1.shape[1], :] = mask1
        expand_mask2[:mask2.shape[0], :mask2.shape[1], :] = mask2
        l1 = len(gt_score1)
        l2 = len(gt_score2)
        if l2 == 0:
            return expand_mask1
        elif l1 == 0:
            return expand_mask2
        mask = np.concatenate((expand_mask1, expand_mask2), axis=-1)
        return mask

    def __call__(self, sample, context=None):
        if 'mixup' not in sample:
            return sample

        # 一定概率触发mixup
        if np.random.uniform(0., 1.) < 0.5:
            sample.pop('mixup')
            return sample

        factor = np.random.beta(self.alpha, self.beta)
        factor = max(0.0, min(1.0, factor))
        if factor >= 1.0:
            sample.pop('mixup')
            return sample
        if factor <= 0.0:
            return sample['mixup']
        im = self._mixup_img(sample['image'], sample['mixup']['image'], factor)
        gt_bbox1 = sample['gt_bbox']
        gt_bbox2 = sample['mixup']['gt_bbox']
        gt_bbox = np.concatenate((gt_bbox1, gt_bbox2), axis=0)
        gt_class1 = sample['gt_class']
        gt_class2 = sample['mixup']['gt_class']
        gt_class = np.concatenate((gt_class1, gt_class2), axis=0)
        gt_score1 = sample['gt_score']
        gt_score2 = sample['mixup']['gt_score']
        gt_score = np.concatenate(
            (gt_score1 * factor, gt_score2 * (1. - factor)), axis=0)
        # mask = self._concat_mask(sample['gt_mask'], sample['mixup']['gt_mask'], gt_score1, gt_score2)
        sample['image'] = im
        # sample['gt_mask'] = mask
        sample['gt_bbox'] = gt_bbox
        sample['gt_score'] = gt_score
        sample['gt_class'] = gt_class
        sample['h'] = im.shape[0]
        sample['w'] = im.shape[1]
        sample.pop('mixup')
        return sample


class PhotometricDistort(BaseOperator):
    def __init__(self):
        super(PhotometricDistort, self).__init__()

    def __call__(self, sample, context=None):
        im = sample['image']

        image = im.astype(np.float32)

        # RandomBrightness
        if np.random.randint(2):
            delta = 32
            delta = np.random.uniform(-delta, delta)
            image += delta

        image[image < 0] = 0
        state = np.random.randint(2)
        if state == 0:
            if np.random.randint(2):
                lower = 0.5
                upper = 1.5
                alpha = np.random.uniform(lower, upper)
                image *= alpha

        image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        if np.random.randint(2):
            lower = 0.5
            upper = 1.5
            image[:, :, 1] *= np.random.uniform(lower, upper)

        if np.random.randint(2):
            delta = 18.0
            image[:, :, 0] += np.random.uniform(-delta, delta)
            image[:, :, 0][image[:, :, 0] > 360.0] -= 360.0
            image[:, :, 0][image[:, :, 0] < 0.0] += 360.0

        image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)

        if state == 1:
            if np.random.randint(2):
                lower = 0.5
                upper = 1.5
                alpha = np.random.uniform(lower, upper)
                image *= alpha

        sample['image'] = image
        return sample


class RandomCrop(BaseOperator):
    """Random crop image and bboxes.

    Args:
        aspect_ratio (list): aspect ratio of cropped region.
            in [min, max] format.
        thresholds (list): iou thresholds for decide a valid bbox crop.
        scaling (list): ratio between a cropped region and the original image.
             in [min, max] format.
        num_attempts (int): number of tries before giving up.
        allow_no_crop (bool): allow return without actually cropping them.
        cover_all_box (bool): ensure all bboxes are covered in the final crop.
    """

    def __init__(self,
                 aspect_ratio=[.5, 2.],
                 thresholds=[.0, .1, .3, .5, .7, .9],
                 scaling=[.3, 1.],
                 num_attempts=50,
                 allow_no_crop=True,
                 cover_all_box=False):
        super(RandomCrop, self).__init__()
        self.aspect_ratio = aspect_ratio
        self.thresholds = thresholds
        self.scaling = scaling
        self.num_attempts = num_attempts
        self.allow_no_crop = allow_no_crop
        self.cover_all_box = cover_all_box

    def __call__(self, sample, context=None):
        if 'gt_bbox' in sample and len(sample['gt_bbox']) == 0:
            return sample

        h = sample['h']
        w = sample['w']
        gt_bbox = sample['gt_bbox']

        # NOTE Original method attempts to generate one candidate for each
        # threshold then randomly sample one from the resulting list.
        # Here a short circuit approach is taken, i.e., randomly choose a
        # threshold and attempt to find a valid crop, and simply return the
        # first one found.
        # The probability is not exactly the same, kinda resembling the
        # "Monty Hall" problem. Actually carrying out the attempts will affect
        # observability (just like opening doors in the "Monty Hall" game).
        thresholds = list(self.thresholds)
        if self.allow_no_crop:
            thresholds.append('no_crop')
        np.random.shuffle(thresholds)

        for thresh in thresholds:
            if thresh == 'no_crop':
                return sample

            found = False
            for i in range(self.num_attempts):
                scale = np.random.uniform(*self.scaling)
                min_ar, max_ar = self.aspect_ratio
                aspect_ratio = np.random.uniform(
                    max(min_ar, scale**2), min(max_ar, scale**-2))
                crop_h = int(h * scale / np.sqrt(aspect_ratio))
                crop_w = int(w * scale * np.sqrt(aspect_ratio))
                crop_y = np.random.randint(0, h - crop_h)
                crop_x = np.random.randint(0, w - crop_w)
                crop_box = [crop_x, crop_y, crop_x + crop_w, crop_y + crop_h]
                iou = self._iou_matrix(
                    gt_bbox, np.array(
                        [crop_box], dtype=np.float32))
                if iou.max() < thresh:
                    continue

                if self.cover_all_box and iou.min() < thresh:
                    continue

                cropped_box, valid_ids = self._crop_box_with_center_constraint(
                    gt_bbox, np.array(
                        crop_box, dtype=np.float32))
                if valid_ids.size > 0:
                    found = True
                    break

            if found:
                sample['image'] = self._crop_image(sample['image'], crop_box)
                gt_mask = self._crop_image(sample['gt_mask'], crop_box)    # 掩码裁剪
                sample['gt_mask'] = np.take(gt_mask, valid_ids, axis=-1)   # 掩码筛选
                sample['gt_bbox'] = np.take(cropped_box, valid_ids, axis=0)
                sample['gt_class'] = np.take(
                    sample['gt_class'], valid_ids, axis=0)
                sample['w'] = crop_box[2] - crop_box[0]
                sample['h'] = crop_box[3] - crop_box[1]
                if 'gt_score' in sample:
                    sample['gt_score'] = np.take(
                        sample['gt_score'], valid_ids, axis=0)
                return sample

        return sample

    def _iou_matrix(self, a, b):
        tl_i = np.maximum(a[:, np.newaxis, :2], b[:, :2])
        br_i = np.minimum(a[:, np.newaxis, 2:], b[:, 2:])

        area_i = np.prod(br_i - tl_i, axis=2) * (tl_i < br_i).all(axis=2)
        area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
        area_b = np.prod(b[:, 2:] - b[:, :2], axis=1)
        area_o = (area_a[:, np.newaxis] + area_b - area_i)
        return area_i / (area_o + 1e-10)

    def _crop_box_with_center_constraint(self, box, crop):
        cropped_box = box.copy()

        cropped_box[:, :2] = np.maximum(box[:, :2], crop[:2])
        cropped_box[:, 2:] = np.minimum(box[:, 2:], crop[2:])
        cropped_box[:, :2] -= crop[:2]
        cropped_box[:, 2:] -= crop[:2]

        centers = (box[:, :2] + box[:, 2:]) / 2
        valid = np.logical_and(crop[:2] <= centers,
                               centers < crop[2:]).all(axis=1)
        valid = np.logical_and(
            valid, (cropped_box[:, :2] < cropped_box[:, 2:]).all(axis=1))

        return cropped_box, np.where(valid)[0]

    def _crop_image(self, img, crop):
        x1, y1, x2, y2 = crop
        return img[y1:y2, x1:x2, :]

class RandomFlipImage(BaseOperator):
    def __init__(self, prob=0.5, is_normalized=False, is_mask_flip=False):
        """
        Args:
            prob (float): the probability of flipping image
            is_normalized (bool): whether the bbox scale to [0,1]
            is_mask_flip (bool): whether flip the segmentation
        """
        super(RandomFlipImage, self).__init__()
        self.prob = prob
        self.is_normalized = is_normalized
        self.is_mask_flip = is_mask_flip
        if not (isinstance(self.prob, float) and
                isinstance(self.is_normalized, bool) and
                isinstance(self.is_mask_flip, bool)):
            raise TypeError("{}: input type is invalid.".format(self))

    def flip_segms(self, segms, height, width):
        def _flip_poly(poly, width):
            flipped_poly = np.array(poly)
            flipped_poly[0::2] = width - np.array(poly[0::2]) - 1
            return flipped_poly.tolist()

        def _flip_rle(rle, height, width):
            if 'counts' in rle and type(rle['counts']) == list:
                rle = mask_util.frPyObjects([rle], height, width)
            mask = mask_util.decode(rle)
            mask = mask[:, ::-1, :]
            rle = mask_util.encode(np.array(mask, order='F', dtype=np.uint8))
            return rle

        def is_poly(segm):
            assert isinstance(segm, (list, dict)), \
                "Invalid segm type: {}".format(type(segm))
            return isinstance(segm, list)

        flipped_segms = []
        for segm in segms:
            if is_poly(segm):
                # Polygon format
                flipped_segms.append([_flip_poly(poly, width) for poly in segm])
            else:
                # RLE format
                import pycocotools.mask as mask_util
                flipped_segms.append(_flip_rle(segm, height, width))
        return flipped_segms

    def __call__(self, sample, context=None):
        """Filp the image and bounding box.
        Operators:
            1. Flip the image numpy.
            2. Transform the bboxes' x coordinates.
              (Must judge whether the coordinates are normalized!)
            3. Transform the segmentations' x coordinates.
              (Must judge whether the coordinates are normalized!)
        Output:
            sample: the image, bounding box and segmentation part
                    in sample are flipped.
        """

        samples = sample
        batch_input = True
        if not isinstance(samples, Sequence):
            batch_input = False
            samples = [samples]
        for sample in samples:
            gt_bbox = sample['gt_bbox']
            im = sample['image']
            gt_mask = sample['gt_mask']
            if not isinstance(im, np.ndarray):
                raise TypeError("{}: image is not a numpy array.".format(self))
            if len(im.shape) != 3:
                raise ImageError("{}: image is not 3-dimensional.".format(self))
            height, width, _ = im.shape
            if np.random.uniform(0, 1) < self.prob:
                im = im[:, ::-1, :]
                gt_mask = gt_mask[:, ::-1, :]
                if gt_bbox.shape[0] == 0:
                    return sample
                oldx1 = gt_bbox[:, 0].copy()
                oldx2 = gt_bbox[:, 2].copy()
                if self.is_normalized:
                    gt_bbox[:, 0] = 1 - oldx2
                    gt_bbox[:, 2] = 1 - oldx1
                else:
                    gt_bbox[:, 0] = width - oldx2 - 1
                    gt_bbox[:, 2] = width - oldx1 - 1
                if gt_bbox.shape[0] != 0 and (
                        gt_bbox[:, 2] < gt_bbox[:, 0]).all():
                    m = "{}: invalid box, x2 should be greater than x1".format(
                        self)
                    raise BboxError(m)
                sample['gt_bbox'] = gt_bbox
                if self.is_mask_flip and len(sample['gt_poly']) != 0:
                    sample['gt_poly'] = self.flip_segms(sample['gt_poly'],
                                                        height, width)
                sample['flipped'] = True
                sample['image'] = im
                sample['gt_mask'] = gt_mask
        sample = samples if batch_input else samples[0]
        return sample

class NormalizeBox(BaseOperator):
    """Transform the bounding box's coornidates to [0,1]."""

    def __init__(self):
        super(NormalizeBox, self).__init__()

    def __call__(self, sample, context):
        gt_bbox = sample['gt_bbox']
        width = sample['w']
        height = sample['h']
        for i in range(gt_bbox.shape[0]):
            gt_bbox[i][0] /= width
            gt_bbox[i][1] /= height
            gt_bbox[i][2] /= width
            gt_bbox[i][3] /= height
        sample['gt_bbox'] = gt_bbox
        return sample

class PadBox(BaseOperator):
    def __init__(self, num_max_boxes=50):
        """
        Pad zeros to bboxes if number of bboxes is less than num_max_boxes.
        Args:
            num_max_boxes (int): the max number of bboxes
        """
        self.num_max_boxes = num_max_boxes
        super(PadBox, self).__init__()

    def __call__(self, sample, context=None):
        assert 'gt_bbox' in sample
        bbox = sample['gt_bbox']
        gt_num = min(self.num_max_boxes, len(bbox))
        num_max = self.num_max_boxes
        fields = context['fields'] if context else []
        pad_bbox = np.zeros((num_max, 4), dtype=np.float32)
        if gt_num > 0:
            pad_bbox[:gt_num, :] = bbox[:gt_num, :]
        sample['gt_bbox'] = pad_bbox

        # 掩码
        mask = sample['gt_mask']
        pad_mask = np.zeros((mask.shape[0], mask.shape[1], num_max), dtype=np.float32)
        if gt_num > 0:
            pad_mask[:, :, :gt_num] = mask[:, :, :gt_num]
        sample['gt_mask'] = pad_mask

        if 'gt_class' in fields:
            pad_class = np.zeros((num_max), dtype=np.int32)
            if gt_num > 0:
                pad_class[:gt_num] = sample['gt_class'][:gt_num, 0]
            sample['gt_class'] = pad_class
        if 'gt_score' in fields:
            pad_score = np.zeros((num_max), dtype=np.float32)
            if gt_num > 0:
                pad_score[:gt_num] = sample['gt_score'][:gt_num, 0]
            sample['gt_score'] = pad_score
        # in training, for example in op ExpandImage,
        # the bbox and gt_class is expandded, but the difficult is not,
        # so, judging by it's length
        if 'is_difficult' in fields:
            pad_diff = np.zeros((num_max), dtype=np.int32)
            if gt_num > 0:
                pad_diff[:gt_num] = sample['difficult'][:gt_num, 0]
            sample['difficult'] = pad_diff
        return sample

class BboxXYXY2XYWH(BaseOperator):
    """
    Convert bbox XYXY format to XYWH format.
    """

    def __init__(self):
        super(BboxXYXY2XYWH, self).__init__()

    def __call__(self, sample, context=None):
        assert 'gt_bbox' in sample
        bbox = sample['gt_bbox']
        bbox[:, 2:4] = bbox[:, 2:4] - bbox[:, :2]
        bbox[:, :2] = bbox[:, :2] + bbox[:, 2:4] / 2.
        sample['gt_bbox'] = bbox
        return sample

class RandomShape(BaseOperator):
    """
    6个分辨率(w, h)，随机选一个分辨率(w, h)训练。也随机选一种插值方式。原版SOLO中，因为设定了size_divisor=32，
    所以被填充黑边的宽（或者高）会填充最少的黑边使得被32整除。所以一个batch最后所有的图片的大小有很大概率是不同的，
    这里为了使得一批图片能被一个四维张量表示，所以按照size_divisor=None处理，即统一填充到被选中的分辨率(w, h)

    Args:
        sizes (list): 最大分辨率(w, h)
        random_inter (bool): whether to randomly interpolation, defalut true.
    """

    def __init__(self, sizes=[(800, 800), (768, 768), (736, 736), (704, 704), (672, 672), (640, 640)],
                 random_inter=True, keep_ratio=True):
        super(RandomShape, self).__init__()
        self.sizes = sizes
        self.random_inter = random_inter
        self.keep_ratio = keep_ratio
        self.interps = [
            cv2.INTER_NEAREST,
            cv2.INTER_LINEAR,
            cv2.INTER_AREA,
            cv2.INTER_CUBIC,
            cv2.INTER_LANCZOS4,
        ] if random_inter else []

    def __call__(self, samples, context=None):
        # 是randomShape导致了Process finished with exit code -1073740940 (0xC0000374)

        size_idx = np.random.randint(len(self.sizes))
        shape = self.sizes[size_idx]
        # mask_shape = shape // 4
        method = np.random.choice(self.interps) if self.random_inter \
            else cv2.INTER_NEAREST
        for i in range(len(samples)):
            im = samples[i]['image']
            h, w = im.shape[:2]

            scale_x = float(shape[0]) / w
            scale_y = float(shape[1]) / h
            scale_factor = min(scale_x, scale_y)
            if self.keep_ratio:
                im = cv2.resize(im, None, None, fx=scale_factor, fy=scale_factor, interpolation=method)
            else:
                im = cv2.resize(im, None, None, fx=scale_x, fy=scale_y, interpolation=method)

            gt_mask = samples[i]['gt_mask']
            # 掩码也跟随着插值成图片大小
            # 不能随机插值方法，有的方法不适合50个通道。
            # 不转换类型的话，插值失败
            # gt_mask = gt_mask.astype(np.uint8)

            # 是这一句randomShape导致了Process finished with exit code -1073740940 (0xC0000374)。填充到3个来避免？
            gt_mask = cv2.resize(gt_mask, (im.shape[1], im.shape[0]), interpolation=cv2.INTER_LINEAR)
            # if len(gt_mask.shape) == 2:   # 只有一个通道时，会变成二维数组
            #     gt_mask = gt_mask[:, :, np.newaxis]
            gt_mask = (gt_mask > 0.5).astype(np.float32)

            # 方框也要变
            gt_bbox = samples[i]['gt_bbox']

            # 填充黑边
            if self.keep_ratio:
                pad_im = np.zeros((shape[1], shape[0], 3), np.float32)
                pad_gt_mask = np.zeros((shape[1], shape[0], gt_mask.shape[2]), np.float32)
                pad_x = (shape[0] - im.shape[1]) // 2
                pad_y = (shape[1] - im.shape[0]) // 2
                pad_im[pad_y:pad_y+im.shape[0], pad_x:pad_x+im.shape[1], :] = im
                pad_gt_mask[pad_y:pad_y+im.shape[0], pad_x:pad_x+im.shape[1], :] = gt_mask
                samples[i]['image'] = pad_im
                samples[i]['gt_mask'] = pad_gt_mask
                gt_bbox *= [scale_factor, scale_factor, scale_factor, scale_factor]
                gt_bbox += [pad_x, pad_y, pad_x, pad_y]
            else:
                samples[i]['image'] = im
                samples[i]['gt_mask'] = gt_mask
                gt_bbox *= [scale_x, scale_y, scale_x, scale_y]
            samples[i]['gt_bbox'] = gt_bbox
            samples[i]['h'] = shape[1]
            samples[i]['w'] = shape[0]
        return samples

class NormalizeImage(BaseOperator):
    def __init__(self,
                 mean=[123.675, 116.28, 103.53],
                 std=[58.395, 57.12, 57.375],
                 is_scale=True,
                 is_channel_first=True):
        """
        Args:
            mean (list): the pixel mean
            std (list): the pixel variance
        """
        super(NormalizeImage, self).__init__()
        self.mean = mean
        self.std = std
        self.is_scale = is_scale
        self.is_channel_first = is_channel_first
        if not (isinstance(self.mean, list) and isinstance(self.std, list) and
                isinstance(self.is_scale, bool)):
            raise TypeError("{}: input type is invalid.".format(self))
        from functools import reduce
        if reduce(lambda x, y: x * y, self.std) == 0:
            raise ValueError('{}: std is invalid!'.format(self))

    def __call__(self, sample, context=None):
        """Normalize the image.
        Operators:
            1.(optional) Scale the image to [0,1]
            2. Each pixel minus mean and is divided by std
        """
        samples = sample
        batch_input = True
        if not isinstance(samples, Sequence):
            batch_input = False
            samples = [samples]
        for sample in samples:
            for k in sample.keys():
                # hard code
                if k.startswith('image'):
                    im = sample[k]
                    im = im.astype(np.float32, copy=False)
                    if self.is_channel_first:
                        mean = np.array(self.mean)[:, np.newaxis, np.newaxis]
                        std = np.array(self.std)[:, np.newaxis, np.newaxis]
                    else:
                        mean = np.array(self.mean)[np.newaxis, np.newaxis, :]
                        std = np.array(self.std)[np.newaxis, np.newaxis, :]
                    if self.is_scale:
                        im = im / 255.0
                    im -= mean
                    im /= std
                    sample[k] = im
        if not batch_input:
            samples = samples[0]
        return samples


def bbox_area(src_bbox):
    if src_bbox[2] < src_bbox[0] or src_bbox[3] < src_bbox[1]:
        return 0.
    else:
        width = src_bbox[2] - src_bbox[0]
        height = src_bbox[3] - src_bbox[1]
        return width * height

def jaccard_overlap(sample_bbox, object_bbox):
    if sample_bbox[0] >= object_bbox[2] or \
        sample_bbox[2] <= object_bbox[0] or \
        sample_bbox[1] >= object_bbox[3] or \
        sample_bbox[3] <= object_bbox[1]:
        return 0
    intersect_xmin = max(sample_bbox[0], object_bbox[0])
    intersect_ymin = max(sample_bbox[1], object_bbox[1])
    intersect_xmax = min(sample_bbox[2], object_bbox[2])
    intersect_ymax = min(sample_bbox[3], object_bbox[3])
    intersect_size = (intersect_xmax - intersect_xmin) * (
        intersect_ymax - intersect_ymin)
    sample_bbox_size = bbox_area(sample_bbox)
    object_bbox_size = bbox_area(object_bbox)
    overlap = intersect_size / (
        sample_bbox_size + object_bbox_size - intersect_size)
    return overlap

class Gt2SoloTarget(BaseOperator):
    """
    Generate SOLO targets.
    """

    def __init__(self,
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
                 with_deform=False):
        super(Gt2SoloTarget, self).__init__()
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
        self.max_pos = 600

    def __call__(self, samples, context=None):
        h, w = samples[0]['image'].shape[:2]
        featmap_sizes = []
        for st in self.strides:
            featmap_sizes.append([h*2//st, w*2//st])

        batch_size = len(samples)
        batch_image = np.zeros((batch_size, h, w, 3))
        batch_gt_objs, batch_gt_clss, batch_gt_masks, batch_gt_pos_idx = [], [], [], []
        for i in range(batch_size):
            im = samples[i]['image']
            gt_bbox = samples[i]['gt_bbox']
            gt_score = samples[i]['gt_score']
            gt_class = samples[i]['gt_class']
            gt_mask = samples[i]['gt_mask']
            gt_mask = gt_mask.transpose(2, 0, 1)
            # Pad去掉
            gt_num = 0
            for q in range(len(gt_score)):
                if gt_score[q] > 0:
                    gt_num += 1
                else:
                    break
            gt_bbox = gt_bbox[:gt_num]
            gt_class = gt_class[:gt_num]
            gt_mask = gt_mask[:gt_num]
            gt_objs_per_layer, gt_clss_per_layer, gt_masks_per_layer, gt_pos_idx_per_layer = self.solo_target_single(
                gt_bbox, gt_class, gt_mask, featmap_sizes)
            batch_gt_objs.append(gt_objs_per_layer)
            batch_gt_clss.append(gt_clss_per_layer)
            batch_gt_masks.append(gt_masks_per_layer)
            batch_gt_pos_idx.append(gt_pos_idx_per_layer)
            batch_image[i, :, :, :] = im

        num_layers = len(batch_gt_objs[0])
        batch_gt_objs_tensors = []
        batch_gt_clss_tensors = []
        batch_gt_masks_tensors = []
        batch_gt_pos_idx_tensors = []
        for lid in range(num_layers):
            list_1 = []
            list_2 = []
            list_3 = []
            list_4 = []
            max_mask = -1
            for bid in range(batch_size):
                gt_objs = batch_gt_objs[bid][lid]
                gt_clss = batch_gt_clss[bid][lid]
                gt_masks = batch_gt_masks[bid][lid]
                gt_pos_idx = batch_gt_pos_idx[bid][lid]

                list_1.append(gt_objs[np.newaxis, :, :, :])
                list_2.append(gt_clss[np.newaxis, :, :, :])
                list_3.append(gt_masks)
                nnn = gt_masks.shape[0]
                if nnn > max_mask:
                    max_mask = nnn
                list_4.append(gt_pos_idx[np.newaxis, :, :])
            list_1 = np.concatenate(list_1, 0)
            list_2 = np.concatenate(list_2, 0)
            temp = np.zeros((batch_size, max_mask, list_3[0].shape[1], list_3[0].shape[2]), np.uint8)
            for bid in range(batch_size):
                gt_masks = list_3[bid]
                nnn = gt_masks.shape[0]
                temp[bid, :nnn, :, :] = gt_masks
            list_4 = np.concatenate(list_4, 0)
            batch_gt_objs_tensors.append(list_1)
            batch_gt_clss_tensors.append(list_2)
            batch_gt_masks_tensors.append(temp)
            batch_gt_pos_idx_tensors.append(list_4)
        return batch_image, batch_gt_objs_tensors, batch_gt_clss_tensors, batch_gt_masks_tensors, batch_gt_pos_idx_tensors

    def solo_target_single(self,
                           gt_bboxes_raw,
                           gt_labels_raw,
                           gt_masks_raw,
                           featmap_sizes=None):
        # ins
        # 平均边长，几何平均数， [n, ]
        gt_areas = np.sqrt((gt_bboxes_raw[:, 2] - gt_bboxes_raw[:, 0]) * (gt_bboxes_raw[:, 3] - gt_bboxes_raw[:, 1]))

        gt_objs_per_layer = []
        gt_clss_per_layer = []
        gt_masks_per_layer = []
        gt_pos_idx_per_layer = []
        # 遍历每个输出层
        #           (1,     96)            8     [104, 104]     40
        for (lower_bound, upper_bound), stride, featmap_size, num_grid \
                in zip(self.scale_ranges, self.strides, featmap_sizes, self.seg_num_grids):
            # [40, 40, 1]  objectness
            gt_objs = np.zeros([num_grid, num_grid, 1], dtype=np.float32)
            # [40, 40, 80]  种类one-hot
            gt_clss = np.zeros([num_grid, num_grid, self.num_classes], dtype=np.float32)
            # [?, 104, 104]  这一输出层的gt_masks，可能同一个掩码重复多次
            gt_masks = []
            # [self.max_pos, 3]    坐标以-2初始化
            # 前2个用于把正样本抽出来gather_nd()，后1个用于把掩码抽出来gather()。为了避免使用layers.where()后顺序没对上，所以不拆开写。
            gt_pos_idx = np.zeros([self.max_pos, 3], dtype=np.int32) - 2
            # 掩码计数
            p = 0

            # 这一张图片，所有物体，若平均边长在这个范围，这一输出层就负责预测。因为面积范围有交集，所以一个gt可以被分配到多个输出层上。
            hit_indices = np.where((gt_areas >= lower_bound) & (gt_areas <= upper_bound))[0]

            if len(hit_indices) == 0:   # 这一层没有正样本
                gt_objs_per_layer.append(gt_objs)   # 全是0
                gt_clss_per_layer.append(gt_clss)   # 全是0
                gt_masks = np.zeros([1, featmap_size[0], featmap_size[1]], dtype=np.uint8)   # 全是0，至少一张掩码，方便gather()
                gt_masks_per_layer.append(gt_masks)
                gt_pos_idx[0, :] = np.array([0, 0, 0], dtype=np.int32)   # 没有正样本，默认会抽第0行第0列格子，默认会抽这一层gt_mask里第0个掩码。
                gt_pos_idx_per_layer.append(gt_pos_idx)
                continue
            gt_bboxes_raw_this_layer = gt_bboxes_raw[hit_indices]   # shape=[m, 4]  这一层负责预测的物体的bbox
            gt_labels_raw_this_layer = gt_labels_raw[hit_indices]   # shape=[m, ]   这一层负责预测的物体的类别id
            gt_masks_raw_this_layer = gt_masks_raw[hit_indices]   # [m, ?, ?]

            half_ws = 0.5 * (gt_bboxes_raw_this_layer[:, 2] - gt_bboxes_raw_this_layer[:, 0]) * self.sigma   # shape=[m, ]  宽的一半
            half_hs = 0.5 * (gt_bboxes_raw_this_layer[:, 3] - gt_bboxes_raw_this_layer[:, 1]) * self.sigma   # shape=[m, ]  高的一半

            output_stride = stride / 2   # 因为网络最后对ins_feat_x、ins_feat_y进行上采样，所以stride / 2

            for seg_mask, gt_label, half_h, half_w in zip(gt_masks_raw_this_layer, gt_labels_raw_this_layer, half_hs, half_ws):
                if seg_mask.sum() < 10:   # 忽略太小的物体
                   continue
                # mass center
                upsampled_size = (featmap_sizes[0][0] * 4, featmap_sizes[0][1] * 4)   # 也就是输入图片的大小
                center_h, center_w = ndimage.measurements.center_of_mass(seg_mask)    # 求物体掩码的质心。scipy提供技术支持。
                # seg_mask.sum()是0时，center_w是数值nan
                coord_w = int((center_w / upsampled_size[1]) // (1. / num_grid))      # 物体质心落在了第几列格子
                coord_h = int((center_h / upsampled_size[0]) // (1. / num_grid))      # 物体质心落在了第几行格子

                # left, top, right, down
                top_box = max(0, int(((center_h - half_h) / upsampled_size[0]) // (1. / num_grid)))      # 物体左上角落在了第几行格子
                down_box = min(num_grid - 1, int(((center_h + half_h) / upsampled_size[0]) // (1. / num_grid)))    # 物体右下角落在了第几行格子
                left_box = max(0, int(((center_w - half_w) / upsampled_size[1]) // (1. / num_grid)))     # 物体左上角落在了第几列格子
                right_box = min(num_grid - 1, int(((center_w + half_w) / upsampled_size[1]) // (1. / num_grid)))   # 物体右下角落在了第几列格子

                # 物体的宽高并没有那么重要。将物体的左上角、右下角限制在质心所在的九宫格内。当物体很小时，物体的左上角、右下角、质心位于同一个格子。
                top = max(top_box, coord_h-1)
                down = min(down_box, coord_h+1)
                # down = top
                left = max(coord_w-1, left_box)
                right = min(right_box, coord_w+1)
                # right = left

                # 40x40的网格，将负责预测gt的格子填上gt_objs和gt_clss，此处同YOLOv3
                # ins  [img_h, img_w]->[img_h/output_stride, img_w/output_stride]  将gt的掩码下采样output_stride倍。
                seg_mask = cv2.resize(seg_mask, None, None, fx=1. / output_stride, fy=1. / output_stride, interpolation=cv2.INTER_LINEAR)
                # seg_mask = mmcv.imrescale(seg_mask, scale=1. / output_stride)
                for i in range(top, down+1):
                    for j in range(left, right+1):
                        if gt_objs[i, j, 0] < 0.5:   # 这个格子没有被填过gt才可以填。
                            gt_objs[i, j, 0] = 1.0   # 此处同YOLOv3
                            gt_clss[i, j, gt_label] = 1.0   # 此处同YOLOv3
                            cp_mask = np.copy(seg_mask)
                            cp_mask = cp_mask[np.newaxis, :, :]
                            gt_masks.append(cp_mask)
                            gt_pos_idx[p, :] = np.array([i, j, p], dtype=np.int32)   # 前2个用于把正样本抽出来gather_nd()，后1个用于把掩码抽出来gather()。
                            p += 1
            # 平均边长在这个范围内，但是因为面积太小seg_mask.sum() < 10导致continue，导致这一输出层也没有正样本的时候，也分配一个。
            if len(gt_masks) == 0:
                gt_masks = np.zeros([1, featmap_size[0], featmap_size[1]], dtype=np.uint8)   # 全是0，至少一张掩码，方便gather()
                gt_pos_idx[0, :] = np.array([0, 0, 0], dtype=np.int32)   # 没有正样本，默认会抽第0行第0列格子，默认会抽这一层gt_mask里第0个掩码。
            else:
                gt_masks = np.concatenate(gt_masks, axis=0)

            gt_objs_per_layer.append(gt_objs)
            gt_clss_per_layer.append(gt_clss)
            gt_masks_per_layer.append(gt_masks)
            gt_pos_idx_per_layer.append(gt_pos_idx)
        return gt_objs_per_layer, gt_clss_per_layer, gt_masks_per_layer, gt_pos_idx_per_layer

