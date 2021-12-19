from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import cv2
from collections import namedtuple

Corner = namedtuple('Corner', 'x1 y1 x2 y2')
# alias
BBox = Corner
Center = namedtuple('Center', 'x y w h')


def corner2center(corner):
    """ convert (x1, y1, x2, y2) to (cx, cy, w, h)
    Args:
        conrner: Corner or np.array (4*N)
    Return:
        Center or np.array (4 * N)
    """
    if isinstance(corner, Corner):
        x1, y1, x2, y2 = corner
        return Center((x1 + x2) * 0.5, (y1 + y2) * 0.5, (x2 - x1), (y2 - y1))
    else:
        x1, y1, x2, y2 = corner[0], corner[1], corner[2], corner[3]
        x = (x1 + x2) * 0.5
        y = (y1 + y2) * 0.5
        w = x2 - x1
        h = y2 - y1
        return x, y, w, h


def center2corner(center):
    """ convert (cx, cy, w, h) to (x1, y1, x2, y2)
    Args:
        center: Center or np.array (4 * N)
    Return:
        center or np.array (4 * N)
    """
    if isinstance(center, Center):
        x, y, w, h = center
        return Corner(x - w * 0.5, y - h * 0.5, x + w * 0.5, y + h * 0.5)
    else:
        x, y, w, h = center[0], center[1], center[2], center[3]
        x1 = x - w * 0.5
        y1 = y - h * 0.5
        x2 = x + w * 0.5
        y2 = y + h * 0.5
        return x1, y1, x2, y2

class Augmentation:
    def __init__(self, shift, scale, blur, flip, color):
        self.shift = shift
        self.scale = scale
        self.blur = blur
        self.flip = flip
        self.color = color
        self.rgbVar = np.array(
            [[-0.55919361,  0.98062831, - 0.41940627],
             [1.72091413,  0.19879334, - 1.82968581],
             [4.64467907,  4.73710203, 4.88324118]], dtype=np.float32)

    @staticmethod
    def random():
        return np.random.random() * 2 - 1.0

    def _crop_roi(self, image, bbox, out_sz, padding=(0, 0, 0)):
        bbox = [float(x) for x in bbox]
        a = (out_sz-1) / (bbox[2]-bbox[0])
        b = (out_sz-1) / (bbox[3]-bbox[1])
        c = -a * bbox[0]
        d = -b * bbox[1]
        mapping = np.array([[a, 0, c],
                            [0, b, d]]).astype(np.float)
        crop = cv2.warpAffine(image, mapping, (out_sz, out_sz),
                              borderMode=cv2.BORDER_CONSTANT,
                              borderValue=padding)
        return crop

    def _blur_aug(self, image):
        def rand_kernel():
            sizes = np.arange(5, 46, 2)
            size = np.random.choice(sizes)
            kernel = np.zeros((size, size))
            c = int(size/2)
            wx = np.random.random()
            kernel[:, c] += 1. / size * wx
            kernel[c, :] += 1. / size * (1-wx)
            return kernel
        kernel = rand_kernel()
        image = cv2.filter2D(image, -1, kernel)
        return image

    def _color_aug(self, image):
        offset = np.dot(self.rgbVar, np.random.randn(3, 1))
        offset = offset[::-1]  # bgr 2 rgb
        offset = offset.reshape(3)
        image = image - offset
        # we have to restrict the range within 0~255
        image = np.clip(image, 0, 255)

        return image

    def _gray_aug(self, image):
        grayed = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.cvtColor(grayed, cv2.COLOR_GRAY2BGR)
        return image

    def _shift_scale_aug(self, image, bbox, mask, crop_bbox, size):
        im_h, im_w = image.shape[:2]

        # adjust crop bounding box
        crop_bbox_center = corner2center(crop_bbox)
        if self.scale:
            scale_x = (1.0 + Augmentation.random() * self.scale)
            scale_y = (1.0 + Augmentation.random() * self.scale)
            h, w = crop_bbox_center.h, crop_bbox_center.w
            scale_x = min(scale_x, float(im_w) / w)
            scale_y = min(scale_y, float(im_h) / h)
            crop_bbox_center = Center(crop_bbox_center.x,
                                      crop_bbox_center.y,
                                      crop_bbox_center.w * scale_x,
                                      crop_bbox_center.h * scale_y)

        crop_bbox = center2corner(crop_bbox_center)
        if self.shift:
            sx = Augmentation.random() * self.shift
            sy = Augmentation.random() * self.shift

            x1, y1, x2, y2 = crop_bbox

            sx = max(-x1, min(im_w - 1 - x2, sx))
            sy = max(-y1, min(im_h - 1 - y2, sy))

            crop_bbox = Corner(x1 + sx, y1 + sy, x2 + sx, y2 + sy)

        # adjust target bounding box and mask
        x1, y1 = crop_bbox.x1, crop_bbox.y1
        bbox = Corner(bbox.x1 - x1, bbox.y1 - y1,
                      bbox.x2 - x1, bbox.y2 - y1)
        mask = [mask[0] - x1, mask[1] - y1,
                mask[2] - x1, mask[3] - y1]

        # print("shift aug search_box: {}, search_mask: {}, x1: {}, y1] {}".format(bbox, mask, x1, y1))


        if self.scale:
            bbox = Corner(bbox.x1 / scale_x, bbox.y1 / scale_y,
                          bbox.x2 / scale_x, bbox.y2 / scale_y)
            mask = [mask[0] / scale_x, mask[1] / scale_y,
                    mask[2] / scale_x, mask[3] / scale_y]

        if mask[0] < 0:
            mask[0] = 0.0
        if mask[1] < 0:
            mask[1] = 0.0
        if mask[2] > size:
            mask[2] = float(size)
        if mask[3] > size:
            mask[3] = float(size)

        # print("scale aug search_box: {}, search_mask: {}".format(bbox, mask))

        image = self._crop_roi(image, crop_bbox, size)
        return image, bbox, mask

    def _flip_aug(self, image, bbox, mask):
        image = cv2.flip(image, 1)
        width = image.shape[1]
        bbox = Corner(width - 1 - bbox.x2, bbox.y1,
                      width - 1 - bbox.x1, bbox.y2)
        mask = Corner(width - 1 - mask[2], mask[1],
                      width - 1 - mask[0], mask[3])

        return image, bbox, mask

    def __call__(self, image, bbox, mask, size, gray=False):
        shape = image.shape
        crop_bbox = center2corner(Center(shape[0]//2, shape[1]//2,
                                         size-1, size-1))
        # gray augmentation
        if gray:
            image = self._gray_aug(image)

        # shift scale augmentation
        image, bbox, mask = self._shift_scale_aug(image, bbox, mask, crop_bbox, size)

        # color augmentation
        if self.color > np.random.random():
            image = self._color_aug(image)

        # blur augmentation
        if self.blur > np.random.random():
            image = self._blur_aug(image)

        # flip augmentation
        if self.flip and self.flip > np.random.random():
            image, bbox, mask = self._flip_aug(image, bbox, mask)

        return image, bbox, mask
