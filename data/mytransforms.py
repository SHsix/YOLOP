import numbers
import random
import numpy as np
from PIL import Image, ImageOps, ImageFilter
#from config import cfg
import torch
import pdb
import cv2

# ===============================img tranforms============================

class Compose2(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask, bbx=None):
        if bbx is None:
            for t in self.transforms:
                img, mask = t(img, mask)
            return img, mask
        for t in self.transforms:
            img, mask, bbx = t(img, mask, bbx)
        return img, mask, bbx

class FreeScale(object):
    def __init__(self, size):
        self.size = size  # (h, w)

    def __call__(self, img, mask):
        return img.resize((self.size[1], self.size[0]), Image.BILINEAR), mask.resize((self.size[1], self.size[0]), Image.NEAREST)

class FreeScaleMask(object):
    def __init__(self,size):
        self.size = size
    def __call__(self,mask):
        return mask.resize((self.size[1], self.size[0]), Image.NEAREST)

class Scale(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, mask):
        if img.size != mask.size:
            print(img.size)
            print(mask.size)
        assert img.size == mask.size
        w, h = img.size
        if (w <= h and w == self.size) or (h <= w and h == self.size):
            return img, mask
        if w < h:
            ow = self.size
            oh = int(self.size * h / w)
            return img.resize((ow, oh), Image.BILINEAR), mask.resize((ow, oh), Image.NEAREST)
        else:
            oh = self.size
            ow = int(self.size * w / h)
            return img.resize((ow, oh), Image.BILINEAR), mask.resize((ow, oh), Image.NEAREST)


class RandomRotate(object):
    """Crops the given PIL.Image at a random location to have a region of
    the given size. size can be a tuple (target_height, target_width)
    or an integer, in which case the target will be of a square shape (size, size)
    """

    def __init__(self, angle):
        self.angle = angle

    def __call__(self, image, label, targets):
        image = Image.fromarray(image)
        assert label is None or image.size == label.size
        w, h = image.size

        angle = random.randint(0, self.angle * 2) - self.angle

        label = label.rotate(angle, resample=Image.NEAREST)
        image = image.rotate(angle, resample=Image.BILINEAR)



        # For object detection Rotation
        R = np.eye(3)
        # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations

        R[:2] = cv2.getRotationMatrix2D(angle=angle, center=(w//2, h//2), scale=1)

        n = len(targets)
        if n:
            xy = np.ones((n * 4, 3))
            xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
            xy = xy @ R.T
            xy = xy[:, :2].reshape(n, 8)

            # create new boxes
            x = xy[:, [0, 2, 4, 6]]
            y = xy[:, [1, 3, 5, 7]]
            xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T
            # clip boxes
            xy[:, [0, 2]] = xy[:, [0, 2]].clip(0, w)
            xy[:, [1, 3]] = xy[:, [1, 3]].clip(0, h)
            
            i = _box_candidates(box1=targets[:, 1:5].T, box2=xy.T)
            targets = targets[i]
            targets[:, 1:5] = xy[i]

        return np.array(image), label, targets


def _box_candidates(box1, box2, wh_thr=2, ar_thr=20, area_thr=0.1):  # box1(4,n), box2(4,n)
    # Compute candidate boxes: box1 before augment, box2 after augment, wh_thr (pixels), aspect_ratio_thr, area_ratio
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    ar = np.maximum(w2 / (h2 + 1e-16), h2 / (w2 + 1e-16))  # aspect ratio
    return (w2 > wh_thr) & (h2 > wh_thr) & (w2 * h2 / (w1 * h1 + 1e-16) > area_thr) & (ar < ar_thr)  # candidates


# ===============================label tranforms============================

class DeNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


class MaskToTensor(object):
    def __call__(self, img):
        return torch.from_numpy(np.array(img, dtype=np.int32)).long()


def find_start_pos(row_sample,start_line):
    # row_sample = row_sample.sort()
    # for i,r in enumerate(row_sample):
    #     if r >= start_line:
    #         return i
    l,r = 0,len(row_sample)-1
    while True:
        mid = int((l+r)/2)
        if r - l == 1:
            return r
        if row_sample[mid] < start_line:
            l = mid
        if row_sample[mid] > start_line:
            r = mid
        if row_sample[mid] == start_line:
            return mid

class RandomLROffsetLABEL(object):
    def __init__(self,max_offset):
        self.max_offset = max_offset
    def __call__(self,img,label,targets):
        offset = np.random.randint(-self.max_offset,self.max_offset)
        h, w = img.shape[:2]

        img = np.array(img)
        if offset > 0:
            img[:,offset:,:] = img[:,0:w-offset,:]
            img[:,:offset,:] = 0
        if offset < 0:
            real_offset = -offset
            img[:,0:w-real_offset,:] = img[:,real_offset:,:]
            img[:,w-real_offset:,:] = 0

        label = np.array(label)
        if offset > 0:
            label[:,offset:] = label[:,0:w-offset]
            label[:,:offset] = 0
        if offset < 0:
            real_offset = -offset
            label[:,0:w-real_offset] = label[:,real_offset:]
            label[:,w-real_offset:] = 0

        n = len(targets)
        if n:
            # xy = np.ones((n * 4, 3))
            # xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
            # xy = xy[:, :2].reshape(n, 8)

            # # create new boxes
            # x = xy[:, [0, 2, 4, 6]]
            # y = xy[:, [1, 3, 5, 7]]
            # xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T
            # # clip boxes
            xy = np.ones((n, 4))
            xy = targets[:, [1,2,3,4]]
            if offset > 0:
                xy[:, [0, 2]] += offset
                xy[:, [0, 2]] = xy[:, [0, 2]].clip(0, w)
            if offset < 0:
                real_offset = -offset
                xy[:, [0, 2]] -= real_offset
                xy[:, [0, 2]] = xy[:, [0, 2]].clip(0, w)

            
            i = _box_candidates(box1=targets[:, 1:5].T, box2=xy.T)
            targets = targets[i]
            targets[:, 1:5] = xy[i]
        return img, Image.fromarray(label), targets

class RandomUDoffsetLABEL(object):
    def __init__(self,max_offset):
        self.max_offset = max_offset
    def __call__(self,img,label, targets):
        offset = np.random.randint(-self.max_offset,self.max_offset)

        h, w = img.shape[:2]

        img = np.array(img)
        if offset > 0:
            img[offset:,:,:] = img[0:h-offset,:,:]
            img[:offset,:,:] = 0
        if offset < 0:
            real_offset = -offset
            img[0:h-real_offset,:,:] = img[real_offset:,:,:]
            img[h-real_offset:,:,:] = 0

        label = np.array(label)
        if offset > 0:
            label[offset:,:] = label[0:h-offset,:]
            label[:offset,:] = 0
        if offset < 0:
            real_offset = -offset
            label[0:h-real_offset,:] = label[real_offset:,:]
            label[h-real_offset:,:] = 0



        n = len(targets)
        if n:
            # xy = np.ones((n * 4, 3))
            # xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
            # xy = xy[:, :2].reshape(n, 8)

            # # create new boxes
            # x = xy[:, [0, 2, 4, 6]]
            # y = xy[:, [1, 3, 5, 7]]
            # xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T
            # # clip boxes
            xy = np.ones((n, 4))
            xy = targets[:, [1,2,3,4]]
            if offset > 0:
                xy[:, [1, 3]] += offset
                xy[:, [1, 3]] = xy[:, [1, 3]].clip(0, h)
            if offset < 0:
                real_offset = -offset
                xy[:, [1, 3]] -= real_offset
                xy[:, [1, 3]] = xy[:, [1, 3]].clip(0, h)
                
            i = _box_candidates(box1=targets[:, 1:5].T, box2=xy.T)
            targets = targets[i]
            targets[:, 1:5] = xy[i]
        return img,Image.fromarray(label), targets
