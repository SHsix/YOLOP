import torch
from PIL import Image
import os
import pdb
import numpy as np
import cv2
from data.mytransforms import find_start_pos
from lib.utils import letterbox_for_img

from tqdm import tqdm
import json

def loader_func(path):
    return Image.open(path)


class LaneTestDataset(torch.utils.data.Dataset):
    def __init__(self, path, list_path, img_transform=None):
        super(LaneTestDataset, self).__init__()
        self.path = path
        self.img_transform = img_transform
        with open(list_path, 'r') as f:
            self.list = f.readlines()
        self.list = [l[1:] if l[0] == '/' else l for l in self.list]  # exclude the incorrect path prefix '/' of CULane
        self.img_size = 640

    def __getitem__(self, index):
        name = self.list[index].split()[0]
        img_path = os.path.join(self.path, name)
        img = loader_func(img_path)
        
    
        
        img0 = cv2.imread(img_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)  # BGR
        #img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
        assert img0 is not None, 'Image Not Found ' + img_path

        h0, w0 = img0.shape[:2]
     

        # Padded resize
        ob_img, ratio, pad = letterbox_for_img(img0, new_shape=self.img_size, auto=True)
        h, w = ob_img.shape[:2]
        shapes = (h0, w0), ((h / h0, w / w0), pad)

        # Convert
        #img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        ob_img = np.ascontiguousarray(ob_img)


        # cv2.imwrite(path + '.letterbox.jpg', 255 * img.transpose((1, 2, 0))[:, :, ::-1])  # save letterbox image
        # return path, img, img0, self.cap, shapes

        if self.img_transform is not None:
            img = self.img_transform[0](img)
            ob_img = self.img_transform[1](ob_img)
            
        if ob_img.ndimension() == 3:
            ob_img = ob_img.unsqueeze(0)

        return img, name, ob_img, img0, shapes

    def __len__(self):
        return len(self.list)


class LaneClsDataset(torch.utils.data.Dataset):
    def __init__(self, path, list_path, img_transform = None,target_transform = None, griding_num=50, load_name = False,
                row_anchor = None,use_aux=False,segment_transform=None, cv_transform = None, num_lanes = 4):
        super(LaneClsDataset, self).__init__()
        self.img_transform = img_transform
        self.target_transform = target_transform
        self.segment_transform = segment_transform
        self.cv_transform = cv_transform
        self.path = path
        self.griding_num = griding_num
        self.load_name = load_name
        self.use_aux = use_aux
        self.num_lanes = num_lanes
        self.inputsize = [640, 640] 

        with open(list_path, 'r') as f:
            self.list = f.readlines()

        self.row_anchor = row_anchor
        self.row_anchor.sort()

    def __getitem__(self, index):
        l = self.list[index]
        l_info = l.split()
        img_name, label_name = l_info[0], l_info[1]
        if img_name[0] == '/':
            img_name = img_name[1:]
            label_name = label_name[1:]

        img_path = os.path.join(self.path, img_name)
        label_path = os.path.join(self.path, label_name)

        img = loader_func(img_path)
        label = loader_func(label_path)

        ob_label_path = os.path.join(self.path, 'object')
        ob_label_path = os.path.join(ob_label_path, img_name)[:-3] + 'json'
        det_label = self._get_ob_label(ob_label_path)

        lane_pts = self._get_index(label)
        # get the coordinates of lanes at row anchors

  
        np_img = cv2.imread(img_path,cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        cv_img = cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB)

        resized_shape = self.inputsize
        if isinstance(resized_shape, list):
            resized_shape = max(resized_shape)
        h0, w0 = cv_img.shape[:2]  # orig hw
        r = resized_shape / max(h0, w0)  # resize image to img_size
        if r != 1:  # always resize down, only resize up if training with augmentation
            interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
            cv_img = cv2.resize(cv_img, (int(w0 * r), int(h0 * r)), interpolation=interp)
            # seg_label = cv2.resize(seg_label, (int(w0 * r), int(h0 * r)), interpolation=interp)
            # lane_label = cv2.resize(lane_label, (int(w0 * r), int(h0 * r)), interpolation=interp)
        h, w = cv_img.shape[:2]
        
        cv_img, ratio, pad = letterbox_for_img(cv_img, resized_shape, auto=True, scaleup=True)
        shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling
        # ratio = (w / w0, h / h0)
        # print(resized_shape)
        
        labels=[]
        
        if det_label.size > 0:
            # Normalized xywh to pixel xyxy format
            labels = det_label.copy()
            labels[:, 1] = ratio[0] * w * (det_label[:, 1] - det_label[:, 3] / 2) + pad[0]  # pad width
            labels[:, 2] = ratio[1] * h * (det_label[:, 2] - det_label[:, 4] / 2) + pad[1]  # pad height
            labels[:, 3] = ratio[0] * w * (det_label[:, 1] + det_label[:, 3] / 2) + pad[0]
            labels[:, 4] = ratio[1] * h * (det_label[:, 2] + det_label[:, 4] / 2) + pad[1]

        labels_out = torch.zeros((len(labels), 6))
        if len(labels):
            labels_out[:, 1:] = torch.from_numpy(labels)
        # Convert
        # img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        # img = img.transpose(2, 0, 1)
        cv_img = np.ascontiguousarray(cv_img)
        if self.cv_transform:
            cv_img = self.cv_transform(cv_img)


        w, h = img.size
        cls_label = self._grid_pts(lane_pts, self.griding_num, w)
        # make the coordinates to classification label
        if self.use_aux:
            assert self.segment_transform is not None
            seg_label = self.segment_transform(label)

        # if self.img_transform is not None:
        #     img = self.img_transform(img)


        if self.use_aux:
            return cv_img, cls_label, seg_label, labels_out, shapes
        if self.load_name:
            return cv_img, cls_label, img_name
        return cv_img, cls_label, labels_out, shapes


        # if self.use_aux:
        #     return img, cls_label, seg_label, cv_img, labels_out, shapes
        # if self.load_name:
        #     return img, cls_label, img_name
        # return img, cls_label, cv_img, labels_out, shapes



    def __len__(self):
        return len(self.list)

    def _get_ob_label(self, ob_label_path):
        with open(ob_label_path, 'r') as f:
            label = json.load(f)
            
        data = label['objects']
        gt = np.zeros((len(data), 5))

        for idx, obj in enumerate(data):
            category = obj['relative_coordinates']
            x = float(category['center_x'])
            y = float(category['center_y'])
            w = float(category['width'])
            h = float(category['height'])
            gt[idx][0] = 0
            gt[idx][1:] = x, y, w, h
            

        return gt

    def _grid_pts(self, pts, num_cols, w):
        # pts : numlane,n,2
        num_lane, n, n2 = pts.shape
        col_sample = np.linspace(0, w - 1, num_cols)

        assert n2 == 2
        to_pts = np.zeros((n, num_lane))
        for i in range(num_lane):
            pti = pts[i, :, 1]
            to_pts[:, i] = np.asarray(
                [int(pt // (col_sample[1] - col_sample[0])) if pt != -1 else num_cols for pt in pti])
        return to_pts.astype(int)

    def _get_index(self, label):
        w, h = label.size

        if h != 256:
            scale_f = lambda x : int((x * 1.0/256) * h)
            sample_tmp = list(map(scale_f,self.row_anchor))

        all_idx = np.zeros((self.num_lanes,len(sample_tmp),2))
        for i,r in enumerate(sample_tmp):
            label_r = np.asarray(label)[int(round(r))]
            for lane_idx in range(1, self.num_lanes + 1):
                pos = np.where(label_r == lane_idx)[0]
                if len(pos) == 0:
                    all_idx[lane_idx - 1, i, 0] = r
                    all_idx[lane_idx - 1, i, 1] = -1
                    continue
                pos = np.mean(pos)
                all_idx[lane_idx - 1, i, 0] = r
                all_idx[lane_idx - 1, i, 1] = pos

        # data augmentation: extend the lane to the boundary of image

        all_idx_cp = all_idx.copy()
        for i in range(self.num_lanes):
            if np.all(all_idx_cp[i,:,1] == -1):
                continue
            # if there is no lane

            valid = all_idx_cp[i,:,1] != -1
            # get all valid lane points' index
            valid_idx = all_idx_cp[i,valid,:]
            # get all valid lane points
            if valid_idx[-1,0] == all_idx_cp[0,-1,0]:
                # if the last valid lane point's y-coordinate is already the last y-coordinate of all rows
                # this means this lane has reached the bottom boundary of the image
                # so we skip
                continue
            if len(valid_idx) < 6:
                continue
            # if the lane is too short to extend

            valid_idx_half = valid_idx[len(valid_idx) // 2:,:]
            p = np.polyfit(valid_idx_half[:,0], valid_idx_half[:,1],deg = 1)
            start_line = valid_idx_half[-1,0]
            pos = find_start_pos(all_idx_cp[i,:,0],start_line) + 1
            
            fitted = np.polyval(p,all_idx_cp[i,pos:,0])
            fitted = np.array([-1  if y < 0 or y > w-1 else y for y in fitted])

            assert np.all(all_idx_cp[i,pos:,1] == -1)
            all_idx_cp[i,pos:,1] = fitted
        if -1 in all_idx[:, :, 0]:
            pdb.set_trace()
        return all_idx_cp
