import cv2
import numpy as np
from PIL import Image
# np.set_printoptions(threshold=np.inf)
import random
import torch
import torchvision.transforms as transforms
# from visualization import plot_img_and_mask,plot_one_box,show_seg_result
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from lib.utils import letterbox_for_img, augment_hsv, random_perspective, xyxy2xywh, cutout
from data.mytransforms import find_start_pos
import pdb

def loader_func(path):
    return Image.open(path)

class AutoDriveDataset(Dataset):
    """
    A general Dataset for some common function
    """
    def __init__(self, cfg, is_train, inputsize=640, transform=None, \
        row_anchor = None,griding_num=None, use_aux=None, num_lanes = None):
        """
        initial all the characteristic

        Inputs:
        -cfg: configurations
        -is_train(bool): whether train set or not
        -transform: ToTensor and Normalize
        
        Returns:
        None
        """
        self.is_train = is_train
        self.cfg = cfg
        self.transform = transform
        self.inputsize = inputsize
        self.Tensor = transforms.ToTensor()
        self.img_path = Path(cfg.DATASET.DATAROOT)
        self.ob_label_root = Path(cfg.DATASET.LABELROOT)
        self.griding_num = griding_num
        self.row_anchor = row_anchor
        self.use_aux = use_aux
        self.num_lanes = num_lanes

        if is_train:
            self.cv_transform = transform[0]
            self.simu_transform = transform[1]
            self.segment_transform = transform[2]

        else:
            self.cv_transform = transform[0]



        if is_train:
            indicator = cfg.DATASET.TRAIN_SET
        else:
            indicator = cfg.DATASET.TEST_SET
        self.img_root = self.img_path / indicator
        self.img_list = []
        with open(self.img_root, 'r') as f:
            self.img_list = f.readlines()
        self.db = []
    
        self.shapes = np.array(cfg.DATASET.ORG_IMG_SIZE)
        self.Tensor = transforms.ToTensor()
        # self.img_root.iterdir()


        # self.label_root = label_root / indicator
        # self.mask_root = mask_root / indicator
        # self.lane_root = lane_root / indicator
        # self.label_list = self.label_root.iterdir()


        self.data_format = cfg.DATASET.DATA_FORMAT

        self.scale_factor = cfg.DATASET.SCALE_FACTOR
        self.rotation_factor = cfg.DATASET.ROT_FACTOR
        self.flip = cfg.DATASET.FLIP
        self.color_rgb = cfg.DATASET.COLOR_RGB

        # self.target_type = cfg.MODEL.TARGET_TYPE
    
    def _get_db(self):
        """
        finished on children Dataset(for dataset which is not in Bdd100k format, rewrite children Dataset)
        """
        raise NotImplementedError

    def evaluate(self, cfg, preds, output_dir):
        """
        finished on children dataset
        """
        raise NotImplementedError
    
    def __len__(self,):
        """
        number of objects in the dataset
        """
        return len(self.db)

    def __getitem__(self, idx):
        """
        Get input and groud-truth from database & add data augmentation on input

        Inputs:
        -idx: the index of image in self.db(database)(list)
        self.db(list) [a,b,c,...]
        a: (dictionary){'image':, 'information':}

        Returns:
        -image: transformed image, first passed the data augmentation in __getitem__ function(type:numpy), then apply self.transform
        -target: ground truth(det_gt,seg_gt)

        function maybe useful
        cv2.imread
        cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
        cv2.warpAffine
        """
        data = self.db[idx]

        img = cv2.imread(data["image"], cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        det_label = data["ob_label"] # xywh

        if self.is_train:
            lane_label = loader_func(data["lane_label"])
            origin_labels=[]
            h0, w0 = img.shape[:2]  # orig hw
            
            if det_label.size > 0:
                # Normalized xywh to pixel xyxy format
                origin_labels = det_label.copy()
                origin_labels[:, 1] = w0 * (det_label[:, 1] - det_label[:, 3] / 2) 
                origin_labels[:, 2] = h0 * (det_label[:, 2] - det_label[:, 4] / 2)  
                origin_labels[:, 3] = w0 * (det_label[:, 1] + det_label[:, 3] / 2)
                origin_labels[:, 4] = h0 * (det_label[:, 2] + det_label[:, 4] / 2)
            # img = cv2.circle(img, (int(origin_labels[0, 1]), int(origin_labels[0, 2])), 5, (255, 0, 0), 3)
            # img = cv2.circle(img, (int(origin_labels[0, 3]), int(origin_labels[0, 4])), 5, (255, 0, 0), 3)
            # cv2.imwrite('/home/YOLOP/before.jpg', img)
            if self.simu_transform is not None:
                img, lane_label, origin_labels = self.simu_transform(img,lane_label, origin_labels)

            # img = cv2.circle(img, (int(origin_labels[0, 1]), int(origin_labels[0, 2])), 5, (0, 255, 0), 3)
            # img = cv2.circle(img, (int(origin_labels[0, 3]), int(origin_labels[0, 4])), 5, (0, 255, 0), 3)
            # cv2.imwrite('/home/YOLOP/After.jpg', img)

            resized_shape = self.inputsize
            if isinstance(resized_shape, list):
                resized_shape = max(resized_shape)

            r = resized_shape / max(h0, w0)  # resize image to img_size
            if r != 1:  # always resize down, only resize up if training with augmentation
                interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
                img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=interp)
            
            h, w = img.shape[:2]

            img, ratio, pad = letterbox_for_img(img, resized_shape, auto=True, scaleup=self.is_train)
            shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling
            # ratio = (w / w0, h / h0)
            # print(resized_shape)
            
            labels = []
            if len(origin_labels):
                # Add the pad to origin label
                labels = origin_labels.copy()
                labels[:, 1] = (w / w0) * origin_labels[:, 1] + pad[0]  # pad width
                labels[:, 2] = (h / h0) * origin_labels[:, 2] + pad[1]  # pad width
                labels[:, 3] = (w / w0) * origin_labels[:, 3] + pad[0]  # pad width
                labels[:, 4] = (h / h0) * origin_labels[:, 4] + pad[1]  # pad width

        else:
            h0, w0 = img.shape[:2] 
            resized_shape = self.inputsize
            if isinstance(resized_shape, list):
                resized_shape = max(resized_shape)

            r = resized_shape / max(h0, w0)  # resize image to img_size
            if r != 1:  # always resize down, only resize up if training with augmentation
                interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
                img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=interp)
            
            h, w = img.shape[:2]

            img, ratio, pad = letterbox_for_img(img, resized_shape, auto=True, scaleup=self.is_train)
            shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling



            det_label = data["ob_label"] # xywh
            labels = []
            if det_label.size > 0:
                # Normalized xywh to pixel xyxy format
                labels = det_label.copy()
                labels[:, 1] = ratio[0] * w * (det_label[:, 1] - det_label[:, 3] / 2) + pad[0]  # pad width
                labels[:, 2] = ratio[1] * h * (det_label[:, 2] - det_label[:, 4] / 2) + pad[1]  # pad height
                labels[:, 3] = ratio[0] * w * (det_label[:, 1] + det_label[:, 3] / 2) + pad[0]
                labels[:, 4] = ratio[1] * h * (det_label[:, 2] + det_label[:, 4] / 2) + pad[1]
 
            img_name = data["image_name"]


        if len(labels):
            # convert xyxy to xywh
            labels[:, 1:5] = xyxy2xywh(labels[:, 1:5])

            # Normalize coordinates 0 - 1
            labels[:, [2, 4]] /= img.shape[0]  # height
            labels[:, [1, 3]] /= img.shape[1]  # width
    

        labels_out = torch.zeros((len(labels), 6))
        if len(labels):
            labels_out[:, 1:] = torch.from_numpy(labels)
        # Convert
        # img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        # img = img.transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        # seg_label = np.ascontiguousarray(seg_label)
        # if idx == 0:
        #     print(seg_label[:,:,0])



        if self.use_aux and self.is_train:
            lane_pts = self._get_index(lane_label)
            w, h = lane_label.size

            
            cls_label = self._grid_pts(lane_pts, self.griding_num, w)
            cls_label = self.Tensor(cls_label)
            cls_label = cls_label.squeeze()

            lane_label = np.array(lane_label)
            h0, w0 = lane_label.shape[:2] 
            resized_shape = self.inputsize
            if isinstance(resized_shape, list):
                resized_shape = max(resized_shape)

            r = resized_shape / max(h0, w0)  # resize image to img_size
            if r != 1:  # always resize down, only resize up if training with augmentation
                interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
                lane_label = cv2.resize(lane_label, (int(w0 * r), int(h0 * r)), interpolation=interp)
            
            h, w = lane_label.shape[:2]

            lane_label, ratio, pad = letterbox_for_img(lane_label, resized_shape, auto=True, scaleup=self.is_train, color=(0, 0, 0))
            shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling
            cv2.imwrite('/home/YOLOP/test.jpg', lane_label)

            lane_label = Image.fromarray(lane_label)
            assert self.segment_transform is not None
            seg_label = self.segment_transform(lane_label)

        # target = [labels_out, seg_label, lane_label]

        target = labels_out
        img = self.cv_transform(img)



        if self.is_train:
            return img, [target, cls_label, seg_label]
        else:
            return img, img_name, target, data["image"], shapes
        #, data["image"]


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
        sample_tmp = self.row_anchor
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

    def select_data(self, db):
        """
        You can use this function to filter useless images in the dataset

        Inputs:
        -db: (list)database

        Returns:
        -db_selected: (list)filtered dataset
        """
        db_selected = ...
        return db_selected

    @staticmethod
    def collate_fn(batch):
        img, label= zip(*batch)
        tar_det, cls_det, seg_det = [], [], [] 
        for i, l in enumerate(label):
            l_tar, l_cls ,l_seg  = l
            l_tar[:, 0] = i  # add target image index for build_targets()
            tar_det.append(l_tar)
            cls_det.append(l_cls)
            seg_det.append(l_seg)
     
        return torch.stack(img, 0), [torch.cat(tar_det, 0), torch.stack(cls_det, 0), torch.stack(seg_det, 0)]

    @staticmethod
    def test_collate_fn(batch):
        img, img_name, target, img_pth, shapes= zip(*batch)
        tar_det = []
        for i, l in enumerate(target):
            l_tar = l
            l_tar[:, 0] = i  # add target image index for build_targets()
            tar_det.append(l_tar)
            
     
        return torch.stack(img, 0), img_name, torch.cat(tar_det, 0), img_pth, shapes

