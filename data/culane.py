import numpy as np
import json
import os
from .AutoDriveDataset import AutoDriveDataset
from .convert import convert, id_dict, id_dict_single
from tqdm import tqdm
     # just detect vehicle

class Culane(AutoDriveDataset):
    def __init__(self, cfg, is_train, inputsize, transform=None, \
        row_anchor = None,griding_num=None, use_aux=None, num_lanes = None):
        super().__init__(cfg, is_train, inputsize, transform, row_anchor, \
            griding_num, use_aux, num_lanes)
        self.db = self._get_db()
        self.cfg = cfg

    def _get_db(self):
        """
        get database from the annotation file

        Inputs:

        Returns:
        gt_db: (list)database   [a,b,c,...]
                a: (dictionary){'image':, 'information':, ......}
        image: image path
        mask: path of the segmetation label
        label: [cls_id, center_x//256, center_y//256, w//256, h//256] 256=IMAGE_SIZE
        """
        print('building CULane Object database...')
        gt_db = []
        height, width = self.shapes
        for dir in tqdm(list(self.img_list)):
            if self.is_train:
                dir_info = dir.split()
                img_name, label_name = dir_info[0], dir_info[1]
                if img_name[0] == '/':
                    img_name = img_name[1:]
                    label_name = label_name[1:]
            
                img_path = os.path.join(self.img_path, img_name)
                lane_label_path = os.path.join(self.img_path, label_name)
                ob_label_path = os.path.join(self.ob_label_root, img_name)[:-3] + 'json'

            else :
                img_name= dir.split()[0]
                if img_name[0] == '/':
                    img_name = img_name[1:]
            
                img_path = os.path.join(self.img_path, img_name)
                ob_label_path = os.path.join(self.ob_label_root, img_name)[:-3] + 'json'

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

            if self.is_train:
                rec = [{
                    'image': img_path,
                    'ob_label': gt,
                    'lane_label' : lane_label_path
                }]
            else:
                rec = [{
                    'image': img_path,
                    'image_name': img_name,
                    'ob_label': gt
                }]
            gt_db += rec
        print('database build finish')
        return gt_db

    def evaluate(self, cfg, preds, output_dir, *args, **kwargs):
        """  
        """
        pass