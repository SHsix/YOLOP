from lib.utils.utils import time_synchronized
from lib.core.evaluate import SegmentationMetric
from lib.utils import check_anchor_order
from torch.nn import Upsample
from lib.models.common import Conv, SPP, Bottleneck, BottleneckCSP, Focus, Concat, Detect, SharpenConv, Detect_lane, Aux_lane
# from lib.utils import initialize_weights
import torch
from torch import tensor
import torch.nn as nn
import sys
import os
import math
import sys
from lib.models.backbone import resnet

import numpy as np
sys.path.append(os.getcwd())
# sys.path.append("lib/models")
# sys.path.append("lib/utils")
# sys.path.append("/workspace/wh/projects/DaChuang")
# from lib.models.common2 import DepthSeperabelConv2d as Conv
# from lib.models.common2 import SPP, Bottleneck, BottleneckCSP, Focus, Concat, Detect


# The lane line and the driving area segment branches without share information with each other and without link
YOLOP = [
    [24, 25],  # Det_out_idx, Da_Segout_idx, LL_Segout_idx
    [-1, Focus, [3, 32, 3]],  # 0
    [-1, Conv, [32, 64, 3, 2]],  # 1
    [-1, BottleneckCSP, [64, 64, 1]],  # 2
    [-1, Conv, [64, 128, 3, 2]],  # 3
    [-1, BottleneckCSP, [128, 128, 3]],  # 4
    [-1, Conv, [128, 256, 3, 2]],  # 5
    [-1, BottleneckCSP, [256, 256, 3]],  # 6
    [-1, Conv, [256, 512, 3, 2]],  # 7
    [-1, SPP, [512, 512, [5, 9, 13]]],  # 8
    [-1, BottleneckCSP, [512, 512, 1, False]],  # 9
    [-1, Conv, [512, 256, 1, 1]],  # 10
    [-1, Upsample, [None, 2, 'nearest']],  # 11
    [[-1, 6], Concat, [1]],  # 12
    [-1, BottleneckCSP, [512, 256, 1, False]],  # 13
    [-1, Conv, [256, 128, 1, 1]],  # 14
    [-1, Upsample, [None, 2, 'nearest']],  # 15
    [[-1, 4], Concat, [1]],  # 16         #Encoder

    [-1, BottleneckCSP, [256, 128, 1, False]],  # 17
    [-1, Conv, [128, 128, 3, 2]],  # 18
    [[-1, 14], Concat, [1]],  # 19
    [-1, BottleneckCSP, [256, 256, 1, False]],  # 20
    [-1, Conv, [256, 256, 3, 2]],  # 21
    [[-1, 10], Concat, [1]],  # 22
    [-1, BottleneckCSP, [512, 512, 1, False]],  # 23
    [[17, 20, 23], Detect,  [1, [[3, 9, 5, 11, 4, 20], [7, 18, 6, 39, 12, 31],
                                 [19, 50, 38, 81, 68, 157]], [128, 256, 512]]],  # Detection head 24

    [[3,5,7], Aux_lane, []],  # 25
    
    # [-1, Conv, [128, 128, 3, 1]],  # 26
    # [-1, Conv, [128, 128, 3, 1]],  # 27
    # [-1, Conv, [128, 5, 3, 1]],  # 28

    # [-1, Upsample, [None, 2, 'nearest']],  # 29
    # [-1, Conv, [32, 16, 3, 1]],  # 30
    # [-1, BottleneckCSP, [16, 8, 1, False]],  # 31
    # [-1, Upsample, [None, 2, 'nearest']],  # 32
    # [-1, Conv, [8, 5, 3, 1]],  # 33 Driving area segmentation head

    # [16, BottleneckCSP, [256, 128, 1, False]],  # 29
    # [-1, Conv, [128, 128, 3, 2]],  # 30
    # [[-1, 14], Concat, [1]],  # 31
    # [-1, BottleneckCSP, [256, 256, 1, False]],  # 32
    # [-1, Conv, [256, 256, 3, 2]],  # 33
    # [[-1, 10], Concat, [1]],  # 34
    # [-1, BottleneckCSP, [512, 512, 1, False]],  # 35
    [7, Detect_lane, [121, 18, 4]]  # 36
]

class conv_bn_relu(torch.nn.Module):
    def __init__(self,in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,bias=False):
        super(conv_bn_relu,self).__init__()
        self.conv = torch.nn.Conv2d(in_channels,out_channels, kernel_size, 
            stride = stride, padding = padding, dilation = dilation,bias = bias)
        self.bn = torch.nn.BatchNorm2d(out_channels)
        self.relu = torch.nn.ReLU()

    def forward(self,x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class MCnet(nn.Module):
    def __init__(self, pretrained=True, backbone='18', cls_dim=(37, 10, 4), use_aux=False):
        super(MCnet, self).__init__()
        layers, save = [], []
        self.nc = 1

        self.cls_dim = cls_dim # (num_gridding, num_cls_per_lane, num_of_lanes)
        # num_cls_per_lane is the number of row anchors
        self.use_aux = use_aux
        self.total_dim = np.prod(cls_dim)

        self.aux_seg = use_aux
        
        self.model = resnet(backbone, pretrained=pretrained)
        self.yolo = Detect(1, [[3, 9, 5, 11, 4, 20], [7, 18, 6, 39, 12, 31],
                                 [19, 50, 38, 81, 68, 157]], [128, 256, 512])
        self.neck_ob1 = torch.nn.Sequential(
            conv_bn_relu(128, 128, kernel_size=3, stride=1, padding=1) if backbone in ['34','18'] else conv_bn_relu(512, 128, kernel_size=3, stride=1, padding=1),
            conv_bn_relu(128,128,3,padding=1),
            conv_bn_relu(128,128,3,padding=1),
            conv_bn_relu(128,128,3,padding=1),
        )
        self.neck_ob2 = torch.nn.Sequential(
            conv_bn_relu(256, 128, kernel_size=3, stride=1, padding=1) if backbone in ['34','18'] else conv_bn_relu(1024, 128, kernel_size=3, stride=1, padding=1),
            conv_bn_relu(128,128,3,padding=1),
            conv_bn_relu(128,256,3,padding=1),
        )
        self.neck_ob3 = torch.nn.Sequential(
            conv_bn_relu(512, 128, kernel_size=3, stride=1, padding=1) if backbone in ['34','18'] else conv_bn_relu(2048, 128, kernel_size=3, stride=1, padding=1),
            conv_bn_relu(128,512,3,padding=1),
        )
        initialize_weights(self.neck_ob1,self.neck_ob2,self.neck_ob3)

        if self.use_aux:
            self.aux_header2 = torch.nn.Sequential(
                conv_bn_relu(128, 128, kernel_size=3, stride=1, padding=1) if backbone in ['34','18'] else conv_bn_relu(512, 128, kernel_size=3, stride=1, padding=1),
                conv_bn_relu(128,128,3,padding=1),
                conv_bn_relu(128,128,3,padding=1),
                conv_bn_relu(128,128,3,padding=1),
            )
            self.aux_header3 = torch.nn.Sequential(
                conv_bn_relu(256, 128, kernel_size=3, stride=1, padding=1) if backbone in ['34','18'] else conv_bn_relu(1024, 128, kernel_size=3, stride=1, padding=1),
                conv_bn_relu(128,128,3,padding=1),
                conv_bn_relu(128,128,3,padding=1),
            )
            self.aux_header4 = torch.nn.Sequential(
                conv_bn_relu(512, 128, kernel_size=3, stride=1, padding=1) if backbone in ['34','18'] else conv_bn_relu(2048, 128, kernel_size=3, stride=1, padding=1),
                conv_bn_relu(128,128,3,padding=1),
            )
            self.aux_combine = torch.nn.Sequential(
                conv_bn_relu(384, 256, 3,padding=2,dilation=2),
                conv_bn_relu(256, 128, 3,padding=2,dilation=2),
                conv_bn_relu(128, 128, 3,padding=2,dilation=2),
                conv_bn_relu(128, 128, 3,padding=4,dilation=4),
                torch.nn.Conv2d(128, cls_dim[-1] + 1,1)
                # output : n, num_of_lanes+1, h, w
            )
            initialize_weights(self.aux_header2,self.aux_header3,self.aux_header4,self.aux_combine)
        
        self.cls = torch.nn.Sequential(
            torch.nn.Linear(8 * 8 * 20, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, self.total_dim),
        )
        self.pool = torch.nn.Conv2d(512,8,1) if backbone in ['34','18'] else torch.nn.Conv2d(2048,8,1)


        # set stride、anchor for detector
        Detector = self.yolo  # detector
        if isinstance(Detector, Detect):
            s = 128  # 2x min stride
            # for x in self.forward(torch.zeros(1, 3, s, s)):
            #     print (x.shape)
            with torch.no_grad():
                model_out = self.forward(torch.zeros(1, 3, s, s))
                detects = model_out[0]
                Detector.stride = torch.tensor(
                    [s / x.shape[-2] for x in detects])  # forward
            # print("stride"+str(Detector.stride ))
            # Set the anchors for the corresponding scale
            Detector.anchors /= Detector.stride.view(-1, 1, 1)
            check_anchor_order(Detector)
            self.stride = Detector.stride
            self._initialize_biases()


    def forward(self, x):

        lane_pred = []
        
        x2,x3,fea = self.model(x)
        # ob_x2 = self.neck_ob1(x2)
        # ob_x3 = self.neck_ob2(x3)
        # ob_fea = self.neck_ob3(fea)
        # if x.shape[-1] == 128:
        #     object_pred = self.yolo([ob_x2, ob_x3, ob_fea])
        #     return [object_pred]
        # object_pred = self.yolo([ob_x2, ob_x3, ob_fea])
        if x.shape[-1] == 128:
            object_pred = self.yolo([x2, x3, fea])
            return [object_pred]
        object_pred = self.yolo([x2, x3, fea])


        if self.use_aux:
            x2 = self.aux_header2(x2)
            x3 = self.aux_header3(x3)
            x3 = torch.nn.functional.interpolate(x3,scale_factor = 2,mode='bilinear')
            x4 = self.aux_header4(fea)
            x4 = torch.nn.functional.interpolate(x4,scale_factor = 4,mode='bilinear')
            aux_seg = torch.cat([x2,x3,x4],dim=1)
            aux_seg = self.aux_combine(aux_seg)
            lane_pred.append(aux_seg)
        else:
            aux_seg = None
        
        fea = self.pool(fea).view(-1, 8 * fea.shape[-1] * fea.shape[-2])
        group_cls = self.cls(fea).view(-1, *self.cls_dim)
        lane_pred.insert(0, group_cls)

        return [object_pred, lane_pred]

    # initialize biases into Detect(), cf is class frequency
    def _initialize_biases(self, cf=None):
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        # m = self.model[-1]  # Detect() module
        m = self.yolo  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            # obj (8 objects per 640 image)
            b[:, 4] += math.log(8 / (640 / s) ** 2)
            b[:, 5:] += math.log(0.6 / (m.nc - 0.99)
                                 ) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)


def initialize_weights(*models):
    for model in models:
        real_init_weights(model)
def real_init_weights(m):

    if isinstance(m, list):
        for mini_m in m:
            real_init_weights(mini_m)
    else:
        if isinstance(m, torch.nn.Conv2d):    
            torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.Linear):
            m.weight.data.normal_(0.0, std=0.01)
        elif isinstance(m, torch.nn.BatchNorm2d):
            torch.nn.init.constant_(m.weight, 1)
            torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m,torch.nn.Module):
            for mini_m in m.children():
                real_init_weights(mini_m)
        else:
            print('unkonwn module', m)

def get_net(cfg):
    m_block_cfg = YOLOP
    aux_seg = cfg.LANE.AUX_SEG
    model = MCnet(pretrained=cfg.TRAIN.PRETRAIN, backbone=cfg.TRAIN.BACKBONE, \
        cls_dim=(cfg.LANE.GRIDING_NUM+1, 18, cfg.LANE.NUM_LANES), use_aux=cfg.LANE.AUX_SEG)
    return model


if __name__ == "__main__":
    from torch.utils.tensorboard import SummaryWriter
    model = get_net(False)
    input_ = torch.randn((1, 3, 256, 256))
    gt_ = torch.rand((1, 2, 256, 256))
    metric = SegmentationMetric(2)
    model_out, SAD_out = model(input_)
    detects, dring_area_seg, lane_line_seg = model_out
    Da_fmap, LL_fmap = SAD_out
    for det in detects:
        print(det.shape)
    print(dring_area_seg.shape)
    print(lane_line_seg.shape)
