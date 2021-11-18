import argparse
import os, sys
import shutil
import time
from pathlib import Path
import imageio

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

print(sys.path)
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import scipy.special
import numpy as np
import torchvision.transforms as transforms
import PIL.Image as image

from lib.config import cfg
from lib.config import update_config
from lib.utils.utils import create_logger, select_device, time_synchronized
from lib.models import get_net
# from lib.dataset import LoadImages, LoadStreams
from lib.core.general import non_max_suppression, scale_coords
from lib.utils import plot_one_box,show_seg_result
from lib.core.function import AverageMeter
from lib.core.postprocess import morphological_process, connect_lane
from tqdm import tqdm


from data.dataset import LaneTestDataset
from data.constant import culane_row_anchor



def detect(cfg,opt):

    logger, _, _ = create_logger(
        cfg, cfg.LOG_DIR, 'demo')

    # device = select_device(logger,opt.device)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    if os.path.exists(opt.save_dir):  # output dir
        shutil.rmtree(opt.save_dir)  # delete dir
    os.makedirs(opt.save_dir)  # make new dir
    half = device.type != 'cpu'  # half precision only supported on CUDA

    print('Using device : ', device)
    # Load model
    model = get_net(cfg)
    # checkpoint = torch.load(opt.weights, map_location= device)
    # model.load_state_dict(checkpoint['state_dict'])
    model_dict = model.state_dict()
    checkpoint_file = opt.weights
    logger.info("=> loading checkpoint '{}'".format(checkpoint_file))
    checkpoint = torch.load(checkpoint_file)
    checkpoint_dict = checkpoint['state_dict']
    # checkpoint_dict = {k: v for k, v in checkpoint['state_dict'].items() if k.split(".")[1] in det_idx_range}
    model_dict.update(checkpoint_dict)
    # model_dict.update(checkpoint)
    

    # normalize = transforms.Normalize(
    #         mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    #     )

    # transform=transforms.Compose([
    #             transforms.ToTensor(),
    #             normalize,
    #         ])

    # model.load_state_dict(model_dict)
    # model = model.to(device)
    # if half:
    #     model.half()  # to FP16

    # # Set Dataloader
    # if opt.source.isnumeric():
    #     cudnn.benchmark = True  # set True to speed up constant image size inference
    #     dataset = LoadStreams(opt.source, img_size=opt.img_size)
    #     bs = len(dataset)  # batch_size
    # else:
    #     dataset = LoadImages(opt.source, img_size=opt.img_size)
    #     bs = 1  # batch_size



    # # # Get names and colors
    # names = model.module.names if hasattr(model, 'module') else model.names
    # colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]


    # # Run inference
    # t0 = time.time()

    # vid_path, vid_writer = None, None
    # img = torch.zeros((1, 3, opt.img_size, opt.img_size), device=device)  # init img
    # _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    # model.eval()

    # inf_time = AverageMeter()
    # nms_time = AverageMeter()
    
    # for i, (path, img, img_det, vid_cap,shapes) in tqdm(enumerate(dataset),total = len(dataset)):
    #     img = transform(img).to(device)
    #     img = img.half() if half else img.float()  # uint8 to fp16/32
    #     if img.ndimension() == 3:
    #         img = img.unsqueeze(0)
    #     # Inference
    #     t1 = time_synchronized()
    #     det_out = model(img)[0]
    #     t2 = time_synchronized()
    #     # if i == 0:
    #     #     print(det_out)
    #     inf_out, _ = det_out
    #     inf_time.update(t2-t1,img.size(0))

    #     # Apply NMS
    #     t3 = time_synchronized()
    #     det_pred = non_max_suppression(inf_out, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False)
    #     t4 = time_synchronized()

    #     nms_time.update(t4-t3,img.size(0))
    #     det=det_pred[0]

    #     save_path = str(opt.save_dir +'/'+ Path(path).name) if dataset.mode != 'stream' else str(opt.save_dir + '/' + "web.mp4")
    #     print(img_det.shape)
    #     print(type(img_det))

    #     print('img', img.shape)
    #     print(len(det))

    #     # img -> torch[1, 3, 384, 640]
    #     # img_det -> original img size
    #     if len(det):
    #         print(det[:, :4])
    #         det[:,:4] = scale_coords(img.shape[2:],det[:,:4],img_det.shape).round()
    #         print(det[:, :4])
    #         for *xyxy,conf,cls in reversed(det):
    #             label_det_pred = f'{names[int(cls)]} {conf:.2f}'
    #             plot_one_box(xyxy, img_det , label=label_det_pred, color=colors[int(cls)], line_thickness=2)
        
    #     if dataset.mode == 'images':
    #         cv2.imwrite(save_path,img_det)

    #     elif dataset.mode == 'video':
    #         if vid_path != save_path:  # new video
    #             vid_path = save_path
    #             if isinstance(vid_writer, cv2.VideoWriter):
    #                 vid_writer.release()  # release previous video writer

    #             fourcc = 'mp4v'  # output video codec
    #             fps = vid_cap.get(cv2.CAP_PROP_FPS)
    #             h,w,_=img_det.shape
    #             vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
    #         vid_writer.write(img_det)
        
    #     else:
    #         cv2.imshow('image', img_det)
    #         cv2.waitKey(1)  # 1 millisecond

    # print('Results saved to %s' % Path(opt.save_dir))
    # print('Done. (%.3fs)' % (time.time() - t0))
    # print('inf : (%.4fs/frame)   nms : (%.4fs/frame)' % (inf_time.avg,nms_time.avg))

    inf_time = AverageMeter()
    nms_time = AverageMeter()
    model.load_state_dict(model_dict)
    model = model.to(device)
    
    model.eval()


    # # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    cv_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,    
    ])

    if cfg.LANE.DATASET == 'CULane':
        splits = ['test0_normal.txt', 'test1_crowd.txt', 'test2_hlight.txt', 'test3_shadow.txt', 'test4_noline.txt', 'test5_arrow.txt', 'test6_curve.txt', 'test7_cross.txt', 'test8_night.txt']
        datasets = [LaneTestDataset(cfg.DATASET.DATAROOT,os.path.join(cfg.DATASET.DATAROOT, 'list/test_split/'+split),img_transform = cv_transform) for split in splits]
        img_w, img_h = 1640, 590
        row_anchor = culane_row_anchor
        cls_num_per_lane = len(row_anchor)
    else:
        raise NotImplementedError

    for split, dataset in zip(splits, datasets):
        loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle = False, num_workers=16)
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        print(split[:-3]+'avi')
        vout = cv2.VideoWriter(split[:-3]+'avi', fourcc , 30.0, (img_w, img_h))
        for i, data in enumerate(tqdm(loader)):
            imgs, names, real_imgs, shapes = data
            # print('test', type(ob_img))
            # ob_img = ob_transform(ob_img).to(device)
            # print(type(ob_img))

            imgs = imgs.cuda()
            t1 = time_synchronized()
            with torch.no_grad():
                det_out, lane_out = model(imgs)
                cls_out, _ = lane_out
            t2 = time_synchronized()
            col_sample = np.linspace(0, 256 - 1,  cfg.LANE.GRIDING_NUM)
            col_sample_w = col_sample[1] - col_sample[0]


            out_j = cls_out[0].data.cpu().numpy()
            out_j = out_j[:, ::-1, :]
            prob = scipy.special.softmax(out_j[:-1, :, :], axis=0)
            idx = np.arange( cfg.LANE.GRIDING_NUM) + 1
            idx = idx.reshape(-1, 1, 1)
            loc = np.sum(prob * idx, axis=0)
            out_j = np.argmax(out_j, axis=0)
            loc[out_j ==  cfg.LANE.GRIDING_NUM] = 0
            out_j = loc


        #     # Inference
        
    

            # if i == 0:
            #     print(det_out)
            inf_out, _ = det_out
            inf_time.update(t2-t1,imgs.size(0))


            t2 = time_synchronized()
            # if i == 0:
            #     print(det_out)
            inf_out, _ = det_out
            inf_time.update(t2-t1,imgs.size(0))

            # Apply NMS
            t3 = time_synchronized()
            det_pred = non_max_suppression(inf_out, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False)
            t4 = time_synchronized()

            nms_time.update(t4-t3,imgs.size(0))
            det=det_pred[0]

        # self.sources, img, img0[0], None, shapes
        # for i, (path, img, img_det, vid_cap,shapes) 

            # save_path = str(opt.save_dir +'/'+ Path('').name) if dataset.mode != 'stream' else str(opt.save_dir + '/' + "web.mp4")


            vis = cv2.imread(os.path.join(cfg.DATASET.DATAROOT, names[0]))
            if len(det):
                det[:,:4] = scale_coords(imgs.shape[2:], det[:,:4], vis.shape).round()
                for *xyxy,conf,cls in reversed(det):
                    label_det_pred = f'{names[int(cls)]} {conf:.2f}'
                    plot_one_box(xyxy, vis , label=label_det_pred, color=colors[int(cls)], line_thickness=2)
            
        
            cv2.imwrite('/home/YOLOP/inference/culane/a.jpg', vis)

            # elif dataset.mode == 'video':
            #     if vid_path != save_path:  # new video
            #         vid_path = save_path
            #         if isinstance(vid_writer, cv2.VideoWriter):
            #             vid_writer.release()  # release previous video writer

            #         fourcc = 'mp4v'  # output video codec
            #         fps = vid_cap.get(cv2.CAP_PROP_FPS)
            #         h,w,_=img_det.shape
            #         vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
            #     vid_writer.write(img_det)
            
            # else:
            #     cv2.imshow('image', img_det)
            #     cv2.waitKey(1)  # 1 millisecond
            # vout.release()


                    # import pdb; pdb.set_trace()
            for i in range(out_j.shape[1]):
                if np.sum(out_j[:, i] != 0) > 2:
                    for k in range(out_j.shape[0]):
                        if out_j[k, i] > 0:
                            ppp = (int(out_j[k, i] * col_sample_w * img_w / 256) - 1, int(img_h * (row_anchor[cls_num_per_lane-1-k]/256)) - 1 )
                            cv2.circle(vis,ppp,5,(0,255,0),-1)
            vout.write(vis)

# /home/YOLOP/runs/BddDataset/_2021-11-08-10-25/epoch-40.pth

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='/home/YOLOP/runs/CULANE/_2021-11-17-02-35/epoch-40.pth', help='model.pth path(s)')
    # parser.add_argument('--weights', nargs='+', type=str, default='/home/YOLOP/runs/CULANE/_2021-11-16-05-39/epoch-8.pth', help='model.pth path(s)')
    # parser.add_argument('--weights', nargs='+', type=str, default='/home/YOLOP/runs/BddDataset/_2021-11-08-10-25/epoch-39.pth', help='model.pth path(s)')
    parser.add_argument('--source', type=str, default='inference/videos', help='source')  # file/folder   ex:inference/images
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--save-dir', type=str, default='inference/output', help='directory to save results')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    opt = parser.parse_args()
    with torch.no_grad():
        detect(cfg,opt)
