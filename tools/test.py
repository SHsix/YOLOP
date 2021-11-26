import argparse
import os, sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import pprint
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import numpy as np
from lib.utils import DataLoaderX
from tensorboardX import SummaryWriter

import lib.dataset as dataset
from lib.config import cfg
from lib.config import update_config
from lib.core.loss import get_loss
from lib.core.function import validate
from lib.core.general import fitness
from lib.models import get_net
from lib.utils.utils import create_logger, select_device

from evaluation.eval_wrapper import eval_lane

def parse_args():
    parser = argparse.ArgumentParser(description='Test Multitask network')

    # philly
    parser.add_argument('--modelDir',
                        help='model directory',
                        type=str,
                        default='')
    parser.add_argument('--logDir',
                        help='log directory',
                        type=str,
                        default='runs/')
    # parser.add_argument('--weights', nargs='+', type=str, default='/home/YOLOP/runs/CULANE/_2021-11-16-05-39/epoch-8.pth', help='model.pth path(s)')
    parser.add_argument('--weights', nargs='+', type=str, default='/home/YOLOP/log/20211124_005915_lr_1e-03_b_80/ep051.pth', help='model.pth path(s)')
    parser.add_argument('--conf_thres', type=float, default=0.001, help='object confidence threshold')
    parser.add_argument('--iou_thres', type=float, default=0.6, help='IOU threshold for NMS')
    args = parser.parse_args()

    return args
#
# /home/YOLOP/runs/BddDataset/_2021-11-08-10-25/epoch-40.pth
# /home/YOLOP/runs/BddDataset/_2021-11-09-05-59/epoch-40.pth
#
def main():
    # set all the configurations
    args = parse_args()
    update_config(cfg, args)

    # TODO: handle distributed training logger
    # set the logger, tb_log_dir means tensorboard logdir

    logger, final_output_dir, tb_log_dir = create_logger(
        cfg, cfg.LOG_DIR, 'test')

    logger.info(pprint.pformat(args))
    logger.info(cfg)
    
    # cfg.LANE.AUX_SEG = False
    cls_num_per_lane = 18
    distributed = False
    if 'WORLD_SIZE' in os.environ:
        distributed = int(os.environ['WORLD_SIZE']) > 1

    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')



    writer_dict = {
        'writer': SummaryWriter(log_dir=tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    # bulid up model
    # start_time = time.time()
    print("begin to bulid up model...")
    # DP mode
    device = select_device(logger, batch_size=cfg.TEST.BATCH_SIZE_PER_GPU* len(cfg.GPUS)) if not cfg.DEBUG \
        else select_device(logger, 'cpu')
    # device = select_device(logger, 'cpu')

    model = get_net(cfg)
    print("finish build model")
    
    # define loss function (criterion) and optimizer
    criterion = get_loss(cfg, device=device)

    # load checkpoint model

    # det_idx_range = [str(i) for i in range(0,25)]
    model_dict = model.state_dict()
    checkpoint_file = args.weights
    logger.info("=> loading checkpoint '{}'".format(checkpoint_file))
    state_dict = torch.load(checkpoint_file, map_location = 'cpu')['model']
    compatible_state_dict = {}
    for k, v in state_dict.items():
        if 'module.' in k:
            compatible_state_dict[k[7:]] = v
        else:
            compatible_state_dict[k] = v

    model.load_state_dict(compatible_state_dict, strict = False)
    logger.info("=> loaded checkpoint '{}' ".format(checkpoint_file))

    model = model.to(device)
    model.gr = 1.0
    model.nc = 1
    print('bulid model finished')

    print("begin to load data")
    # Data loading
    # normalize = transforms.Normalize(
    #     mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    # )

    # valid_dataset = eval('dataset.' + cfg.DATASET.DATASET)(
    #     cfg=cfg,
    #     is_train=False,
    #     inputsize=cfg.MODEL.IMAGE_SIZE,
    #     transform=transforms.Compose([
    #         transforms.ToTensor(),
    #         normalize,
    #     ])
    # )

    # valid_loader = DataLoaderX(
    #     valid_dataset,
    #     batch_size=cfg.TEST.BATCH_SIZE_PER_GPU * len(cfg.GPUS),
    #     shuffle=False,
    #     num_workers=cfg.WORKERS,
    #     pin_memory=False,
    #     collate_fn=dataset.AutoDriveDataset.collate_fn
    # )
    # print('load data finished')

    # epoch = 0 #special for test
    # detect_results, total_loss, maps, times = validate(
    #     epoch,cfg, valid_loader, valid_dataset, model, criterion,
    #     final_output_dir, tb_log_dir, writer_dict,
    #     logger, device
    # )
    # fi = fitness(np.array(detect_results).reshape(1, -1))
    # msg =   'Test:    Loss({loss:.3f})\n' \
    #                   'Detect: P({p:.3f})  R({r:.3f})  mAP@0.5({map50:.3f})  mAP@0.5:0.95({map:.3f})\n'\
    #                   'Time: inference({t_inf:.4f}s/frame)  nms({t_nms:.4f}s/frame)'.format(
    #                       loss=total_loss, 
    #                       p=detect_results[0],r=detect_results[1],map50=detect_results[2],map=detect_results[3],
    #                       t_inf=times[0], t_nms=times[1])
    # logger.info(msg)



    if not os.path.exists(cfg.LANE.TEST_DIR):
        os.mkdir(cfg.LANE.TEST_DIR)

    eval_lane(cfg, model, cfg.LANE.DATASET, cfg.DATASET.DATAROOT, cfg.LANE.TEST_DIR, cfg.LANE.GRIDING_NUM, True, distributed)
    print("test finish")


if __name__ == '__main__':
    main()
    