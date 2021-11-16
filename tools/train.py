import argparse
import os, sys
import math
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import pprint
import time
import torch
import torch.nn.parallel
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda import amp
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import numpy as np

from tensorboardX import SummaryWriter


from lib.config import cfg
from lib.config import update_config
from lib.core.loss import get_loss
from lib.core.function import train
from lib.core.function import validate
from lib.core.general import fitness
from lib.models import get_net
from lib.utils import is_parallel
from lib.utils.utils import get_optimizer
from lib.utils.utils import save_checkpoint
from lib.utils.utils import create_logger, select_device
from lib.utils import run_anchor


def parse_args():
    parser = argparse.ArgumentParser(description='Train Multitask network')
    # general
    # parser.add_argument('--cfg',
    #                     help='experiment configure file name',
    #                     required=True,
    #                     type=str)

    # philly
    parser.add_argument('--modelDir',
                        help='model directory',
                        type=str,
                        default='')
    parser.add_argument('--logDir',
                        help='log directory',
                        type=str,
                        default='runs/')
    parser.add_argument('--dataDir',
                        help='data directory',
                        type=str,
                        default='')
    parser.add_argument('--prevModelDir',
                        help='prev Model directory',
                        type=str,
                        default='')

    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='IOU threshold for NMS')
    args = parser.parse_args()

    return args


def main():
    # set all the configurations
    args = parse_args()
    update_config(cfg, args)

    # Set DDP variables
    world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    global_rank = int(os.environ['RANK']) if 'RANK' in os.environ else -1

    rank = global_rank
    #print(rank)
    # TODO: handle distributed training logger
    # set the logger, tb_log_dir means tensorboard logdir

    logger, final_output_dir, tb_log_dir = create_logger(
        cfg, cfg.LOG_DIR, 'train', rank=rank)

    if rank in [-1, 0]:
        logger.info(pprint.pformat(args))
        logger.info(cfg)

        writer_dict = {
            'writer': SummaryWriter(log_dir=tb_log_dir),
            'train_global_steps': 0,
            'valid_global_steps': 0,
        }
    else:
        writer_dict = None

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    # bulid up model
    # start_time = time.time()
    print("begin to bulid up model...")
    # DP mode
    device = select_device(logger, batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU* len(cfg.GPUS)) if not cfg.DEBUG \
        else select_device(logger, 'cpu')


    if args.local_rank != -1:
        assert torch.cuda.device_count() > args.local_rank
        torch.cuda.set_device(args.local_rank)
        device = torch.device('cuda', args.local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')  # distributed backend
    
    print("load model to device")
    print(device)
    model = get_net(cfg).to(device)

    # print("load finished")
    #model = model.to(device)
    # print("finish build model")
    

    # define loss function (criterion) and optimizer
    criterion = get_loss(cfg, device=device)
    optimizer = get_optimizer(cfg, model)


    # load checkpoint model
    best_perf = 0.0
    best_model = False
    last_epoch = -1

    Encoder_para_idx = [str(i) for i in range(0, 17)]
    Det_Head_para_idx = [str(i) for i in range(17, 25)]
    lane_Head_para_idx = [str(i) for i in range(25, 35)]


    lf = lambda x: ((1 + math.cos(x * math.pi / cfg.TRAIN.END_EPOCH)) / 2) * \
                   (1 - cfg.TRAIN.LRF) + cfg.TRAIN.LRF  # cosine
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    begin_epoch = cfg.TRAIN.BEGIN_EPOCH

    if rank in [-1, 0]:
        checkpoint_file = os.path.join(
            os.path.join(cfg.LOG_DIR, cfg.DATASET.DATASET), 'checkpoint.pth'
        )
        
        # Section for getting any pretrained model
        
        if os.path.exists(cfg.MODEL.PRETRAINED):
            logger.info("=> loading model '{}'".format(cfg.MODEL.PRETRAINED))
            checkpoint = torch.load(cfg.MODEL.PRETRAINED)
            begin_epoch = checkpoint['epoch']
            # best_perf = checkpoint['perf']
            last_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info("=> loaded checkpoint '{}' (epoch {})".format(
                cfg.MODEL.PRETRAINED, checkpoint['epoch']))
            #cfg.NEED_AUTOANCHOR = False     #disable autoanchor
        
        if os.path.exists(cfg.MODEL.PRETRAINED_DET):
            logger.info("=> loading model weight in det branch from '{}'".format(cfg.MODEL.PRETRAINED))
            det_idx_range = [str(i) for i in range(0,25)]
            model_dict = model.state_dict()
            checkpoint_file = cfg.MODEL.PRETRAINED_DET
            checkpoint = torch.load(checkpoint_file)
            begin_epoch = checkpoint['epoch']
            last_epoch = checkpoint['epoch']
            checkpoint_dict = {k: v for k, v in checkpoint['state_dict'].items() if k.split(".")[1] in det_idx_range}
            model_dict.update(checkpoint_dict)
            model.load_state_dict(model_dict)
            logger.info("=> loaded det branch checkpoint '{}' ".format(checkpoint_file))
        
        
        # Resume the training
        if cfg.AUTO_RESUME and os.path.exists(checkpoint_file):
            logger.info("=> loading checkpoint '{}'".format(checkpoint_file))
            checkpoint = torch.load(checkpoint_file)
            begin_epoch = checkpoint['epoch']
            # best_perf = checkpoint['perf']
            last_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            # optimizer = get_optimizer(cfg, model)
            optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info("=> loaded checkpoint '{}' (epoch {})".format(
                checkpoint_file, checkpoint['epoch']))
            #cfg.NEED_AUTOANCHOR = False     #disable autoanchor
        # model = model.to(device)

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        # model = torch.nn.DataParallel(model, device_ids=cfg.GPUS)
        
        # model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).cuda()
    # # DDP mode
    if rank != -1:
        model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank,find_unused_parameters=True)


    # assign model params
    model.gr = 1.0
    model.nc = 1
    # print('bulid model finished')

    
    # lane parts
    from data.dataloader import get_train_loader
        
    from utils.dist_utils import dist_print, dist_tqdm, is_main_process, DistSummaryWriter
    from utils.factory import get_metric_dict, get_loss_dict
    from utils.metrics import MultiLabelAcc, AccTopk, Metric_mIoU, update_metrics, reset_metrics
    
    distributed  = world_size > 1
    print("begin to load lane data")              
    lane_train_loader, cls_num_per_lane = get_train_loader(cfg, cfg.LANE.BATCH_SIZE, \
        cfg.LANE.GRIDING_NUM, cfg.LANE.DATASET, cfg.LANE.AUX_SEG, distributed, cfg.LANE.NUM_LANES)
    
    lane_optimizer = get_optimizer(cfg, model)
    lane_metric_dict = get_metric_dict(cfg)
    lane_loss_dict = get_loss_dict(cfg, device)
    
    print('=> start training...')
    
    for epoch in range(begin_epoch+1, cfg.TRAIN.END_EPOCH+1):
        
        # logger.info('freeze object detection head...')
        # for k, v in model.named_parameters():
        #     v.requires_grad = True  # train all layers
        #     if k.split(".")[1] in Det_Head_para_idx:
        #         # print('freezing %s' % k)
        #         v.requires_grad = False

        
        train(model, lane_train_loader, lane_loss_dict, lane_optimizer, lr_scheduler,logger, epoch, lane_metric_dict, cfg.LANE.AUX_SEG, device)
            # if perf_indicator >= best_perf:
            #     best_perf = perf_indicator
            #     best_model = True
            # else:
            #     best_model = False

        # save checkpoint model and best model
        if rank in [-1, 0]:
            savepath = os.path.join(final_output_dir, f'epoch-{epoch}.pth')
            logger.info('=> saving checkpoint to {}'.format(savepath))
            save_checkpoint(
                epoch=epoch,
                name=cfg.MODEL.NAME,
                model=model,
                # 'best_state_dict': model.module.state_dict(),
                # 'perf': perf_indicator,
                optimizer=optimizer,
                output_dir=final_output_dir,
                filename=f'epoch-{epoch}.pth'
            )
            save_checkpoint(
                epoch=epoch,
                name=cfg.MODEL.NAME,
                model=model,
                # 'best_state_dict': model.module.state_dict(),
                # 'perf': perf_indicator,
                optimizer=optimizer,
                output_dir=os.path.join(cfg.LOG_DIR, cfg.DATASET.DATASET),
                filename='checkpoint.pth'
            )
            

    # # save final model
    # if rank in [-1, 0]:
    #     final_model_state_file = os.path.join(
    #         final_output_dir, 'final_state.pth'
    #     )
    #     logger.info('=> saving final model state to {}'.format(
    #         final_model_state_file)
    #     )
    #     model_state = model.module.state_dict() if is_parallel(model) else model.state_dict()
    #     torch.save(model_state, final_model_state_file)
    #     writer_dict['writer'].close()
    # else:
    #     dist.destroy_process_group()



if __name__ == '__main__':
    main()