import torch, os, sys, datetime
import numpy as np
import argparse
import math

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from utils.dist_utils import dist_print, dist_tqdm
from utils.factory import get_metric_dict, get_loss_dict, get_scheduler#, get_optimizer
from utils.metrics import update_metrics, reset_metrics
from lib.utils.utils import get_optimizer
from utils.common import merge_config, save_model, cp_projects
from utils.common import get_work_dir, get_logger

from lib.config import cfg
from lib.config import update_config
from lib.models import get_net

from data.dataloader import get_train_loader
import time

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

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

    parser.add_argument('--auto_backup', action='store_true', help='automatically backup current code in the log path')

    args = parser.parse_args()

    return args
def inference(net, img, cls_label, seg_label, target, use_aux):
    if use_aux:
        target, cls_label, seg_label = target.cuda(), cls_label.long().cuda(), seg_label.long().cuda()
        det_out, lane_out = net(img)
        cls_out, seg_out = lane_out

        return {'cls_out': cls_out, 'cls_label': cls_label, \
                'seg_out':seg_out, 'seg_label': seg_label, \
                'det_out': det_out, 'target':target, 'model' : net
                }
    else:
        det_out, lane_out = net(img)
        cls_out, seg_out = lane_out

        return {'cls_out': cls_out, 'cls_label': cls_label, \
                'det_out': det_out, 'target':target, 'model' : net
                }


def resolve_val_data(results, use_aux):
    results['cls_out'] = torch.argmax(results['cls_out'], dim=1)
    if use_aux:
        results['seg_out'] = torch.argmax(results['seg_out'], dim=1)
    return results


def calc_loss(loss_dict, results, logger, global_step):
    loss = 0

    for i in range(len(loss_dict['name'])):

        data_src = loss_dict['data_src'][i]

        datas = [results[src] for src in data_src]

        loss_cur = loss_dict['op'][i](*datas)

        if global_step % 20 == 0:
            logger.add_scalar('loss/'+loss_dict['name'][i], loss_cur, global_step)

        loss += loss_cur * loss_dict['weight'][i]
    return loss


def train(net, data_loader, loss_dict, optimizer, scheduler,logger, epoch, metric_dict, use_aux, device):
    net.train()
    progress_bar = dist_tqdm(train_loader)
    t_data_0 = time.time()
    for b_idx, (img, labels) in enumerate(progress_bar):

        target, cls_label, seg_label = labels
        img = img.to(device, non_blocking=True)

        t_data_1 = time.time()
        reset_metrics(metric_dict)
        global_step = epoch * len(data_loader) + b_idx

        t_net_0 = time.time()
        results = inference(net, img, cls_label, seg_label, target, use_aux)

        loss = calc_loss(loss_dict, results, logger, global_step)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step(global_step)
        t_net_1 = time.time()

        results = resolve_val_data(results, use_aux)

        update_metrics(metric_dict, results)
        if global_step % 20 == 0:
            for me_name, me_op in zip(metric_dict['name'], metric_dict['op']):
                logger.add_scalar('metric/' + me_name, me_op.get(), global_step=global_step)
        logger.add_scalar('meta/lr', optimizer.param_groups[0]['lr'], global_step=global_step)

        if hasattr(progress_bar,'set_postfix'):
            kwargs = {me_name: '%.3f' % me_op.get() for me_name, me_op in zip(metric_dict['name'], metric_dict['op'])}
            progress_bar.set_postfix(loss = '%.3f' % float(loss), 
                                    data_time = '%.3f' % float(t_data_1 - t_data_0), 
                                    net_time = '%.3f' % float(t_net_1 - t_net_0), 
                                    **kwargs)
        t_data_0 = time.time()
        


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True

    args = parse_args()
    update_config(cfg, args)

    work_dir = get_work_dir(cfg)

    distributed = False
    if 'WORLD_SIZE' in os.environ:
        distributed = int(os.environ['WORLD_SIZE']) > 1

    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    dist_print(datetime.datetime.now().strftime('[%Y/%m/%d %H:%M:%S]') + ' start training...')
    dist_print(cfg)
   


    train_loader, cls_num_per_lane = get_train_loader(cfg, cfg.TRAIN.BATCH_SIZE, \
        cfg.LANE.GRIDING_NUM, cfg.LANE.DATASET, cfg.LANE.AUX_SEG, distributed, cfg.LANE.NUM_LANES)
    
    device = 'cuda:0'
    net = get_net(cfg).to(device)

    # assign model params
    net.gr = 1.0
    net.nc = 1
    # net = parsingNet(pretrained = True, backbone=cfg.backbone,cls_dim = (cfg.griding_num+1,cls_num_per_lane, cfg.num_lanes),use_aux=cfg.use_aux).cuda()
    

    if distributed:
        net = torch.nn.parallel.DistributedDataParallel(net, device_ids = [args.local_rank])
    
    # optimizer = get_optimizer(net, cfg)
    optimizer = get_optimizer(cfg, net)
    

    if cfg.TRAIN.FINETUNE is not None:
        dist_print('finetune from ', cfg.TRAIN.FINETUNE)
        state_all = torch.load(cfg.TRAIN.FINETUNE)['model']
        state_clip = {}  # only use backbone parameters
        for k,v in state_all.items():
            if 'model' in k:
                state_clip[k] = v
        net.load_state_dict(state_clip, strict=False)

    if cfg.TRAIN.RESUME is not None:
        dist_print('==> Resume model from ' + cfg.TRAIN.RESUME)
        resume_dict = torch.load(cfg.TRAIN.RESUME, map_location='cpu')
        net.load_state_dict(resume_dict['model'])
        if 'optimizer' in resume_dict.keys():
            optimizer.load_state_dict(resume_dict['optimizer'])
        resume_epoch = int(os.path.split(cfg.resume)[1][2:5]) + 1
    else:
        resume_epoch = 0

    lf = lambda x: ((1 + math.cos(x * math.pi / cfg.TRAIN.END_EPOCH)) / 2) * \
                   (1 - cfg.TRAIN.LRF) + cfg.TRAIN.LRF  # cosine
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    # scheduler = get_scheduler(optimizer, cfg, len(train_loader))
    dist_print(len(train_loader))
    metric_dict = get_metric_dict(cfg)
    loss_dict = get_loss_dict(cfg, device)
    logger = get_logger(work_dir, cfg)
    cp_projects(args.auto_backup, work_dir)
    
    for epoch in range(resume_epoch, cfg.TRAIN.END_EPOCH+1):

        # train(net, train_loader, loss_dict, optimizer, scheduler,logger, epoch, metric_dict, cfg.use_aux)
        train(net, train_loader, loss_dict, optimizer, scheduler, \
            logger, epoch, metric_dict, cfg.LANE.AUX_SEG, device)
        save_model(net, optimizer, epoch ,work_dir, distributed)
    logger.close()