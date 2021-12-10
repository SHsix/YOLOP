import argparse
import os, sys
import math
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)



import torch
import torch.nn.parallel
from torch.nn.parallel import DistributedDataParallel as DDP

import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed

from ptflops import get_model_complexity_info
import pytorch_model_summary
from torchsummary import summary
from tensorboardX import SummaryWriter


from lib.config import cfg
from lib.config import update_config
from lib.models import get_net

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


    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    # bulid up model
    # start_time = time.time()
    print("begin to bulid up model...")
    # DP mode
    device = 'cuda'


    if args.local_rank != -1:
        assert torch.cuda.device_count() > args.local_rank
        torch.cuda.set_device(args.local_rank)
        device = torch.device('cuda', args.local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')  # distributed backend
    
    print("load model to device")

    model = get_net(cfg)
    print(model)
    # dummy_size = (3, 256, 640)
    # macs, params = get_model_complexity_info(model, dummy_size, as_strings=True, print_per_layer_stat=True, verbose=True)
                                
    # print('computational complexity: ', macs)
    # print('number of parameters: ', params)

    print(pytorch_model_summary.summary(model, torch.zeros(1, 3, 256, 640)))
    # summary(model, torch.zeros(3, 256, 640).cuda())
        

if __name__ == '__main__':
    main()