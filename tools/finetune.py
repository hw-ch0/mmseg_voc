# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import logging
import os
import os.path as osp

from mmengine.config import Config, DictAction
from mmengine.logging import print_log
from mmengine.runner import Runner

from mmseg.registry import RUNNERS

import torch
import torch.optim as optim

def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentor')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume',
        action='store_true',
        default=False,
        help='resume from the latest checkpoint in the work_dir automatically')
    parser.add_argument(
        '--amp',
        action='store_true',
        default=False,
        help='enable automatic-mixed-precision training')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    args = parse_args()

    # load config
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    # enable automatic-mixed-precision training
    if args.amp is True: # default: False
        optim_wrapper = cfg.optim_wrapper.type
        if optim_wrapper == 'AmpOptimWrapper':
            print_log(
                'AMP training is already enabled in your config.',
                logger='current',
                level=logging.WARNING)
        else: # --> Default
            assert optim_wrapper == 'OptimWrapper', (
                '`--amp` is only supported when the optimizer wrapper type is '
                f'`OptimWrapper` but got {optim_wrapper}.')
            cfg.optim_wrapper.type = 'AmpOptimWrapper'
            cfg.optim_wrapper.loss_scale = 'dynamic'

    
    
    ############ load parameters from pretrain model  ############    
    # 1. Load pretrained model (Deeplap-v3 pretrained on COCO-Stuff 164k )
    pretrained_dlv3 = torch.load('checkpoint/deeplabv3_r50-d8_512x512_4x4_320k_coco-stuff164k_20210709_155403-51b21115.pth')
    
    # resume training
    cfg.resume = args.resume

    # build the runner from config
    if 'runner_type' not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)

    
     # 3. Get names of layers in the current model (MaskFormer)
    layer_names = []
    for idx, (layer, param) in enumerate(runner.model.named_parameters()):
        layer_names.append(layer[7:]) # except module.
        print(f"[CHECK A] [idx {idx}] {layer[7:]}")
    '''
    [3, 4, 6, 3] -> (30, 39, 57, 30) -> w/o downsample -> (27, 36, 54, 27)
    3~32   : backbone.layer1  
    33~71  : backbone.layer2  
    72~128 : backbone.layer3
    129~158: backbone.layer4
    159~302: decode_head
    '''
    
    cnt_2 = 0
    for idx, layer in enumerate(pretrained_dlv3['state_dict'].keys()):
        # print(f"[CHECK B] [idx {idx}] {layer}")
        if layer in layer_names:
            cnt_2 += 1
    print(f"[CHECK C] Overapped backbone layers: {cnt_2}") # 156
    
    
    
    original_lr = 0.0001
    finetune_ratio = 0.05
    parameters = []
    cnt_3 = 0
    for idx, layer in enumerate(layer_names):    
        if layer in pretrained_dlv3['state_dict'].keys() and idx<=128: # backbone
            runner.model._parameters[layer] = pretrained_dlv3['state_dict'][layer]
            # lr = original_lr * finetune_ratio
            lr = original_lr
            cnt_3 += 1
        else: # fc layer
            lr = original_lr
        parameters += [{'params': [p for l, p in runner.model.named_parameters() if l[7:] == layer and p.requires_grad],
                        'lr':     lr}]
    
    print(f"[CHECK D] cnt_3: {cnt_3}") # 126
    

    ############ set optimizer & optim_wrapper  ############ 
    optimizer = optim.AdamW(parameters, betas=(0.9, 0.999), weight_decay=0.0001)
    # cfg.optimizer = optimizer
    
    optim_wrapper = dict(
    # _delete_=True,
    type='OptimWrapper',
    optimizer=optimizer,
    clip_grad=dict(max_norm=0.01, norm_type=2),
    # paramwise_cfg=dict(custom_keys={
    #     'backbone': dict(lr_mult=0.1),
    # })
    )
    
    # runner.train때 runner.build_optim_wrapper가 선언되므로 괜찮음
    runner.optim_wrapper = optim_wrapper
    
    # start training
    runner.auto_scale_lr = {}
    runner.auto_scale_lr['enable'] = True
    runner.auto_scale_lr['base_batch_size'] = 8
    
    runner.train()


if __name__ == '__main__':
    main()
