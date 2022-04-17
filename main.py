import os
import argparse
import paddle
import pandas as pd
import random
import numpy as np
from models.dcl import DCLNet as MainModel
from utils.train_model import train
from utils.transforms import load_data_transformers
from utils.dataset import MyDataSet, collate_fn4train, collate_fn4test
import warnings

warnings.filterwarnings('ignore')


def set_random_seed(seed):
    paddle.seed(seed)
    np.random.seed(seed)
    random.seed(seed)


class LoadConfig:
    def __init__(self, args):
        self.describe = args.describe
        # 预训练权重
        self.pretrained_model = '/data/zhangzichao/models/resnet50.pdparams'
        # 数据集信息
        if args.dataset == 'CUB':
            self.dataset = args.dataset
            self.rawdata_root = '/data/zhangzichao/datasets/CUB/images'
            self.anno_root = './datasets/CUB'
            self.numcls = 200
        elif args.dataset == 'STCAR':
            self.dataset = args.dataset
            self.rawdata_root = '/data/zhangzichao/datasets/StanfordCars/car_ims/'
            self.anno_root = './datasets/STCAR'
            self.numcls = 196
        elif args.dataset == 'AIR':
            self.dataset = args.dataset
            self.rawdata_root = '/data/zhangzichao/datasets/fgvc-aircraft-2013b/data/images/'
            self.anno_root = './datasets/AIR'
            self.numcls = 100
        elif args.dataset == 'CUB_TINY':
            self.dataset = args.dataset
            self.rawdata_root = './datasets/CUB_TINY'
            self.anno_root = './datasets/CUB_TINY'
            self.numcls = 4
        else:
            raise Exception('dataset not defined')

        self.train_anno = pd.read_csv(os.path.join(self.anno_root, 'train.txt'), sep=",", header=None,
                                      names=['ImageName', 'label'])
        self.test_anno = pd.read_csv(os.path.join(self.anno_root, 'test.txt'), sep=",", header=None,
                                     names=['ImageName', 'label'])

        self.save_dir = f'./outputs/{args.dataset}/checkpoints'
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        self.log_folder = f'./outputs/{args.dataset}/logs'
        if not os.path.exists(self.log_folder):
            os.makedirs(self.log_folder)


def parse_args():
    parser = argparse.ArgumentParser(description='dcl parameters')
    parser.add_argument('--gpus', dest='gpus', default='0', type=str)
    parser.add_argument('--data', dest='dataset', default='CUB', type=str)
    parser.add_argument('--backbone', dest='backbone', default='resnet50', type=str)
    parser.add_argument('--epoch', dest='epoch', default=120, type=int)
    parser.add_argument('--T_max', dest='T_max', default=90, type=int)
    parser.add_argument('--ls', dest='label_smooth', default=0.1, type=float)
    parser.add_argument('--tb', dest='train_batch', default=8, type=int)
    parser.add_argument('--vb', dest='val_batch', default=8, type=int)
    parser.add_argument('--tnw', dest='train_num_workers', default=8, type=int)
    parser.add_argument('--vnw', dest='val_num_workers', default=8, type=int)
    parser.add_argument('--lr', dest='base_lr', default=0.001, type=float)
    parser.add_argument('--start_epoch', dest='start_epoch', default=0, type=int)
    parser.add_argument('--detail', dest='describe', default='dcl_cub', type=str)
    parser.add_argument('--size', dest='resize_resolution', default=512, type=int)
    parser.add_argument('--crop', dest='crop_resolution', default=448, type=int)
    parser.add_argument('--swap_num', dest='swap_num', default=7, type=int, help='specify a range')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    set_random_seed(seed=2022)
    paddle.device.set_device(f'gpu:{args.gpus}')
    print(args, flush=True)
    Config = LoadConfig(args)

    # 数据集加载
    transformers = load_data_transformers(args.resize_resolution, args.crop_resolution, (args.swap_num, args.swap_num))
    train_set = MyDataSet(Config=Config, anno=Config.train_anno, common_aug=transformers["common_aug"],
                          swap=transformers["swap"], totensor=transformers["train_totensor"], train=True)
    test_set = MyDataSet(Config=Config, anno=Config.test_anno, common_aug=transformers["None"],
                         swap=transformers["None"], totensor=transformers["test_totensor"], test=True)
    dataloader = {}
    dataloader['train'] = paddle.io.DataLoader(train_set, batch_size=args.train_batch, shuffle=True,
                                               num_workers=args.train_num_workers, collate_fn=collate_fn4train,
                                               drop_last=False)
    setattr(dataloader['train'], 'total_item_len', len(train_set))
    dataloader['test'] = paddle.io.DataLoader(test_set, batch_size=args.val_batch, shuffle=False,
                                              num_workers=args.val_num_workers, collate_fn=collate_fn4test,
                                              drop_last=False)
    setattr(dataloader['test'], 'total_item_len', len(test_set))
    setattr(dataloader['test'], 'num_cls', Config.numcls)

    # 模型加载、定义优化器
    model = MainModel(Config)
    scheduler = paddle.optimizer.lr.CosineAnnealingDecay(learning_rate=args.base_lr, T_max=args.T_max)
    optimizer = paddle.optimizer.Momentum(learning_rate=scheduler, parameters=model.parameters())

    # 开始训练
    train(Config,
          model,
          epoch_num=args.epoch,
          start_epoch=args.start_epoch,
          optimizer=optimizer,
          scheduler=scheduler,
          data_loader=dataloader)
