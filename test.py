import os
import argparse
from utils.transforms import load_data_transformers
from utils.dataset import MyDataSet, collate_fn4test
import paddle
from models.dcl import DCLNet as MainModel
from utils.eval_model import eval_turn
import pandas as pd
import warnings

warnings.filterwarnings('ignore')


class LoadConfig:
    def __init__(self, args):
        # 预训练权重
        self.pretrained_model = '/data/models/resnet50.pdparams'
        if args.dataset == 'CUB':
            self.dataset = args.dataset
            self.rawdata_root = '/data/datasets/CUB/images'
            self.anno_root = './datasets/CUB'
            self.numcls = 200
        elif args.dataset == 'STCAR':
            self.dataset = args.dataset
            self.rawdata_root = '/data/datasets/StanfordCars/car_ims/'
            self.anno_root = './datasets/STCAR'
            self.numcls = 196
        elif args.dataset == 'AIR':
            self.dataset = args.dataset
            self.rawdata_root = '/data/datasets/fgvc-aircraft-2013b/data/images/'
            self.anno_root = './datasets/AIR'
            self.numcls = 100
        else:
            raise Exception('dataset not defined')
        self.test_anno = pd.read_csv(os.path.join(self.anno_root, 'test.txt'), sep=",", header=None,
                                     names=['ImageName', 'label'])
        self.swap_num = args.swap_num


def parse_args():
    parser = argparse.ArgumentParser(description='dcl parameters')
    parser.add_argument('--gpus', dest='gpus', default='0', type=str)
    parser.add_argument('--data', dest='dataset', default='CUB', type=str)
    parser.add_argument('--pdparams', dest='model_weight',
                        default='./checkpoints/training_descibe_41420_CUB/3.pdparams', type=str)
    parser.add_argument('--vb', dest='val_batch', default=8, type=int)
    parser.add_argument('--vnw', dest='val_num_workers', default=8, type=int)
    parser.add_argument('--size', dest='resize_resolution', default=512, type=int)
    parser.add_argument('--crop', dest='crop_resolution', default=448, type=int)
    parser.add_argument('--swap_num', dest='swap_num', default=7, type=int, help='specify a range')
    parser.add_argument('--not_used', dest='not_used', default='0', type=str)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    paddle.device.set_device(f'gpu:{args.gpus}')
    print(args, flush=True)
    Config = LoadConfig(args)
    # 数据集加载
    transformers = load_data_transformers(args.resize_resolution, args.crop_resolution, (args.swap_num, args.swap_num))
    test_set = MyDataSet(Config=Config, anno=Config.test_anno, common_aug=transformers["None"],
                         swap=transformers["None"], totensor=transformers["test_totensor"], test=True)
    dataloader = {'test': paddle.io.DataLoader(test_set, batch_size=args.val_batch, shuffle=False,
                                               num_workers=args.val_num_workers, collate_fn=collate_fn4test,
                                               drop_last=False)}
    setattr(dataloader['test'], 'total_item_len', len(test_set))
    setattr(dataloader['test'], 'num_cls', Config.numcls)

    # 模型加载评估
    model = MainModel(Config)
    model.load_dict(paddle.load(args.model_weight))
    accuracy = eval_turn(model, dataloader['test'])
    print('accuracy on {} is {:.2f}%'.format(Config.dataset, accuracy))
