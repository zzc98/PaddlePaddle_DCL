import os
import argparse
from PIL import Image
from paddle.vision.transforms import transforms
import cv2
import paddle
from models.dcl import DCLNet as MainModel
from utils.eval_model import eval_turn
import pandas as pd
import warnings

warnings.filterwarnings('ignore')


class LoadConfig:
    def __init__(self, args):
        # 预训练权重
        self.pretrained_model = '/data/zhangzichao/models/resnet50.pdparams'
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
        self.test_anno = pd.read_csv(os.path.join(self.anno_root, 'test.txt'), sep=",", header=None,
                                     names=['ImageName', 'label'])
        self.swap_num = args.swap_num


def parse_args():
    parser = argparse.ArgumentParser(description='dcl parameters')

    parser.add_argument('--gpus', dest='gpus', default='0', type=str)
    parser.add_argument('--data', dest='dataset', default='CUB', required=True, type=str)
    parser.add_argument('--img', dest='img', default='resources/Black_Footed_Albatross_0001_796111.jpg',
                        required=True, type=str)
    parser.add_argument('--pdparams', dest='model_weight',
                        default='outputs/CUB/checkpoints/dcl_cub-20220416-183150.pdparams', type=str)
    parser.add_argument('--size', dest='resize_resolution', default=512, type=int)
    parser.add_argument('--crop', dest='crop_resolution', default=448, type=int)
    parser.add_argument('--swap_num', dest='swap_num', default=7, type=int, help='specify a range')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    paddle.device.set_device(f'gpu:{args.gpus}')
    Config = LoadConfig(args)
    # 数据集加载
    tfs = transforms.Compose([
        transforms.Resize(args.resize_resolution),
        transforms.CenterCrop(args.crop_resolution),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    test_img = Image.open(args.img).convert('RGB')
    cv_img = cv2.imread(args.img)
    cv2.imshow('pic', cv_img)
    cv2.waitKey()


    input_tensor = tfs(test_img).unsqueeze(0)
    model = MainModel(Config)
    model.load_dict(paddle.load(args.model_weight))
    model.eval()
    with paddle.no_grad():
        output = model(input_tensor)
    pre = paddle.argmax(output[0], axis=-1).item()
    print("prediction:", pre)
