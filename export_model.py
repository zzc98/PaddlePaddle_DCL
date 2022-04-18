import os
import argparse

import paddle
from test import LoadConfig
from models.dcl import DCLNet as MainModel


def parse_args():
    parser = argparse.ArgumentParser(description='Model export.')
    parser.add_argument('--data', dest='dataset', default='CUB', type=str)
    parser.add_argument('--save_dir', dest='save_dir', help='The directory for saving the exported model', type=str,
                        default='outputs/STATIC')
    parser.add_argument('--model_path', dest='model_path', help='The path of model for export', type=str,
                        default='outputs/CUB/checkpoints/dcl_cub-20220416-183150.pdparams')
    parser.add_argument('--swap_num', dest='swap_num', default=7, type=int, help='specify a range')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    Config = LoadConfig(args)
    model = MainModel(Config)
    print(args.model_path)
    if os.path.exists(args.model_path):
        print('Loaded trained params of model successfully.')
        model.set_state_dict(paddle.load(args.model_path))
    else:
        print('Weight file dose not exist.')
    model.eval()

    input_spec = paddle.static.InputSpec(shape=[-1, 3, 448, 448], dtype='float32', name='image')
    model = paddle.jit.to_static(model, input_spec=[input_spec])
    save_path = os.path.join(args.save_dir, 'model')
    paddle.jit.save(model, save_path)
    print(f'Model is saved in {args.save_dir}.')
