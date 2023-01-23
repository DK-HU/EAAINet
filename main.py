import argparse
from scripts.train import train
from scripts.test import test
import numpy as np
import torch
import random

def set_random_seed(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FDANET')

    parser.add_argument('--model', nargs='?', type=str, default='fdanet',
                        choices=('fdanet', 'multi_scale', 'backbone', 'multi_scale_sp', 'multi_scale_trans' ,
                                 'multi_scale_struct'),
                        help='Model to use [\'fdanet, reg-only\']')
    parser.add_argument('--ori_path', type=str)
    parser.add_argument('--dataset', nargs='?', type=str, default='7S',
                        choices=('7S', '7S_SP', '12S', 'my', 'cambridge'), help='Dataset to use')
    parser.add_argument('--scene', nargs='?', type=str, default='fire', help='Scene')
    parser.add_argument('--savemodel_path', nargs='?', type=str, default='fire', help='Scene')
    
    parser.add_argument('--flag', nargs='?', type=str, required=True,
                        choices=('train','test'), help='train or test')
    parser.add_argument('--eval', nargs='?', type=str, default=True, help='train or test')
    parser.add_argument('--init_lr', nargs='?', type=float, default=0.0002,
                        help='Initial learning rate')
    parser.add_argument('--batch_size', nargs='?', type=int, default=4,
                        help='Batch size')
    parser.add_argument('--aug', nargs='?', type=bool, default=True)
    parser.add_argument('--resume', nargs='?', type=str, default=None,
                        help='Path to saved model to resume from')  
    parser.add_argument('--data_path', required=True, type=str,
                        help='Path to dataset')
    parser.add_argument('--log_summary', default='progress_log_summary.txt',
                        metavar='PATH',
                        help='txt where to save per-epoch stats')
    parser.add_argument('--train_id', nargs='?', type=str, default='',
                        help='An identifier string')
    parser.add_argument('--n_epoch', nargs='?', type=int, default=5000,
                        help='the number of epoch to train')
    parser.add_argument('--output', nargs='?', type=str, default='./',
                        help='Output directory')        # 误差文件保存路径
    args = parser.parse_args()

    if args.dataset == '7S':
        if args.scene not in ['chess', 'heads', 'fire', 'office', 'pumpkin',
                              'redkitchen', 'stairs']:
            print('Selected scene is not valid.')
            exit()

    if args.dataset == '12S':
        if args.scene not in ['apt1/kitchen','apt1/living','apt2/bed','apt2/kitchen',
                              'apt2/living','apt2/luke','office1/gates362',
                              'office1/gates381','office1/lounge','office1/manolis',
                              'office2/5a','office2/5b']:
            print('Selected scene is not valid.')
            exit()

    if args.dataset == 'cambridge':
        if args.scene not in ['GreatCourt', 'KingsCollege', 'OldHospital',
                              'ShopFacade', 'StMarysChurch']:
            print('Selected scene is not valid.')
            exit()
    
    if args.dataset == 'my':
        if args.scene not in ['floor2','K201','K544']:
            print('Selected scene is not valid.')
            exit()
    
    if args.flag == 'train':
        train(args)
    elif args.flag == 'test':
        test(args, save_pose=True)
