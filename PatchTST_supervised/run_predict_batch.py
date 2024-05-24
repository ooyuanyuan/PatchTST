#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/10/13 11:51

import torch
import random
import argparse
import numpy as np
from exp.exp_main import Exp_Main
from run_longExp import get_setting

from deploy.json2args import get_args_from_json


def init_trained_experiment(args_json_file: str, predict_data_file_name: str = None):
    """
    初始化已训练的模型，并加载；
    :param args_json_file:
    :param predict_data_file_name: 若未指定，则使用默认test.txt;
    :return:
    """
    args = argparse.Namespace(**get_args_from_json(args_json_file))
    # 重新定义预测数据；默认为 test.txt;
    if predict_data_file_name:
        args.test_data_list = predict_data_file_name
    # random seed
    fix_seed = args.random_seed
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print(f'Args in experiment:{args}')

    Exp = Exp_Main  # Experiments Main;
    exp = Exp(args)

    setting = get_setting(args=args, itr=0)  # noqa
    print('模型预测结果路径: {}'.format(setting))
    return exp, setting


if __name__ == '__main__':
    exp, setting = init_trained_experiment(
        args_json_file='./scripts/product/args_bcs_patchtst_d3h6p1.json',  # noqa
        # predict_data_file="reply_example.txt"
    )
    print(exp.args.checkpoints)
    # 批量预测；
    exp.batch_online_predict(setting, load=True)
    torch.cuda.empty_cache()
