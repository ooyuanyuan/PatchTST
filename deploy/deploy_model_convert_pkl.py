#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/10/31 16:14

import os
import torch
import argparse

try:
    import cPickle as pickle  # noqa
except:
    import pickle

from json2args import get_args_from_json
from deploy import projectPath


# ---------- 系统其他接口；--------------------------------------------------------------
# def dumps_model(serialization_model, **kwargs):
#     """序列化模型.
#
#     :param serialization_model: 训练返回的模型
#     :param kwargs:
#     :return: 序列化文件
#     """
#     # 平台为Scikit-learn模型，提供如下序列化方法(pickle)，您也可以import或者自定义其他序列化方法
#     serialization_model = "/data/project/patchtst/PatchTST_supervised/checkpoints/AIOps_36_0_6_PatchTST_AIOps_ftS_sl36_ll0_pl6_dm16_nh4_el3_dl1_df128_fc1_ebtimeF_dtFalse_Exp_0/checkpoint.pth")
#     with open(serialization_model, 'wb') as f:
#         pickle.dump(banana, filehandler)


def convert_gpu2cpu_model_state(args_json_file):
    """
    平台目前只支持：加载的模型是CPU模型，GPU
    :return:
    """

    from run_predict_batch import init_trained_experiment
    exp, setting = init_trained_experiment(args_json_file=args_json_file)
    path = os.path.join(projectPath, exp.args.checkpoints, setting)
    best_model_path = f'{path}/checkpoint.pth'
    print(f"获取训练模型: {best_model_path}")

    cpu_out_model = f'{path}/checkpoint.cpu.pth'
    # 加载模型参数，并转化成CPU级别；
    if os.path.exists(cpu_out_model):
        print(f"请注意当前文件已存在，生成将覆盖!!! {cpu_out_model}")

    exp.model.load_state_dict(torch.load(best_model_path))
    exp.model.to('cpu')
    torch.save(exp.model.state_dict(), cpu_out_model)
    # 写文件
    # torch.save(exp.model.state_dict(), 'model.pkl')
    # with open(f"./{path}/model_cpu.pkl", 'wb') as f:
    #     pickle.dump(exp.model.to('cpu'), f)
    return 1


def convert_pth2pkl(args_json_file='args_aiops.json', cuda='cpu'):
    """
    :param args_json_file:
    :param cuda: 将模型解析成CPU类型的模型；
    :return:
    """
    # convert gpu to cpu for model;
    if cuda == 'cpu':
        convert_gpu2cpu_model_state(args_json_file=args_json_file)

    # convert to pickle;
    from run_longExp import get_setting
    args = argparse.Namespace(**get_args_from_json(args_json_file))
    print(f'Args in experiment:{args}')
    setting = get_setting(args, itr=0)
    path = os.path.join(projectPath, args.checkpoints, setting)

    model_name = f'checkpoint.cpu.pth' if cuda == 'cpu' else 'checkpoint.pth'
    model_path = f'{path}/{model_name}'
    print(f"获取模型地址: {model_path}")
    '''
    写文件[注意：只将模型权重写进pkl文件中，解析模型应该放置算法内:
    1. 解耦
    2. pickle.dump() 对类对象封装时，会根据所加载的类对象对数据进行对象化，同时也会把类对象的路径也打包进去，
    记录下它是根据那个目录下的哪个类进行封装的，同样解析时也要找到对应目录下的对应类进行解析还原]
    '''
    # torch.save(model.state_dict(), 'model.pkl')
    out_pkl = f"{path}/{args.seq_len}_{args.pred_len}_{args.decomposition}.{model_name.replace('.pth', '')}.pkl"
    with open(out_pkl, 'wb') as f:
        pickle.dump(torch.load(model_path), f)
    return


def loads_model(serialization_model, **kwargs):
    """加载模型.
    :param serialization_model: dumps_model 返回的序列化文件
    :param kwargs:
    :return: 模型
    """
    # 平台为Scikit-learn模型，提供如下序列化方法(pickle)，您也可以import或者自定义其他序列化方法
    # from aiops.logic.toolkit.serialization import pickle
    with open(serialization_model, 'rb') as f:
        model = pickle.load(f)
    return model

    # model = pickle.loads(serialization_model)
    # return model


if __name__ == '__main__':
    # convert_gpu2cpu_model_state("./args_aiops_patchtst_h2p1_decp.json")
    # convert_pth2pkl("./args_aiops_patchtst_h2p1_decp.json")
    convert_pth2pkl("../scripts/product/args_bcs_patchtst_d3h6p1.json", cuda='cpu')
    # model_dict = loads_model(
    #     # serialization_model="./checkpoints/AIOps_36_0_6_PatchTST_AIOps_ftS_sl36_ll0_pl6_dm16_nh4_el3_dl1_df128_fc1_ebtimeF_dtFalse_Exp_0/model.pkl"
    #     "./checkpoints/AIOps_2880_0_1440_PatchTST_AIOps_ftS_sl2880_ll0_pl1440_dm16_nh4_el3_dl1_df128_fc1_ebtimeF_dtFalse_Exp_decp1_0/2880_1440_1_model_cpu.pkl"
    # )
    # print(model_dict)
