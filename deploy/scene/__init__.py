#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/3/18 16:40
__all__ = ("_err_output_format",)

import json

import pandas as pd

"""
目前应用：
# 数据平台任务的资源预测
# BCS的资源预测
"""


def _err_output_format(agent, output_schema, dataframe, err_code=-1, predict_column="prediction"):  # noqa
    """
    错误输出；
    :param agent:
    :param output_schema:
    :param dataframe:
    :param err_code:
        -1：预测异常
        -2：历史数据不足，无法预测
        -3：历史数据质量差，无法预测（历史数据中存在非法值比例过高）
    :param predict_column: 预测字段；
    :return:
    """
    # 以错误码的方式作为预测值，与用户输入的其他字段（或者预测的其他字段）一起join作为最终输出（为了保持输出格式一致）；
    result = output_schema.integrate(  # noqa
        input_df=dataframe[-1:],  # 当前时间戳的内容值；
        patch_df=pd.DataFrame({predict_column: [json.dumps([err_code])]}),
        concat=True,
    )
    return result
