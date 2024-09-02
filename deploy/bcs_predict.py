#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/3/14 09:38

"""
采用patchTST模型预测；
"""
import copy
import traceback
import numpy as np
import pandas as pd

from modeling.logic.toolkit.logger import serving_logger as logger
from deploy.platform_resource_predict import _err_output_format
from deploy.ts_pred_utils.patchTST import Model, TSTAgent, invalid_check, loads_model
from deploy.ts_pred_utils.timeseries_process import time_series_process
from utils.aiops_deploy_utils.oformat import DataTypesSchema, OutputIntegrate


class HyperParam:
    """hyper parametric define"""

    c_in = 1
    context_window = 96
    target_window = 12

    e_layers = 3
    n_heads = 4
    d_model = 16
    d_ff = 128
    dropout = 0.3
    fc_dropout = 0.3
    head_dropout = 0.0

    individual = 0

    patch_len = 16
    stride = 8
    padding_patch = "end"

    revin = 1
    affine = 0
    subtract_last = 0

    decomposition = 1
    kernel_size = 25


def predict(dataframe, dataframe_schema, model, **kwargs):
    """
    时序预测；
    :param dataframe:
    :param dataframe_schema:
    :param model:
    :param kwargs:
    :return:
    """
    df_cols = list(set(dataframe.columns) - set(dataframe_schema.get("system_columns")))
    logger.warning(f"输入数据长度：{dataframe.shape}, 数据:\n{dataframe[df_cols].head(5)}")

    def _reload_model():
        """#也可以不用防止用户使用模型二次开发时，遇到加载模型：用于将预训练的参数权重加载到新的模型之中"""
        if not isinstance(model, dict):
            return model
        _model = Model().float()
        _model.load_state_dict(model)
        _model.eval()
        return copy.copy(_model)

    predict_column = "prediction"
    output_schema = OutputIntegrate(
        datatypes_schema=DataTypesSchema(dtypes={predict_column: str, "value": np.float}),  # 'max_predict': np.float,
        dataframe_schema=dataframe_schema,
    )
    # load model:
    model = _reload_model()

    # start predict
    agent = TSTAgent(feature_list=["value"], model=model)
    positive_check = kwargs.get("positive_check", 1)  # BCS资源>0;
    invalid_rate_threshold = kwargs.get("invalid_rate", 0.3)  # 预测参数；

    try:
        # 1. time series preprocess: filled ts and frequency normalization;
        dataframe, _, _ = time_series_process(df=dataframe)
        logger.info(f"start to predict and input dataframe shape={dataframe.shape}")

        # 2. check history data to be enough;
        if len(dataframe) < HyperParam.context_window:
            logger.error(f"数据长度:{len(dataframe)}(<={HyperParam.context_window}), 无法预测.")
            return _err_output_format(
                agent=agent,
                output_schema=output_schema,
                dataframe=dataframe,
                predict_column=predict_column,
                err_code=-2,
            )
        # 2. get history data and predict;
        dataframe = dataframe.iloc[-HyperParam.context_window:]
        invalid_rate = invalid_check(dataframe["value"].values, positive_check=positive_check)

        if invalid_rate >= invalid_rate_threshold:
            logger.error(
                "无效数据[空值{}]比例过高:{}%(阈值<={}%), 无法预测.".format(
                    f"/零值/负值" if kwargs.get("positive_check", 0) else "",
                    invalid_rate * 100,
                    invalid_rate_threshold * 100,
                )
            )
            return _err_output_format(
                agent=agent,
                output_schema=output_schema,
                dataframe=dataframe,
                predict_column=predict_column,
                err_code=-3,
            )
        # 3. predict
        predict_result_df = agent.run(dataframe=dataframe, to_dataframe=True)
        return output_schema.integrate(input_df=dataframe[-1:], patch_df=predict_result_df, concat=True)

    except Exception:
        _err_log = traceback.format_exc()
        logger.error(f"Predict failed, error: {_err_log}")
        return _err_output_format(
            agent=agent,
            output_schema=output_schema,
            dataframe=dataframe,
            predict_column=predict_column,
            err_code=-1,
        )


def test_online_deploy_predict(pkl_model, predict_df):
    """
    :param pkl_model:
    :param predict_df:
    :return:
    """
    from src import projectPath

    patch_model = loads_model(serialization_model=f"{projectPath}/dataset/{pkl_model}", configs=HyperParam)

    # 加载模型，预测结果；
    result = predict(
        dataframe=predict_df,
        dataframe_schema={
            "feature_columns": ["dtEventTime", "value"],
            "system_columns": [],
        },
        model=patch_model,
        config=HyperParam,
        positive_check=1,
        invalid_rate=0.3,
    )
    with pd.option_context("expand_frame_repr", False, "display.max_rows", None):
        print(result)

    # dataframe, _, _ = time_series_process(df=df)
    # print(dataframe)


if __name__ == "__main__":
    from src import projectPath

    # df = pd.read_csv(f"{projectPath}/dataset/queryset_2024-03-14_1631.csv", low_memory=False)[
    #     ['dtEventTime', 'dtEventTimeStamp', 'value']]
    # df = df.rename(columns={'dtEventTimeStamp': 'timestamp'})

    df = pd.read_csv(f"{projectPath}/dataset/BCS-K8S-40976-9.150.12.135.csv", low_memory=False)
    test_online_deploy_predict(pkl_model="96_12_1.checkpoint.cpu.pth.pkl", predict_df=df)
