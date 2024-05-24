#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/12/23 17:18
import warnings

warnings.filterwarnings('ignore')

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

TZ = "Asia/Shanghai"


def stamp2date(stamps):
    return pd.to_datetime(stamps, unit='s', utc=True).tz_convert(TZ)  # .strftime("%Y-%m-%d %H:%M:%S")


def plot_predict_result(preds_result, source_ts_csv, **kwargs):
    """
    plot predict for timeseries;
    :param preds_result:
    :param source_ts_csv:
    :return:
    """
    from datetime import datetime
    isload = kwargs.get('isload', False)  # noqa
    if isload:
        preds_result = np.load(preds_result)

    preds, trues, stamps = preds_result[0], preds_result[1], preds_result[2]
    # 仅画出预测的每个点的第一个值作为可视化对象；
    df = pd.DataFrame()
    df['true'] = np.concatenate((trues[:-1, 0, :].reshape(-1), trues[-1, :, :].reshape(-1)))
    df['pred'] = np.concatenate((preds[:-1, 0, :].reshape(-1), preds[-1, :, :].reshape(-1)))
    # 随机挑点画出一次性预测未来多个点的数据；，并且画出预测最大值的线；
    # 预测的最大值；
    pred_len = preds.shape[1]
    df['pred_max'] = np.concatenate(
        (preds.transpose((0, 2, 1)).reshape(-1, pred_len).max(axis=1), np.repeat(np.nan, pred_len - 1)))

    df['timestamp'] = np.concatenate((stamps[:-1, 0, :].reshape(-1), stamps[-1, :, :].reshape(-1)))
    df = df.set_index('timestamp')

    src_data = pd.read_csv(source_ts_csv).set_index(['timestamp'])
    df = pd.concat([src_data, df], axis=1).sort_index()
    df.index = stamp2date(df.index)

    start, end = df.index[0], df.index[-1]
    fig = plt.figure(figsize=(30, 5))
    ax = fig.add_subplot(111)
    ax.plot(df['value'], color='#4169E1', ms=1, lw=0.5, markeredgecolor='#183dd8', label='value')
    ax.plot(df['pred'], marker="o", color='r', ms=1, lw=0.5, alpha=1, label='pred')
    # pred enhance；
    ax.plot(df['pred'].iloc[-pred_len:], color='r', lw=5, alpha=0.4)
    ax.plot(df['pred_max'], color='g', ms=2, lw=1, alpha=1, label='pred-max')

    ax.set_ylim([0, max(df['value']) * 1.05])
    # 有时差问题
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1, tz=TZ))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))

    ax.tick_params(which='both', rotation=30, labelsize=8, axis="x")
    ax.set_xlim([start, end])

    ax.xaxis.grid(True, which="major", linestyle='dashed', linewidth=1)  # x坐标轴的网格使用主刻度
    ax.yaxis.grid(True, which="major", linestyle='dashed', linewidth=1)
    ax.set_ylabel(f"value vs predict", fontsize=10)
    ax.legend(loc='right')
    # 存储
    savefig = kwargs.get("savefig")
    if not savefig:
        savefig = "../var/"
    os.makedirs(savefig, exist_ok=True)
    out_fig = f"{savefig}/{kwargs.get('title', datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))}.png"
    print(f"可视化存储：{out_fig}")
    plt.savefig(out_fig, dpi=300)
    plt.close()


if __name__ == '__main__':
    plot_predict_result(
        # npy_data="../results/AIOps_2880_0_1440_DLinear_AIOps_ftS_sl2880_ll0_pl1440_dm16_nh4_el3_dl1_df128_fc1_ebtimeF_dtFalse_Exp_0//real_prediction.npy",
        # source_ts_csv="../dataset/bkbase/data/BCS-K8S-41021~ieg-bkbase-dataflow-flink-prod~sql-8ac47ad112314d90a9aac641d55d670a-78f55969f9-29vsk~cpu_usage.csv"
        # preds_result="../results/AIOps_2880_0_1440_PatchTST_AIOps_ftS_sl2880_ll0_pl1440_dm16_nh4_el3_dl1_df128_fc1_ebtimeF_dtFalse_Exp_0/pred.npy",
        # source_ts_csv="../dataset/bkbase/data/BCS-K8S-41021~ieg-bkbase-dataflow-flink-prod~sql-8ac47ad112314d90a9aac641d55d670a-78f55969f9-29vsk~cpu_usage.csv",
        # preds_result="../results/AIOps_2880_0_1440_PatchTST_AIOps_ftS_sl2880_ll0_pl1440_dm16_nh4_el3_dl1_df128_fc1_ebtimeF_dtFalse_Exp_decp1_0/BCS-K8S-41021~ieg-bkbase-dataflow-flink-prod~sql-8ac47ad112314d90a9aac641d55d670a-78f55969f9-29vsk~cpu_usage_real_prediction.npy",

        # source_ts_csv="../dataset/bkbase/data/queryset_cpu.csv",
        # preds_result="../results/AIOps_2880_0_1440_PatchTST_AIOps_ftS_sl2880_ll0_pl1440_dm16_nh4_el3_dl1_df128_fc1_ebtimeF_dtFalse_Exp_decp1_0/queryset_cpu_real_prediction.npy",
        source_ts_csv="../dataset/bcs-v2/bcs_data/BCS-K8S-40976-9.144.207.234.csv",
        preds_result="../pred_results/BCS_96_12_1_PatchTST_AIOps_ftS_sl96_ll0_pl12_dm16_nh4_el3_dl1_df128_fc1_ebtimeF_dtFalse_Exp_decp1_0/BCS-K8S-40976-9.144.207.234_real_prediction.npy",
        isload=True
    )
