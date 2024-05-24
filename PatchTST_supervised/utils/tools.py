import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.switch_backend('agg')

TZ = "Asia/Shanghai"


def stamp2date(stamps):
    return pd.to_datetime(stamps, unit='s', utc=True).tz_convert(TZ)  # .strftime("%Y-%m-%d %H:%M:%S")


def adjust_learning_rate(optimizer, scheduler, epoch, args, printout=True):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == 'type3':
        lr_adjust = {epoch: args.learning_rate if epoch < 3 else args.learning_rate * (0.9 ** ((epoch - 3) // 1))}
    elif args.lradj == 'constant':
        lr_adjust = {epoch: args.learning_rate}
    elif args.lradj == '3':
        lr_adjust = {epoch: args.learning_rate if epoch < 10 else args.learning_rate * 0.1}
    elif args.lradj == '4':
        lr_adjust = {epoch: args.learning_rate if epoch < 15 else args.learning_rate * 0.1}
    elif args.lradj == '5':
        lr_adjust = {epoch: args.learning_rate if epoch < 25 else args.learning_rate * 0.1}
    elif args.lradj == '6':
        lr_adjust = {epoch: args.learning_rate if epoch < 5 else args.learning_rate * 0.1}
    elif args.lradj == 'TST':
        lr_adjust = {epoch: scheduler.get_last_lr()[0]}

    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        if printout: print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')


def test_params_flop(model, x_shape):
    """
    If you want to thest former's flop, you need to give default value to inputs in model.forward(), the following code can only pass one argument to forward()
    """
    model_params = 0
    for parameter in model.parameters():
        model_params += parameter.numel()
        print('INFO: Trainable parameter count: {:.2f}M'.format(model_params / 1000000.0))
    from ptflops import get_model_complexity_info
    with torch.cuda.device(0):
        macs, params = get_model_complexity_info(model.cuda(), x_shape, as_strings=True, print_per_layer_stat=True)
        # print('Flops:' + flops)
        # print('Params:' + params)
        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))


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
