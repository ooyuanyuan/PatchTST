from data_provider.data_loader import (
    Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Pred,
    Dattest_AIOps_Custom, Dattest_AIOps_Pred, Dattest_AIOps_Pred_2
)
from torch.utils.data import DataLoader

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'custom': Dataset_Custom,
    'AIOps': Dattest_AIOps_Custom  # 平台自定义读取数据；
}


# def data_provider(args, flag):
#     Data = data_dict[args.data]
#     timeenc = 0 if args.embed != 'timeF' else 1
#
#     if flag == 'test':
#         shuffle_flag = False
#         drop_last = True
#         batch_size = args.batch_size
#         freq = args.freq
#     elif flag == 'pred':
#         shuffle_flag = False
#         drop_last = False
#         batch_size = 1
#         freq = args.freq
#         Data = Dataset_Pred
#     else:
#         shuffle_flag = True
#         drop_last = True
#         batch_size = args.batch_size
#         freq = args.freq
#
#     data_set = Data(
#         root_path=args.root_path,
#         data_path=args.data_path,
#         flag=flag,
#         size=[args.seq_len, args.label_len, args.pred_len],
#         features=args.features,
#         target=args.target,
#         timeenc=timeenc,
#         freq=freq
#     )
#     print(flag, len(data_set))
#     data_loader = DataLoader(
#         data_set,
#         batch_size=batch_size,
#         shuffle=shuffle_flag,
#         num_workers=args.num_workers,
#         drop_last=drop_last)
#     return data_set, data_loader


def data_provider(args, flag, **kwargs):
    """
    数据提供的入口：
    :param args:
    :param flag:
    :return:
    """
    Data = data_dict[args.data]  # Dataset_ETT_hour <- "ETTh1"
    timeenc = 0 if args.embed != 'timeF' else 1  # embedding；

    if flag == 'test':  # 数据标签
        shuffle_flag = False
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq  # "h"，频率；
    elif flag == 'pred':
        shuffle_flag = False
        drop_last = False
        batch_size = 1
        freq = args.freq
        Data = Dattest_AIOps_Pred_2  # Dataset_Pred  # 预测的数据处理；

    else:  # train
        shuffle_flag = True  # 训练时需要shuffle;
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq

    # 加载不同的数据；# Dattest_AIOps_Custom； # noqa
    data_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,  # dataset path.
        # add data, only act in aiops data[for file get data list -> read data one by one]
        pretrain_data_list=args.pretrain_data_list,
        test_data_list=args.test_data_list,
        valid_data_list=args.valid_data_list,

        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],  # 336/24=14D，48=2D，96=4D；
        features=args.features,  # 'M'
        target=args.target,  # target col;
        timeenc=timeenc,
        freq=freq,
        **kwargs
    )

    print(f"{flag:>5}: data_size={len(data_set)}, batch_size={batch_size}")
    data_loader = DataLoader(  # return iterator:是对每个序列的index迭代，获取数据：data[index]
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last

    )  # 当最后一组数据不够batch大小时选择是否抛弃
    return data_set, data_loader
