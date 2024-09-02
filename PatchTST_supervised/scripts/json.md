数据配置参数说明；

```
     "data": "AIOps", ### 自定义数据获取类；
"root_path": "./dataset/bkbase", ### 数据配置目录，里面包含：train.txt, test.txt, pred.txt, 同时也包含数据目录；
"data_path": "data", ### 数据目录；
"pretrain_data_list": "train.txt",
"test_data_list": "test.txt",
"valid_data_list": "valid.txt",
```

数据格式如下：

```
timestamp,value
1683504000,2.3340639999999744
1683504060,2.255731333333339
```