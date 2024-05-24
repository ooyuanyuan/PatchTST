#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/7/14 18:23


import json
import argparse


def get_args_from_json(json_file_path):
    with open(json_file_path) as f:
        data = json.load(fp=f)
    return data


if __name__ == '__main__':
    args = get_args_from_json('../args.json')
    ns = argparse.Namespace(**args)
