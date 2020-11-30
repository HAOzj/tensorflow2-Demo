# !/usr/bin/python
# -*- coding:utf-8 -*-
"""
Created on 24 Nov, 2020

Author : woshihaozhaojun@sina.com
"""
import os
import json
from tqdm import tqdm
from src.conf_loader import (data_dir, indexing_dir)


def get_map(data_dir):
    """生成
    1. 视频id和序号的映射,序号从1开始,0留给unk
    2. 栏目id和代表视频的映射

    :param data_dir: 映射文件所在文件夹
    :return:
    """
    vid2idx, cid2vidx = {}, {}
    idx = 1
    for file in os.listdir(data_dir):
        if file.startswith("vid2title"):
            file = os.path.join(data_dir, "vid2title.json")

            with open(file, "r", encoding="utf8") as fp:
                lines = fp.readlines()
                for line in tqdm(lines):
                    dict_tmp = json.loads(line)
                    if all([key in dict_tmp for key in ["cover_id", "title"]]):
                        vid = dict_tmp["cover_id"]
                        vid2idx[vid] = idx
                        idx += 1

    for file in os.listdir(data_dir):
        if file.startswith("cid2vids"):
            file = os.path.join(data_dir, file)

            with open(file, "r", encoding="utf8") as fp:
                lines = fp.readlines()
                for line in tqdm(lines):
                    dict_tmp = json.loads(line)
                    if all([key in dict_tmp for key in ["cid", "vids"]]):
                        cid = dict_tmp["cid"]
                        vids = dict_tmp["vids"]
                        cid2vidx[cid] = [vid2idx[vid] for vid in vids.split(",") if vid in vid2idx]

    return vid2idx, cid2vidx


def write_map(vid2idx, cid2vidx, map_dir):
    files = ["vid2idx.json", "cid2vidx.json"]
    dicts = [vid2idx, cid2vidx]
    for file_tmp, dict_tmp in zip(files, dicts):
        with open(os.path.join(map_dir, file_tmp), "w", encoding="utf8") as fp:
            json.dump(dict_tmp, fp)


def main(data_dir, map_dir):
    vid2idx, cid2vidx = get_map(data_dir)
    write_map(vid2idx, cid2vidx, map_dir)


if __name__ == "__main__":
    main(data_dir, indexing_dir)
