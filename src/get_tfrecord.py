# !/usr/bin/python
# -*- coding:utf-8 -*-
"""
Created on 25 Nov, 2020

Author: woshihaozhaojun@sina.com

Refer to: https://blog.csdn.net/Jerr__y/article/details/78594740
"""
import os
import sys
src_path = os.path.abspath("..")
print(src_path)
sys.path.append(src_path)
import json
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from src.conf_loader import (
    tfrecord_dir, click_seq_dir, indexing_dir,
    user_max_len, item_max_len
)
from src.utils import (
    print_run_time, do_cprofile
)
from multiprocessing import (
    Pool, Manager, Process, context
)


@print_run_time
def load_map(indexing_dir):
    """获得1.栏目id到内容视频的idx序列, 2.视频id到idx 的映射"""
    res = []
    file_list = ["cid2vidx.json", "vid2idx.json"]
    for file in file_list:
        file_path = os.path.join(indexing_dir, file)

        with open(file_path, "r", encoding="utf8") as fp:
            res.append(json.load(fp))
    return res


def get_col_indices(click_seq, vid2idx, min_his):
    """从点击序列中获得点击是栏目且序号大于等于min_his的序号并把vid换成idx

    序号大于等于min_his是为了保证点击栏目前至少有足够的点击行为
    e.g.
        点击序列为[vid0, vid1, vid2, vid3, vid4_cid1, vid5], 其中vid4是栏目中的

    :param click_seq, iterable, 用户的点击序列,每个元素为<vid>或者,当点击视频在栏目中时,<vid>_<cid>
    :param vid2idx, dict, vid到序号的映射
    :param min_his, int, 最小的用户行为数
    :returns
        点击视频的idx的序列
        栏目下标的序列
        栏目id的序列
    """
    idx_seq = []
    col_indices = []
    cids = []
    for idx, item in enumerate(click_seq):
        compo_list = item.split("_")
        if len(compo_list) > 1 and idx >= min_his:
            col_indices.append(idx)
            cids.append(compo_list[1])
        idx_seq.append(vid2idx.get(compo_list[0], 0))

        # 每个用户只取前10个栏目点击
        if len(cids) > 10:
            break

    return idx_seq, col_indices, cids


def write_example(user_seq, item_seq, label, writer):
    """

    :param user_seq: 用户历史点击序列
    :param item_seq: 栏目的视频序列
    :param label: int, choices=[0, 1]
    :param writer: TFRecord
    :return:
    """
    # 限制user_seq和item_seq的长度
    user_seq = user_seq[-user_max_len:]
    item_seq = item_seq[-item_max_len:]

    user_feature = [
        tf.train.Feature(int64_list=tf.train.Int64List(value=[vid])) for vid in user_seq
    ]
    item_feature = [
        tf.train.Feature(int64_list=tf.train.Int64List(value=[vid])) for vid in item_seq
    ]

    seq_example = tf.train.SequenceExample(
        # context 来放置非序列化部分
        context=tf.train.Features(feature={
            "label": tf.train.Feature(float_list=tf.train.FloatList(value=[label]))
        }),
        # feature_lists 放置变长序列
        feature_lists=tf.train.FeatureLists(feature_list={
            "user_feature": tf.train.FeatureList(feature=user_feature),
            "item_feature": tf.train.FeatureList(feature=item_feature),
        })
    )

    """ 存成定长序列
    if len(user_seq) < user_max_len:
        user_seq = user_seq + [0 for _ in range(user_max_len - len(user_seq))]
    if len(item_seq) < item_max_len:
        item_seq = item_seq + [0 for _ in range(item_max_len - len(item_seq))]

    seq_example = tf.train.Example(
        # context 来放置非序列化部分
        features=tf.train.Features(feature={
            "label": tf.train.Feature(float_list=tf.train.FloatList(value=[label])),
            "user_feature": tf.train.Feature(int64_list=tf.train.Int64List(value=user_seq)),
            "item_feature": tf.train.Feature(int64_list=tf.train.Int64List(value=item_seq))
        })
    )
    """

    serialized = seq_example.SerializeToString()
    writer.write(serialized)


def _trans_file_to_tfrecord(data_file, tf_file, min_his, vid2idx, cid2vidx, cids_candidate):
    """

    :param data_file: .json文件地址
    :param tf_file: tfrecord的handler
    :return:
    """
    cnt = 0
    print(f"Start convert {data_file} to {tf_file}")
    with tf.io.TFRecordWriter(tf_file) as writer:
        with open(data_file, "r", encoding="utf8") as fp:
            lines = fp.readlines()
            for line in tqdm(lines):
                try:
                    dict_tmp = json.loads(line)
                    if "click_seq" not in dict_tmp:
                        continue
                    click_seq = dict_tmp["click_seq"]

                    if len(click_seq) <= min_his:
                        continue
                    idx_seq, col_indices, cids = get_col_indices(click_seq, vid2idx, min_his)

                    # 每一次栏目点击都可以作为一个基础样例
                    for col_index, cid in zip(col_indices, cids):
                        user_seq = idx_seq[:col_index]

                        # 每个基础样例可以作为一个正例并利用几何分布随机抽取出一些负例
                        # 先正例
                        pos_seq = cid2vidx.get(cid, [])
                        if not pos_seq:
                            continue
                        write_example(user_seq, pos_seq, 1.0, writer)

                        # # 如果class_ratio大于1
                        # # 用几何分布来生成负例
                        # # 耗时太久
                        # while True:
                        #     trial = np.random.randint(low=0, high=class_ratio+1)
                        #     if trial == 0:
                        #         break
                        #     neg_idx = np.random.randint(low=0, high=col_cnt)
                        #     neg_cid = cids_candidate[neg_idx]
                        #     if neg_cid not in cids:
                        #         CNT_SAMPLE += 1
                        #         if CNT_SAMPLE % capacity == 0:
                        #             writer.close()
                        #             tf_file_path = os.path.join(tfrecord_dir, f"{int(CNT_SAMPLE / capacity)}.tfrecord")
                        #             writer = tf.io.TFRecordWriter(tf_file_path)
                        #         write_example(user_seq, cid2vidx[neg_cid], 0, writer)

                        neg_idx = np.random.randint(low=0, high=len(cids_candidate))
                        neg_cid = cids_candidate[neg_idx]
                        if neg_cid not in cids:
                            cnt += 1
                            # cnt_sample += 1
                            # if CNT_SAMPLE % capacity == 0:
                            #     writer.close()
                            #     tf_file_path = os.path.join(tfrecord_dir, f"{int(CNT_SAMPLE / capacity)}.tfrecord")
                            #     writer = tf.io.TFRecordWriter(tf_file_path)
                            write_example(user_seq, cid2vidx[neg_cid], 0.0, writer)
                            # if cnt % 1000 == 0:
                            #     print(f"{tf_file}写了{cnt} samples")
                except json.decoder.JSONDecodeError:
                    print(line)
                    continue
                except Exception as e:
                    print(e)
                    exit(1)

    print(f"{cnt} samples stored in {tf_file}")
    return cnt


# @do_cprofile("get_tfrecord.prof")
# @print_run_time
def trans_dir_to_tfrecord(
        json_dir=click_seq_dir, tfrecord_dir=tfrecord_dir, indexing_dir=indexing_dir):
    """从json_dir中读取json文件后准话

    :param json_dir: 每个click_seq中都有栏目
    :param tfrecord_dir:
    :param class_ratio:
    :param capacity: 每个.tfrecord文件中的样例数目
    :param indexing_dir:
    :return:
    """
    min_his = 10

    # 用于indexing
    [cid2vidx, vid2idx] = load_map(indexing_dir)
    for cid in cid2vidx:
        cid2vidx[cid] = list(map(int, cid2vidx[cid]))
    for vid in vid2idx:
        vid2idx[vid] = int(vid2idx[vid])

    # 所有栏目的视频序号序列,用于随机生成负例
    cids_candidate = list(cid2vidx.keys())

    # 记录samples数
    # manager = Manager()
    # cnt_sample = manager.Value("i", 0)
    process_pool = Pool(8)  # define 8 process to execute this task
    # import time
    # start_time = time.time()
    # record = []
    print("---start processing click sequence---")
    for file in os.listdir(json_dir):
        json_file = os.path.join(json_dir, file)
        tfrecord_file = os.path.join(tfrecord_dir, file.split(".")[0] + ".tfrecord")
        r = process_pool.apply_async(_trans_file_to_tfrecord, args=(json_file, tfrecord_file, min_his, vid2idx, cid2vidx, cids_candidate))

        try:
            print(r.get(timeout=30))  # 设置子进程的响应超时为30s
        except context.TimeoutError:
            pass

    # for process in record:
    #     process.join(timeout=300)
    print("waiting for all subprocesses done...")
    process_pool.close()  # 关闭进程池,不接受新任务
    process_pool.join()  # 主进程阻塞等待子进程推出

    print("---finished---")


def main():
    trans_dir_to_tfrecord()


if __name__ == "__main__":
    main()
