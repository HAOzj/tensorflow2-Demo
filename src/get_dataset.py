# !/usr/bin/python
# -*- coding:utf-8 -*-
"""
Created on 26 Nov, 2020

Author : woshihaozhaojun@sina.com
"""
import os
import tensorflow as tf
import warnings
warnings.filterwarnings("ignore")


def parse_tfrecords(serialized_example):
    """解析获得例子的用户行为序列,物品表示序列和标签

    :param serialized_example:
    :return:
    """
    context_features = {
        "label": tf.io.FixedLenFeature([1], dtype=tf.float32)
    }
    sequence_features = {
        "user_feature": tf.io.FixedLenSequenceFeature([], dtype=tf.int64),
        "item_feature": tf.io.FixedLenSequenceFeature([], dtype=tf.int64)
    }

    # 一次仅仅解析一条样本example
    context_parsed, sequence_parsed, _ = tf.io.parse_sequence_example(
        serialized=serialized_example,
        context_features=context_features,
        sequence_features=sequence_features)
    res = dict()
    user_feature = sequence_parsed["user_feature"]
    item_feature = sequence_parsed["item_feature"]
    label = context_parsed["label"]
    res["user_feature"] = user_feature
    res["item_feature"] = item_feature
    res["label"] = label
    return res


def load_data(tfrecord_dir, user_max_len, item_max_len, batch_size, test_ratio, val_ratio, strict_flag=False):
    """从tfrecord文件中解析样例,生成训练集,测试集和预测集

    :param tfrecord_dir, path, TFRecord文件所在的文件夹
    :param user_max_len, int, 用户行为序列的最大长度
    :param item_max_len, int, 物品序列的最大长度
    :param batch_size, int
    :param test_ratio, float, 测试集的占比
    :param val_ratio, float, 发展集的占比
    :param strict_flag, bool, 是否把数据集全部shuffle后分割.对于大数据量,建议为False
    :return:
    """
    padded_shapes = {"user_feature": [user_max_len], "item_feature": [item_max_len], "label": [1]}
    filenames = []
    for file in os.listdir(tfrecord_dir):
        tf_file = os.path.join(tfrecord_dir, file)
        filenames.append(tf_file)

    # 全部数据随机来切分数据集
    if strict_flag:
        filenames = tf.convert_to_tensor(filenames, dtype=tf.string)

        dataset = tf.data.TFRecordDataset(filenames).map(parse_tfrecords)

        # 获得sample size
        sample_size = 0
        for _ in dataset:
            sample_size += 1
        print("------sample size=", sample_size)

        dataset = dataset.shuffle(sample_size)\
            .padded_batch(batch_size=batch_size, padded_shapes=padded_shapes, drop_remainder=False)

        sample_size = int(sample_size / batch_size)
        print(f"------in total, we have {sample_size} batches, each made up of {batch_size} samples")
        test_size = int(sample_size * test_ratio)
        val_size = int(sample_size * val_ratio)
        test_data = dataset.take(test_size)
        res_data = dataset.skip(test_size)
        val_data = res_data.take(val_size)
        train_data = res_data.skip(val_size)

    else:
        # 按照文件来切分数据集
        file_cnt = len(filenames)
        test_cnt, val_cnt = int(file_cnt * test_ratio), int(file_cnt * val_ratio)
        test_files = filenames[: test_cnt]
        filenames = filenames[test_cnt:]
        val_files = filenames[: val_cnt]
        train_files = filenames[val_cnt:]

        test_files = tf.convert_to_tensor(test_files, dtype=tf.string)
        val_files = tf.convert_to_tensor(val_files, dtype=tf.string)
        train_files = tf.convert_to_tensor(train_files, dtype=tf.string)

        data_list = []
        for files_tmp in [train_files, test_files, val_files]:
            dataset_tmp = tf.data.TFRecordDataset(files_tmp).map(parse_tfrecords) \
                .padded_batch(batch_size=batch_size, padded_shapes=padded_shapes, drop_remainder=False)
            data_list.append(dataset_tmp)
        [train_data, test_data, val_data] = data_list

    return train_data, test_data, val_data
