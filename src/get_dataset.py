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


def load_data(tfrecord_dir, user_max_len, item_max_len, n_epoch, batch_size, buffer_size, test_ratio, val_ratio):
    """从tfrecord文件中解析样例,生成训练集,测试集和预测集

    :param tfrecord_dir:
    :param user_max_len:
    :param item_max_len:
    :param n_epoch:
    :param batch_size:
    :param buffer_size:
    :return:
    """
    padded_shapes = {"user_feature": [user_max_len], "item_feature": [item_max_len], "label": [1]}
    filenames = []
    for file in os.listdir(tfrecord_dir):
        tf_file = os.path.join(tfrecord_dir, file)
        filenames.append(tf_file)
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

    test_size = int(sample_size * test_ratio)
    val_size = int(sample_size * val_ratio)
    test_data = dataset.take(test_size)
    res_data = dataset.skip(test_size)
    val_data = res_data.take(val_size)
    train_data = res_data.skip(val_size)\
        .repeat(n_epoch)\
        .shuffle(buffer_size=buffer_size)

    return train_data, test_data, val_data
