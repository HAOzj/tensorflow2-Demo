# !/usr/bin/python
# -*- coding:utf-8 -*-
"""
Created on 09 Nov, 2020

Author : woshihaozhaojun@sina.com
"""
import tensorflow as tf
import numpy as np
import warnings
warnings.filterwarnings("ignore")

x = np.random.randint(low=0, high=100, size=(5, 2))
y = np.random.randint(0, high=2, size=(5, 1))
data = np.concatenate((x, y), axis=1)
print(data)


def from_tensor():
    print("------from tensor slices")

    dataset = tf.data.Dataset.from_tensor_slices(data)
    dataset = dataset.repeat(2).batch(3, drop_remainder=True).shuffle(buffer_size=5).map(lambda x: x+1)

    batch_num = 1
    for next_element in dataset.as_numpy_iterator():
        print(batch_num)
        print(next_element)
        batch_num += 1


def from_gen():
    print("\n\n------from generator")

    def gen():
        for i in range(x.shape[0]):
            x_new = np.concatenate((x[i, :], np.random.randint(low=0, high=100, size=i)))
            yield {"x": x_new, "y": y[i, :]}

    # 多个输入或输出时,需要output_types和output_shapes参数
    output_shapes = {"x": tf.TensorShape([None]), "y": tf.TensorShape([1])}
    output_types = {"x": tf.int8, "y": tf.int8}
    padded_shapes = {"x": [10], "y": [None]}
    padding_values = {"x": 0, "y": 0}
    dataset = tf.data.Dataset.from_generator(gen, output_types=output_types, output_shapes=output_shapes)\
        .repeat(2)\
        .padded_batch(batch_size=2, padded_shapes=padded_shapes, drop_remainder=False)\
        .shuffle(buffer_size=5)
    # dataset = dataset.repeat(2).batch(3, drop_remainder=True).shuffle(buffer_size=5)
    batch_num = 1
    for next_element in dataset.as_numpy_iterator():
        print(batch_num)
        print(next_element)
        batch_num += 1


# a = tf.random.uniform(shape=(2, 2))
# print(a)
# norm = tf.sqrt(tf.reduce_sum(tf.square(a), 1, keepdims=True))
# print(norm)
# normalized_embeddings = a / norm
# print(normalized_embeddings)
from_gen()
