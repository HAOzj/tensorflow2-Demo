# !/usr/bin/python
# -*- coding:utf-8 -*-
"""
Created on 09 Nov, 2020

Author : woshihaozhaojun@sina.com
"""
import os
import sys
from tqdm import tqdm
src_path = os.path.abspath("..")
print(src_path)
sys.path.append(src_path)
from src.model_tf2 import BST_DSSM as Model
from src.get_dataset import load_data
from src.conf_loader import (
 MODEL_DIR, tfrecord_dir,
 user_max_len, item_max_len,
 n_epoch, batch_size, buffer_size, test_ratio, val_ratio
)


train_data, test_data, val_data = load_data(
    tfrecord_dir, user_max_len, item_max_len, n_epoch, batch_size,
    buffer_size=buffer_size,
    test_ratio=test_ratio, val_ratio=val_ratio)
model = Model()
model.compile(
    optimizer=model.optimizer,
    loss=model.loss_fn
    # ,
    # metrics=[
    #     tf.keras.metrics.AUC,
    #     tf.keras.metrics.binary_accuracy]
)
model.build(input_shape=[(None, item_max_len), (None, user_max_len)])


def val_data_generator(val_data):
    for mini_batch in val_data.as_numpy_iterator():
        x_train = (mini_batch["item_feature"], mini_batch["user_feature"])
        y_train = mini_batch["label"]
        yield x_train, y_train


batch_num = 0
losses_val = [1 for _ in range(2)]
# print("train_data is of type", type(train_data))
for mini_batch in train_data.as_numpy_iterator():
    x_train = (mini_batch["item_feature"], mini_batch["user_feature"])
    y_train = mini_batch["label"]
    batch_num += 1
    model.fit(x_train, y_train)

    if batch_num % 20 == 0:
        loss_val = model.evaluate(val_data_generator(val_data), verbose=False)
        print(f"{batch_num} batch, loss on val data is ", loss_val)
        losses_val.append(loss_val)
        if losses_val[-1] >= losses_val[-2] >= losses_val[-3]:
            print(losses_val)
            break

model.summary()
model.save(MODEL_DIR, save_format="tf")

