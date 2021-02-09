# !/usr/bin/python
# -*- coding:utf-8 -*-
"""
Created on 09 Nov, 2020

Updated on 26 Jan, 2021

Author : woshihaozhaojun@sina.com
"""
import os
import sys
import time
src_path = os.path.abspath("..")
sys.path.append(src_path)
from src.model import BST_DSSM as Model
from src.get_dataset import load_data
from src.conf_loader import (
 MODEL_DIR, tfrecord_dir, MODEL_WEIGHT_DIR, FIT_LOGS_DIR,
 user_max_len, item_max_len,
 n_epoch, batch_size, test_ratio, val_ratio
)
from src.utils import (print_run_time, NBatchLogger, get_all_metrics)
import tensorflow as tf


class Break(Exception):
    pass


@print_run_time
def train(k=100, log_file="loss_on_val.txt"):
    train_data, test_data, val_data = load_data(
        tfrecord_dir, user_max_len, item_max_len, batch_size=batch_size,
        test_ratio=test_ratio, val_ratio=val_ratio)

    model = Model()
    model.compile(
        optimizer=model.optimizer,
        loss=model.loss_fn
    )
    model.build(input_shape=[(None, item_max_len), (None, user_max_len)])

    losses_val = [1 for _ in range(2)]
    start_time = time.time()

    """生成发展集的dataset
    
    分为输入和标签
    """
    def convert2input_x(mini_batch):
        return mini_batch["item_feature"], mini_batch["user_feature"]

    def convert2input_y(mini_batch):
        return mini_batch["label"]

    val_x, val_y = val_data.map(convert2input_x), val_data.map(convert2input_y)

    try:
        def convert2input(mini_batch):
            x_train = (mini_batch["item_feature"], mini_batch["user_feature"])
            y_train = mini_batch["label"]
            return x_train, y_train

        # 每k个batch输出损失以及tensorboard的callback
        callbacks = NBatchLogger(k)
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=FIT_LOGS_DIR, histogram_freq=1)

        for epoch in range(n_epoch):
            # 喂入generator可以用多进程
            model.fit(
                x=train_data.map(convert2input),
                verbose=2,
                callbacks=[callbacks, tensorboard_callback],
                shuffle=True,
                workers=4,
                use_multiprocessing=True)

            # for mini_batch in train_data.shuffle(buffer_size).as_numpy_iterator():
            #     x_train = (mini_batch["item_feature"], mini_batch["user_feature"])
            #     y_train = mini_batch["label"]
            #     batch_num += 1
            #     model.fit(x_train, y_train)

            loss_val = get_all_metrics(
                model=model, epoch=epoch,
                val_x=val_x, val_y=val_y,
                start_time=start_time, loss_fn=model.loss_fn)
            losses_val.append(loss_val)

            if losses_val[-1] >= losses_val[-2] >= losses_val[-3]:
                print(losses_val)
                raise Break("break")
            if losses_val[-1] < losses_val[-2]:

                # subclassed model最好用save_weights
                model.save(
                    filepath=os.path.join(
                        MODEL_DIR,
                        f"{user_max_len}_{item_max_len}_{model.emb_dim}_{batch_size}_{epoch}"),
                    save_format="tf"
                )
                model.save_weights(filepath=os.path.join(
                        MODEL_WEIGHT_DIR,
                        f"{user_max_len}_{item_max_len}_{model.emb_dim}_{batch_size}_{epoch}"))
    except Break as e:
        print(e)


def main():
    train(k=100)


if __name__ == "__main__":
    main()
