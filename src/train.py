# !/usr/bin/python
# -*- coding:utf-8 -*-
"""
Created on 09 Nov, 2020

Updated on 26 Jan, 2021

Author : woshihaozhaojun@sina.com
"""
import os
import sys
import numpy as np
from sklearn.metrics import roc_auc_score
src_path = os.path.abspath("..")
sys.path.append(src_path)
from src.model import BST_DSSM as Model
from src.get_dataset import load_data
from src.conf_loader import (
 MODEL_DIR, tfrecord_dir, MODEL_WEIGHT_DIR, FIT_LOGS_DIR,
 user_max_len, item_max_len,
 n_epoch, batch_size, test_ratio, val_ratio
)
from src.utils import (print_run_time, NBatchLogger)
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

    fp = open(log_file, "w")
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

            y_pred_val, y_true_val = [], []
            loss_val = 0
            sample_size_val = 0
            for batch_tmp in val_data.as_numpy_iterator():
                x_tmp = (batch_tmp["item_feature"], batch_tmp["user_feature"])
                y_tmp = batch_tmp["label"]
                pred_tmp = model.predict(x_tmp)
                y_pred_val.append(pred_tmp)
                y_true_val.append(y_tmp)
                loss_val += np.sum(model.loss_fn(y_tmp, pred_tmp))
                sample_size_val += x_tmp[0].shape[0]
            pred = np.concatenate(y_pred_val)
            y_val = np.concatenate(y_true_val)

            loss_val /= sample_size_val
            auc = roc_auc_score(y_val, pred)
            # loss_val = model.evaluate(val_data_generator(val_data), verbose=True)
            line = f"Epoch {epoch}, loss on val set is {round(loss_val, 5)}, auc is {round(auc, 5)}"
            print(line)
            fp.write(line+"\n")
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
    fp.close()


def main():
    train(k=100)


if __name__ == "__main__":
    main()
