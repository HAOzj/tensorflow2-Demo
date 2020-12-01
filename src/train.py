# !/usr/bin/python
# -*- coding:utf-8 -*-
"""
Created on 09 Nov, 2020

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
 MODEL_DIR, tfrecord_dir,
 user_max_len, item_max_len,
 n_epoch, batch_size, buffer_size, test_ratio, val_ratio
)
from src.utils import print_run_time


class Break(Exception):
    pass


@print_run_time
def train(k=100, log_file="loss_on_val.txt"):
    train_data, test_data, val_data = load_data(
        tfrecord_dir, user_max_len, item_max_len, batch_size=batch_size,
        test_ratio=test_ratio, val_ratio=val_ratio)

    # 是否用generator的方式返回发展集数据
    # 如果False的话需要把发展集加载到内存中
    gen_flag = True

    if not gen_flag:
        # 把发展集放进一个np数组,用于计算loss和auc等
        x_val, y_val = [], []
        for mini_batch in val_data.as_numpy_iterator():
            x_train = np.concatenate((mini_batch["item_feature"], mini_batch["user_feature"]), axis=-1)
            y_train = mini_batch["label"]
            x_val.append(x_train)
            y_val.append(y_train)
        x_val = np.concatenate(x_val, axis=0)
        x_val = np.split(x_val, 2, axis=-1)
        y_val = np.concatenate(y_val, axis=0)

    model = Model()
    model.compile(
        optimizer=model.optimizer,
        loss=model.loss_fn
    )
    model.build(input_shape=[(None, item_max_len), (None, user_max_len)])

    # def val_data_generator(val_data):
    #     """生成发展集的迭代器,用于喂入数据
    #     这个配合model.evaluate使用,无法计算auc
    #     """
    #     for mini_batch in val_data.as_numpy_iterator():
    #         x_train = (mini_batch["item_feature"], mini_batch["user_feature"])
    #         y_train = mini_batch["label"]
    #         yield x_train, y_train

    batch_num = 0
    losses_val = [1 for _ in range(2)]

    fp = open(log_file, "w")
    try:
        for epoch in range(n_epoch):
            for mini_batch in train_data.shuffle(buffer_size).as_numpy_iterator():
                x_train = (mini_batch["item_feature"], mini_batch["user_feature"])
                y_train = mini_batch["label"]
                batch_num += 1
                model.fit(x_train, y_train)

                if batch_num % k == 0:
                    if gen_flag:
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
                    else:
                        pred = model.predict(x_val)
                        loss_val = np.sum(model.loss_fn(y_val, pred))
                        sample_size_val = y_val.shape[0]
                    loss_val /= sample_size_val
                    auc = roc_auc_score(y_val, pred)
                    # loss_val = model.evaluate(val_data_generator(val_data), verbose=True)
                    line = f"{batch_num} batch, loss on val set is {round(loss_val, 5)}, auc is {round(auc, 5)}"
                    print(line)
                    fp.write(line+"\n")
                    losses_val.append(loss_val)

                    if losses_val[-1] >= losses_val[-2] >= losses_val[-3]:
                        print(losses_val)
                        raise Break("break")
    except Break as e:
        print(e)
    fp.close()
    model.save(
        filepath=os.path.join(MODEL_DIR, f"{user_max_len}_{item_max_len}_{model.emb_dim}_{batch_size}_{n_epoch}"),
        save_format="tf"
    )


def main():
    train(k=100)


if __name__ == "__main__":
    main()
