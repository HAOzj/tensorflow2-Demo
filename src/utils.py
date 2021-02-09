# !/usr/bin/python
# -*- coding:utf-8 -*-
"""
Created on 25 Nov, 2020

Updated on 1 Feb, 2021

Author: woshihaozhaojun@sina.com
"""
import cProfile
import pstats
import time
import numpy as np
from tensorflow.keras.callbacks import Callback
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, roc_auc_score, confusion_matrix
)


def do_cprofile(filename):
    """
    Decorator for function profiling.
    """
    def wrapper(func):
        def profiled_func(*args, **kwargs):
            profile = cProfile.Profile()
            profile.enable()
            result = func(*args, **kwargs)
            profile.disable()
            # Sort stat by internal time.
            sortby = "tottime"
            ps = pstats.Stats(profile).sort_stats(sortby)
            ps.dump_stats(filename)
            return result
        return profiled_func
    return wrapper


def print_run_time(func):
    """ 计算时间函数
    """

    def wrapper(*args, **kw):
        local_time = time.time()
        res = func(*args, **kw)
        print('Current function : {function}, time used : {temps}'.format(
            function=func.__name__, temps=time.time() - local_time)
        )
        return res

    return wrapper


class NBatchLogger(Callback):
    def __init__(self, display, time_flag=False, val_x=None, val_y=None):
        self.seen = 0
        self.display = display
        self.start_time = time.time()
        self.time_flag = time_flag
        self.val_x = val_x
        self.val_y = val_y

    def on_batch_end(self, batch, logs=None):
        if batch % self.display == 0:
            line = f"""For batch {batch}, loss is {round(logs["loss"], 4)}"""
            if self.time_flag:
                line += f", time elapsed {(time.time() - self.start_time) / 60} mins"
            print(line)

    def on_epoch_end(self, epoch, logs=None):
        if self.val_x and self.val_y:
            model = self.model
            y_pred_val, y_true_val = [], []
            for x_tmp, y_tmp in zip(self.val_x.as_numpy_iterator(), self.val_y.as_numpy_iterator()):
                pred_tmp = model.predict(x_tmp)
                y_pred_val.append(pred_tmp)
                y_true_val.append(y_tmp)

            y_pred = np.concatenate(y_pred_val).astype(dtype=float)
            y_true = np.concatenate(y_true_val).astype(dtype=float)
            roc_auc_val = roc_auc_score(y_true, y_pred)

            # 把预测的连续值转化为类别
            y_pred = np.where(y_pred > 0.5, np.ones_like(y_pred), np.zeros_like(y_pred))

            recall = recall_score(y_true=y_true, y_pred=y_pred)
            precision = precision_score(y_true=y_true, y_pred=y_pred)
            accuracy = accuracy_score(y_true=y_true, y_pred=y_pred)
            line = f"""For epoch {epoch}, auc is {round(roc_auc_val, 4)}, recall is {round(recall, 4)}, 
                        precision is {round(precision, 4)}, accuracy is {round(accuracy, 4)}"""
            if self.time_flag:
                line += f", time elapsed {(time.time() - self.start_time) / 60} mins"
            print(line)


def get_all_metrics(model, epoch, val_x, val_y, start_time, loss_fn):
    """每个epoch结束后在发展集上预测,得到一些指标

    :param model: tf.keras.Model, epoch训练后的模型
    :param epoch: int, 轮数
    :param val_x: tf.data.Dataset, 发展集的输入, 和val_y一样的sample_size
    :param val_y: tf.data.Dataset, 发展集的标签
    :param start_time: time.time, 开始时间
    :param loss_fn: 损失函数
    :return: 模型在发展集上的损失
    """
    y_pred_val, y_true_val = [], []
    loss_val = 0
    sample_size_val = 0

    for x_tmp, y_tmp in zip(val_x.as_numpy_iterator(), val_y.as_numpy_iterator()):
        pred_tmp = model.predict(x_tmp)
        y_pred_val.append(pred_tmp)
        y_true_val.append(y_tmp)

        loss_tmp = loss_fn(y_tmp, pred_tmp)
        loss_val += np.sum(loss_tmp)
        sample_size_val += x_tmp[0].shape[0]

    # 计算损失
    loss_val /= sample_size_val

    # 计算auc
    y_pred = np.concatenate(y_pred_val).astype(dtype=float)
    y_true = np.concatenate(y_true_val).astype(dtype=float)
    roc_auc_val = roc_auc_score(y_true, y_pred)

    # 转化预测概率为类别
    y_pred = np.where(y_pred > 0.5, np.ones_like(y_pred), np.zeros_like(y_pred))

    # 计算混淆矩阵相关的
    recall = recall_score(y_true=y_true, y_pred=y_pred)
    precision = precision_score(y_true=y_true, y_pred=y_pred)
    accuracy = accuracy_score(y_true=y_true, y_pred=y_pred)
    line = f"""For epoch {epoch}, on val set loss is {round(loss_val, 5)}, auc is {round(roc_auc_val, 4)}, 
        recall is {round(recall, 4)}, precision is {round(precision, 4)}, accuracy is {round(accuracy, 4)},
        confusion_matrix is {confusion_matrix(y_true=y_true, y_pred=y_pred)}"""
    line += f", time elapsed {(time.time() - start_time) / 60} mins"
    print("HZJ info: ", line)
    return loss_val
