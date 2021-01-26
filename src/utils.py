# !/usr/bin/python
# -*- coding:utf-8 -*-
"""
Created on 25 Nov, 2020

Updated on 26 Jan, 2021

Author: woshihaozhaojun@sina.com
"""
import cProfile
import pstats
import time
from tensorflow.keras.callbacks import Callback


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
    def __init__(self, display, time_flag=False):
        self.seen = 0
        self.display = display
        self.start_time = time.time()
        self.time_flag = time_flag

    def on_batch_end(self, batch, logs=None):
        if batch % self.display == 0:
            line = f"""For batch {batch}, loss is {logs["loss"]}"""
            if self.time_flag:
                line += f"time elapsed {(time.time() - self.start_time) / 60} mins"
            print(line)