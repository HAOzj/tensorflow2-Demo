# !/usr/bin/python
# -*- coding:utf-8 -*-
"""
Created on 03 Feb, 2021

Author: woshihaozhaojun@sina.com
"""
import os
import sys
import numpy as np
import tensorflow as tf
from sklearn.metrics import pairwise_distances
src_path = os.path.abspath("..")
sys.path.append(src_path)
from src.conf_loader import (
    item_max_len, emb_dim, user_max_len,
    batch_size, n_epoch, vocab_size,
    MODEL_WEIGHT_DIR
)
from src.model import BST_DSSM as Model
test_flag = False


class LoadedModel(object):
    def __init__(self, emb_dim=emb_dim):
        self.emb_dim = emb_dim
        self.model = None

        self.user_embedding = None
        self.item_embedding = None

    def load_model(self, weight_filepath):

        # 换用load_weights的方式
        self.model = Model(optimizer_type="adam")
        self.model.compile(
            optimizer=self.model.optimizer,
            loss=self.model.loss_fn
        )
        self.model.build(input_shape=[(None, item_max_len), (None, user_max_len)])
        self.model.load_weights(weight_filepath)

        self.user_embedding = self.model.user_embedding
        self.item_embedding = self.model.item_embedding

    @staticmethod
    def convert_to_str(tensor, is_numpy=False, func=lambda x: str(x)):
        if not is_numpy:
            tensor = tensor.numpy()
        return ", ".join([func(x) for x in tensor.tolist()])

    def item_emb(self):
        weights = self.item_embedding.get_weights()[0]

        # cosine相似度
        norm_mat = tf.sqrt(tf.reduce_sum(tf.multiply(weights, weights), axis=1, keepdims=True))
        normed_weights = tf.divide(weights, norm_mat)
        sim_mat = tf.matmul(normed_weights, normed_weights, transpose_b=True)
        with open("item_embedding/cosine_sim_item.txt", "w") as fp:
            for i in range(sim_mat.shape[0]):
                fp.write(self.convert_to_str(sim_mat[i]) + "\n")
        print(sim_mat)
        most_close_mat = np.argsort(-sim_mat, axis=1)[:, :10]
        print(most_close_mat)
        with open("item_embedding/cosine_most_close_item.txt", "w") as fp:
            for i in range(most_close_mat.shape[0]):
                fp.write(self.convert_to_str(most_close_mat[i], is_numpy=True) + "\n")

        most_close_arr = np.reshape(
            np.argsort(-sim_mat, axis=None)[:vocab_size+200], (1, -1)
        )[0, vocab_size: vocab_size+200: 2]
        flattened = np.reshape(sim_mat, (1, -1))
        with open("item_embedding/cosine_most_close_item_among_all.txt", "w") as fp:
            fp.write(self.convert_to_str(
                most_close_arr,
                is_numpy=True,
                func=lambda x: f"{int(x/vocab_size)}_{int(x) % vocab_size}_{round(flattened[0][x], 5)}") + "\n")

        # l2 距离
        dist_mat = pairwise_distances(X=weights)
        with open("item_embedding/l2_dist_item.txt", "w") as fp:
            for i in range(sim_mat.shape[0]):
                fp.write(self.convert_to_str(dist_mat[i], is_numpy=True) + "\n")
        most_close_l2_mat = np.argsort(dist_mat, axis=1)[:, :10]
        with open("item_embedding/l2_most_close_item.txt", "w") as fp:
            for i in range(most_close_mat.shape[0]):
                fp.write(self.convert_to_str(most_close_l2_mat[i], is_numpy=True) + "\n")
        most_close_l2_arr = np.reshape(
            np.argsort(dist_mat, axis=None)[: vocab_size + 200], (1, -1)
        )[0, vocab_size: vocab_size+200: 2]
        flattened_l2 = np.reshape(dist_mat, (1, -1))
        with open("item_embedding/l2_most_close_item_among_all.txt", "w") as fp:
            fp.write(self.convert_to_str(
                most_close_l2_arr,
                is_numpy=True,
                func=lambda x: f"{int(x/vocab_size)}_{int(x) % vocab_size}_{round(flattened_l2[0][x], 5)}") + "\n")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p", "--path",
        action="store",
        default=os.path.join(
            MODEL_WEIGHT_DIR,
            f"{user_max_len}_{item_max_len}_{emb_dim}_{batch_size}_{n_epoch}"),
        help="模型的路径"
    )
    args = parser.parse_args()

    model = LoadedModel()
    model.load_model(weight_filepath=args.path)
    model.item_emb()


if __name__ == "__main__":
    main()




