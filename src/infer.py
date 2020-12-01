# !/usr/bin/python
# -*- coding:utf-8 -*-
"""
Created on 25 Nov, 2020

Author: woshihaozhaojun@sina.com
"""
import os
import sys
import json
import gzip
import faiss
import numpy as np
from tqdm import tqdm
import tensorflow as tf
src_path = os.path.abspath("..")
print(src_path)
sys.path.append(src_path)
from src.conf_loader import (
    n_layer, emb_dim,
    user_max_len, item_max_len,
    MODEL_DIR, indexing_dir
)


# filepath = os.path.join(MODEL_DIR, f"{user_max_len}_{item_max_len}_{emb_dim}_{batch_size}_{n_epoch}")


class LoadedModel(object):
    def __init__(self, n_layer=n_layer, item_max_len=item_max_len, user_max_len=user_max_len, emb_dim=emb_dim):
        self.n_layer = n_layer
        self.item_max_len = item_max_len
        self.user_max_len = user_max_len
        self.emb_dim = emb_dim
        self.model = None
        self.user_embedding = None
        self.item_embedding = None
        self.mha_user = None
        self.mha_item = None
        self.cid_list = []
        self.cid_vec_list = []

        self.metric = faiss.METRIC_INNER_PRODUCT
        self.index = None
        self.topK = None

    def load_model(self, filepath):
        self.model = tf.keras.models.load_model(filepath, compile=False)
        self.user_embedding = self.model.user_embedding
        self.item_embedding = self.model.item_embedding
        self.mha_user = self.model.mha_user
        self.mha_item = self.model.mha_item

    def calc_item_vec(self, item_inputs):
        """ 基于补齐的物品序列来计算双塔中的物品

        经过以下步骤:
            1. 嵌入
            2. 两层mha
            3. max pooling
            4. L-2归一化
        """
        item_sequence_embeddings = self.item_embedding(item_inputs)
        for i in range(self.n_layer):
            item_sequence_embeddings = self.mha_item(item_sequence_embeddings)
        item_outputs_max = tf.nn.max_pool(
            item_sequence_embeddings,
            [1, self.item_max_len, 1],
            [1 for _ in range(len(item_sequence_embeddings.shape))],
            padding="VALID")
        item_normalized = tf.nn.l2_normalize(
            item_outputs_max, axis=2)
        return item_normalized

    def calc_user_vec(self, user_inputs):
        """ 基于补齐的用户序列来计算双塔中的用户

        经过以下步骤:
            1. 嵌入
            2. 两层mha
            3. max pooling
            4. L-2归一化
        """
        user_sequence_embeddings = self.user_embedding(user_inputs)
        for i in range(self.n_layer):
            user_sequence_embeddings = self.mha_user(user_sequence_embeddings)
        user_outputs_max = tf.nn.max_pool(
            user_sequence_embeddings,
            [1, self.user_max_len, 1],
            [1 for _ in range(len(user_sequence_embeddings.shape))],
            padding="VALID")
        user_normalized = tf.nn.l2_normalize(
            user_outputs_max, axis=2)
        return user_normalized

    def pad_item_seq(self, vids):
        """将栏目中视频idx列表补齐

        如果不够,则用0补齐;
        如果超长则把前面的去掉.

        :param vids, iterable(int), 视频序号列表
        :return np.array, of shape [1, self.item_max_len]
        """
        if len(vids) < self.item_max_len:
            vids += [0 for _ in range(self.item_max_len - len(vids))]
        return np.expand_dims(np.array(vids[-self.item_max_len:]), axis=0)

    def pad_user_seq(self, click_seq):
        """将用户点击的视频idx列表补齐

        如果不够,则用0补齐;
        如果超长则把前面的去掉.

        :param click_seq, iterable(int), 点击的视频序号列表
        :return np.array, of shape [1, self.user_max_len]
        """
        if len(click_seq) < self.item_max_len:
            click_seq += [0 for _ in range(self.user_max_len - len(click_seq))]
        return np.expand_dims(np.array(click_seq[-self.user_max_len:]), axis=0)

    def get_col_vec(self, item_seq):
        """根据原始的代表视频的序号列表获得双塔模型中的物品向量

        :param item_seq, iterable(int)
        :return tf.Tensor, of shape [1, 32]
        """
        item_seq = self.pad_item_seq(item_seq)
        item_normalized = self.calc_item_vec(item_seq)
        return tf.reshape(item_normalized, shape=[1, -1])

    def get_user_vec(self, user_seq):
        """根据用户原始的点击的序号列表获得双塔模型的用户向量

        :param user_seq, iterable(int)
        :return tf.Tensor, of shape [1, 32]
        """
        user_seq = self.pad_user_seq(user_seq)
        user_normalized = self.calc_user_vec(user_seq)
        return tf.reshape(user_normalized, shape=[1, -1])

    def get_all_col_tensor(self, cid2vidx_file):
        """
        """
        with open(cid2vidx_file, "r") as fp:
            cid2vidx = json.load(fp)

        idx = 0
        self.cid_list, self.cid_vec_list = [], []
        for cid, vidx in cid2vidx.items():
            self.cid_list.append(str(cid))
            tensor = self.get_col_vec(vidx)
            assert tensor.shape[-1] == self.emb_dim
            self.cid_vec_list.append(tensor)
            idx += 1
        print(f"succeed in loading normalized vectors of {idx} items, by which I mean target columns")

    # # TODO: 批量读取用户的行为序列来排序
    # def save_rec(self, input_file, output_file, user_block_size=10 ** 5):
    #     with gzip.open(output_file, "wb") as wf:
    #         content = "\t".join(
    #             [dnum, period, ",".join(rec_vids)]) + "\r\n"
    #         content = content.encode()
    #         wf.write(content)
    #     return

    def init_faiss(self, xb=None, nlist=2):
        """ 初始化FAISS,把物品向量存起来

        :param xb, 数据库中的向量
        :param nlist, 聚类的数量
        """
        # 量化器索引
        quantizer = faiss.IndexFlatIP(self.emb_dim)

        # 默认用內积faiss.METRIC_INNER_PRODUCT
        self.index = faiss.IndexIVFFlat(quantizer, self.emb_dim, nlist, faiss.METRIC_INNER_PRODUCT)

        if xb is None:
            xb = np.concatenate(self.cid_vec_list, axis=0)
        assert not self.index.is_trained
        self.index.train(xb)
        assert self.index.is_trained
        self.index.add(xb)
        self.index.nprobe = nlist

    def sort_by_faiss(self, queries):
        """通过FAISS来获取queries中每个向量的KNN

        :returns queries的最近的向量的距离和序号
        """
        if self.topK is None:
            self.topK = len(self.cid_list)
        return self.index.search(queries, self.topK)


def main():
    model = LoadedModel()
    model.load_model(filepath=MODEL_DIR)
    cid2vidx_file = os.path.join(indexing_dir, "cid2vidx.json")
    model.get_all_col_tensor(cid2vidx_file)

    print("\n------FAISS inner product------")
    model.init_faiss()
    user_seq = [i for i in range(model.user_max_len - 1)]
    user_vec = model.get_user_vec(user_seq)
    user_vec = np.array(user_vec)
    dist, ind = model.sort_by_faiss(queries=user_vec)
    print("ind=", ind[0][:10])
    print("dist=", dist[0][:10])

    print("------numpy matmul------")
    col_mat = np.concatenate(model.cid_vec_list, axis=0)
    user_mat = user_vec
    sim = np.matmul(user_mat, np.transpose(col_mat))
    for i in range(sim.shape[0]):
        print("ind=", np.argsort(-sim[i, :], axis=-1)[:10])
        print("dist=", sorted(sim[i, :], reverse=True)[:10])


if __name__ == "__main__":
    main()




