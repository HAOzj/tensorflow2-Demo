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
sys.path.append(src_path)
from src.conf_loader import (
    n_layer, emb_dim, user_max_len, item_max_len,
    batch_size, n_epoch,
    MODEL_DIR, indexing_dir,
    USER_PROFILE_FILE, OUTPUT_FILE,
    COL_INFO_FILE, LAST_DIGITS
)
from src.utils import print_run_time
test_flag = False


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

        self.cid2vidx_file = None

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

        # [1, self.user_max_len, self.emb_dim]
        user_sequence_embeddings = self.user_embedding(user_inputs)
        for i in range(self.n_layer):
            user_sequence_embeddings = self.mha_user(user_sequence_embeddings)

        # [1, 1, self.emb_dim]
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

    def _write_cid2vidx_file(self, vid2idx_file, col_info_file):
        if vid2idx_file is None or col_info_file is None:
            sys.exit("vid2idx_file is None or col_info_file is None")

        with open(vid2idx_file, "r") as fp:
            vid2idx = json.load(fp)

        cid_list = []
        cid2vidx = dict()
        with open(col_info_file, "r") as fp:
            column_cnt = 1
            for line in fp:
                column_cnt += 1
                info = json.loads(line.strip("\r\t\n"))
                column_id = str(info.get("column_id", "")).strip(" ")

                if column_id == "" or column_id is None:
                    print("------cid missed", line)
                    continue
                cid_list.append(column_id)

                """混合频道的格式为
                {
                    "tab_name": str, "column_id": str, "source_id": str, "column_name": str, "client_type": "ct1,ct2",
                    "vid_info": [<channel_id>\f<meizi_count>\f<cid1>\t<cid2>\t<cid3>,  ]
                }
                """
                vid_info = info.get("vid_info", [])
                if not vid_info:
                    print("------", column_id, "vid_info missed")
                    continue
                vids = []
                for item in vid_info:
                    tuple3 = item.split("\f")
                    if len(tuple3) != 3:
                        print("------tuples3 missed", column_id)
                        print("------", tuple3)
                        continue
                    vids += tuple3[2].split("\t")
                vidx = [vid2idx[vid] for vid in vids if vid in vid2idx]
                if column_id in cid2vidx:
                    current = cid2vidx[column_id]
                    nouveau = [vind for vind in vidx if vind not in current]
                    vidx = current + nouveau
                if vidx:
                    cid2vidx[column_id] = vidx
                else:
                    print("------", column_id, "lacking valid cids", vids)
        print(f"In total, {column_cnt} columns found, among which {len(set(cid_list))} are distinct")

        if cid2vidx:
            with open(self.cid2vidx_file, "w") as fp:
                json.dump(cid2vidx, fp)
            print(f"------succeed in writing cid2vidx, containing {len(cid2vidx)} columns")

    def get_all_col_tensor(self, cid2vidx_file, vid2idx_file=None, col_info_file=None):
        """ 计算得到每个栏目的向量

        :param cid2vidx_file, path, 保存着 栏目 -> 对应的视频的序号列表
        :param vid2idx_file, path, 保存着 视频id -> 序号 的
        :param col_info_file, path, 保存着 栏目 -> 对应的视频id列表
        """
        if self.cid2vidx_file is None:
            self.cid2vidx_file = cid2vidx_file
            self._write_cid2vidx_file(vid2idx_file, col_info_file)

        with open(self.cid2vidx_file, "r") as fp:
            cid2vidx = json.load(fp)

        idx = 0
        self.cid_list, self.cid_vec_list = [], []
        for cid, vidx in cid2vidx.items():
            self.cid_list.append(str(cid))
            tensor = self.get_col_vec(vidx)
            assert tensor.shape[-1] == self.emb_dim
            self.cid_vec_list.append(tensor)
            idx += 1
        print(f"------succeed in loading normalized vectors of {idx} items, by which I mean target columns")

    def init_faiss(self, xb=None, nlist=2):
        """ 初始化FAISS,把物品向量存起来

        :param xb, 数据库中的向量
        :param nlist, 聚类的数量
        """
        if not self.cid_vec_list:
            print("please call `get_all_col_tensor method` first")
            return

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
        if self.index is None:
            print("please call `init_faiss` first")
            return
        if self.topK is None:
            self.topK = len(self.cid_list)
        return self.index.search(queries, self.topK)

    @staticmethod
    def write2file_from_sim(sim, dnum_list, cid_list, wf):
        """ 根据相似度矩阵,把用户的推荐序列写入文件

        :param sim, np.ndarray, 用户喝物品的相似矩阵,每行代表一个用户
        :param dnum_list, iterable(str), 设备序列号的列表
        :param cid_list, iterable(str), 栏目id的列表
        :param wf, 输出文件的句柄
        """
        try:
            assert sim.shape[1] == len(cid_list)
        except AssertionError:
            print(f"HZJ info: self.cid_list is of size {len(cid_list)} while sim is of shape {sim.shape}")
            return

        for i in range(sim.shape[0]):
            ind = np.argsort(-sim[i, :], axis=-1)
            rec_vids = [cid_list[idx] for idx in ind]
            dnum = dnum_list[i]
            content = "\t".join(
                [dnum, "day", "1", ",".join(rec_vids)]) + "\r\n"
            content = content.encode()
            wf.write(content)

    @print_run_time
    def save_rec(self, input_file, output_file, user_block_size=10 ** 3):
        if not self.cid_vec_list:
            print("please call `get_all_col_tensor method` first")
            return
        col_mat = np.transpose(np.concatenate(self.cid_vec_list, axis=0))
        user_cnt = 0
        rest_flag = True
        dnum_list = []
        uvec_list = []
        with gzip.open(output_file, "wb") as wf:
            with gzip.open(input_file, "rb") as fp:
                for line in fp:
                    line = line.decode().strip("\r\t\n")
                    info = json.loads(line)
                    dnum = str(info.get("dnum", ""))

                    if len(dnum) < 1:
                        continue

                    if dnum[-1] not in LAST_DIGITS:
                        continue

                    click_seq = info.get("click_seq", [])
                    click_seq = list(map(int, click_seq))
                    if len(click_seq) < 1:
                        continue
                    rest_flag = True
                    user_cnt += 1

                    dnum_list.append(dnum)
                    user_vec = self.get_user_vec(click_seq)
                    user_vec = np.array(user_vec)
                    uvec_list.append(user_vec)

                    if test_flag and user_cnt >= 100:
                        break

                    if user_cnt & user_block_size == 0:
                        rest_flag = False
                        user_mat = np.concatenate(uvec_list, axis=0)
                        sim = np.matmul(user_mat, col_mat)
                        self.write2file_from_sim(sim, dnum_list, self.cid_list, wf)
                        dnum_list = []
                        uvec_list = []

            print("HZJ info: user_cnt = ", user_cnt)
            if rest_flag and uvec_list:
                user_mat = np.concatenate(uvec_list, axis=0)
                sim = np.matmul(user_mat, col_mat)
                self.write2file_from_sim(sim, dnum_list, self.cid_list, wf)


def main():
    filepath = os.path.join(MODEL_DIR, f"{user_max_len}_{item_max_len}_{emb_dim}_{batch_size}_{n_epoch}")
    model = LoadedModel()
    model.load_model(filepath=filepath)
    cid2vidx_file = os.path.join(indexing_dir, "cid2vidx.json")
    vid2idx_file = os.path.join(indexing_dir, "vid2idx.json")
    model.get_all_col_tensor(cid2vidx_file, vid2idx_file, col_info_file=COL_INFO_FILE)
    model.save_rec(input_file=USER_PROFILE_FILE, output_file=OUTPUT_FILE)
    # print("\n------FAISS inner product------")
    # model.init_faiss()
    # user_seq = [i for i in range(model.user_max_len - 1)]
    # user_vec = model.get_user_vec(user_seq)
    # user_vec = np.array(user_vec)
    # dist, ind = model.sort_by_faiss(queries=user_vec)
    # print("ind=", ind[0][:10])
    # print("dist=", dist[0][:10])
    #
    # print("------numpy matmul------")
    # col_mat = np.concatenate(model.cid_vec_list, axis=0)
    # user_mat = user_vec
    # sim = np.matmul(user_mat, np.transpose(col_mat))
    # for i in range(sim.shape[0]):
    #     print("ind=", np.argsort(-sim[i, :], axis=-1)[:10])
    #     print("dist=", sorted(sim[i, :], reverse=True)[:10])


if __name__ == "__main__":
    main()




