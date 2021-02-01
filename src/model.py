# !/usr/bin/python
# -*- coding:utf-8 -*-
"""
Created on 09 Nov, 2020

Author : woshihaozhaojun@sina.com
"""
import os
import sys
src_path = os.path.abspath("..")
print(src_path)
sys.path.append(src_path)
from src.conf_loader import (
    MODEL_DIR, n_epoch, emb_dim, n_layer,
    item_max_len, user_max_len,
    batch_size, lr, l2_reg,
    vocab_size
)
from src.component import MultiHeadAttention
import numpy as np
import tensorflow as tf


class BST_DSSM(tf.keras.Model):
    """define BST+DSSM model stucture

    用subclass的方法来定义模型
    """

    def __init__(self,
                 item_embedding=None, user_embedding=None,
                 emb_dim=emb_dim,
                 vocab_size=vocab_size,
                 item_max_len=item_max_len, user_max_len=user_max_len,
                 epoch=10, batch_size=batch_size, n_layers=n_layer,
                 learning_rate=lr, optimizer_type="adam",
                 random_seed=2019,
                 l2_reg=l2_reg, has_residual=True):
        """
        initial model related parms and tensors
        """
        super(BST_DSSM, self).__init__()
        self.emb_dim = emb_dim

        self.l2_reg = l2_reg

        self.epoch = epoch
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.optimizer_type = optimizer_type

        self.blocks = n_layers

        self.random_seed = random_seed

        self.vocab_size = vocab_size
        self.item_max_len = item_max_len
        self.user_max_len = user_max_len
        self.has_residual = has_residual

        self.mha_user = MultiHeadAttention(scope_name="user", embed_dim=self.emb_dim)
        self.mha_item = MultiHeadAttention(scope_name="item", embed_dim=self.emb_dim)

        # optimizer
        if self.optimizer_type == "adam":
            self.optimizer = tf.keras.optimizers.Adam(
                learning_rate=self.learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
        elif self.optimizer_type == "adagrad":
            self.optimizer = tf.keras.optimizers.Adagrad(
                learning_rate=self.learning_rate,
                initial_accumulator_value=1e-8)
        elif self.optimizer_type == "gd":
            self.optimizer = tf.keras.optimizers.SGD(
                learning_rate=self.learning_rate)
        elif self.optimizer_type == "momentum":
            self.optimizer = tf.keras.optimizers.SGD(
                learning_rate=self.learning_rate, momentum=0.95)

        self.user_embedding = tf.keras.layers.Embedding(input_dim=self.vocab_size, output_dim=self.emb_dim)
        self.item_embedding = tf.keras.layers.Embedding(input_dim=self.vocab_size, output_dim=self.emb_dim)

    @tf.function
    def call(self, inputs, training=True):
        # multiple inputs
        item_inputs = inputs[0]
        user_inputs = inputs[1]

        # 用户和物品各用一个lut,类似DSSM中的双塔
        # 维度变成[batch_size, max_length, emb_dim]
        item_sequence_embeddings = self.item_embedding(item_inputs)
        user_sequence_embeddings = self.user_embedding(user_inputs)

        # mask
        item_mask = tf.where(
            item_inputs == 0,
            x=tf.ones_like(item_inputs, dtype=tf.float32),
            y=tf.zeros_like(item_inputs, dtype=tf.float32))
        item_mask = tf.expand_dims(item_mask, axis=-1)
        user_mask = tf.where(
            user_inputs == 0,
            x=tf.ones_like(user_inputs, dtype=tf.float32),
            y=tf.zeros_like(user_inputs, dtype=tf.float32))
        user_mask = tf.expand_dims(user_mask, axis=-1)

        # 维度变成[batch_size, max_length, 16]
        for i in range(self.blocks):
            item_sequence_embeddings = self.mha_item(item_sequence_embeddings, src_mask=item_mask)
            user_sequence_embeddings = self.mha_user(user_sequence_embeddings, src_mask=user_mask)

        # 最大池化层, 维度变成[batch_size, 1, 16]
        item_outputs_max = tf.nn.max_pool(
            item_sequence_embeddings,
            [1, self.item_max_len, 1],
            [1 for _ in range(len(item_sequence_embeddings.shape))],
            padding="VALID")
        user_outputs_max = tf.nn.max_pool(
            user_sequence_embeddings,
            [1, self.user_max_len, 1],
            [1 for _ in range(len(user_sequence_embeddings.shape))],
            padding="VALID")

        # 向量归一化用于计算cosine相似度
        item_normalized = tf.nn.l2_normalize(
            item_outputs_max, axis=2)
        user_normalized = tf.nn.l2_normalize(
            user_outputs_max, axis=2)

        # cosine相似度
        outputs = tf.matmul(
            item_normalized,
            user_normalized,
            transpose_b=True)
        return tf.reshape(outputs, [-1, 1])

    def loss_fn(self, target, output):
        cross_entropy = tf.keras.backend.binary_crossentropy(
            target, output, from_logits=False
        )
        if self.l2_reg > 0:
            _regularizer = tf.keras.regularizers.l2(self.l2_reg)
            cross_entropy += _regularizer(self.user_embedding)
            cross_entropy += _regularizer(self.item_embedding)
        return cross_entropy

    def focal_loss(self, target, output):
        target = tf.reshape(target, [-1, 1])
        y = tf.multiply(output, target) + tf.multiply(1 - output, 1 - target)
        loss = tf.pow(1. - y, self.gamma) * tf.math.log(y + self.epsilon)
        return - tf.reduce_mean(loss)


def debug():
    x_train = [
        np.random.randint(low=0, high=20, size=(5, item_max_len)),
        np.random.randint(low=0, high=20, size=(5, user_max_len))]

    y_train = np.random.randint(low=0, high=2, size=5).astype(dtype=float)

    model = BST_DSSM()
    model.compile(
                optimizer=model.optimizer,
                loss=model.loss_fn
        # ,
                # metrics=[
                #     tf.keras.metrics.AUC,
                #     tf.keras.metrics.binary_accuracy]
            )
    model.fit(x_train, y_train, epochs=n_epoch)
    model.summary()
    # model.save(MODEL_DIR, save_format="tf")


if __name__ == "__main__":
    debug()


