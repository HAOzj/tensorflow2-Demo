# !/usr/bin/python
# -*- coding:utf-8 -*-
"""
Created on 24 Nov, 2020

Author : woshihaozhaojun@sina.com
"""
import tensorflow as tf


class MultiHeadAttention(tf.keras.layers.Layer):
    """ def multi head attention layer

    q, k, v分别经过Wq, Wk, Wv生成q', k', v'
    q' * k' 生成 w, w / sqrt(q'.shape[1])标准化生成w'
    w' * v' 生成 z, z经过Wz生成 z'
    z' add v做一个残差连接,然后经过LRelu,最后LN
    """

    def __init__(
            self,
            scope_name,
            num_units=8,
            num_heads=1,
            embed_dim=32,
            has_residual=True,
            dropout_keep_prob=1.0):
        super(MultiHeadAttention, self).__init__()
        assert num_units % num_heads == 0
        assert scope_name in ["user", "item"]
        self.num_heads = num_heads
        self.units = num_units
        self.embed_dim = embed_dim
        self.dropout_keep_prob = dropout_keep_prob

        self.Wq = tf.keras.layers.Dense(
            units=self.units, activation=tf.nn.leaky_relu, name=f"{scope_name}_Wq")
        self.Wk = tf.keras.layers.Dense(
            units=self.units, activation=tf.nn.leaky_relu, name=f"{scope_name}_Wk")
        self.Wv = tf.keras.layers.Dense(
            units=self.units, activation=tf.nn.leaky_relu, name=f"{scope_name}_Wv")

        self.has_residual = has_residual
        self.Wz = tf.keras.layers.Dense(embed_dim, name=f"{scope_name}_Wz")
        self.ln = tf.keras.layers.LayerNormalization(
            beta_initializer="zeros",
            gamma_initializer="ones",
            name=f"{scope_name}_ln")

    def call(self, inputs):
        """多头注意力模型

        :param queries: of shape [batch_size, max_length, emb_dim]
        :param keys_:  of shape [batch_size, max_length, emb_dim]
        :param values: of shape [batch_size, max_length, emb_dim]
        :return:
        """
        assert inputs.get_shape().as_list()[-1] == self.embed_dim
        # Linear projections
        # [batch_size, max_length, emb_dim] -> [batch_size, max_length, num_units]
        Q = self.Wq(inputs)
        K = self.Wk(inputs)
        V = self.Wv(inputs)

        # Split and concat
        # [batch_size, max_length, num_units] -> [batch_size * num_heads, max_length, num_units / num_heads]
        Q_ = tf.concat(tf.split(Q, self.num_heads, axis=2), axis=0)
        K_ = tf.concat(tf.split(K, self.num_heads, axis=2), axis=0)
        V_ = tf.concat(tf.split(V, self.num_heads, axis=2), axis=0)

        # Multiplication
        # [batch_size * num_heads, max_length, max_length]
        weights = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))

        # Scale
        weights = weights / (K_.get_shape().as_list()[-1] ** 0.5)

        # 转化为概率向量
        weights = tf.nn.softmax(weights)

        # Dropouts
        if 0 < self.dropout_keep_prob < 1:
            weights = tf.keras.layers.AlphaDropout(
                rate=1 - self.dropout_keep_prob)(weights)

        # Weighted sum
        # [batch_size * num_heads, max_length, num_units / num_heads]
        outputs = tf.matmul(weights, V_)

        # Restore shape to [batch_size, max_length, num_units]
        z = tf.concat(tf.split(outputs, self.num_heads, axis=0), axis=2)

        # Restore shape to [batch_size, max_length, embed_dim]
        z = self.Wz(z)

        # Residual connection
        if self.has_residual:
            z += inputs

        z = tf.nn.leaky_relu(z)

        # Normalize
        z = self.ln(z)

        return z