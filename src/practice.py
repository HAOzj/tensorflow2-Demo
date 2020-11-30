"""
Created on Friday 6th Nov 2020

@author : Hao Zhaojun
"""
import tensorflow as tf
from tqdm import tqdm
emb_dim = 10

lut = tf.Variable(tf.random.uniform([20, 10], maxval=1))
for i in range(20):
    lut[i, 0].assign(i)
item_seq_index = tf.reshape(tf.constant([i for i in range(20)]), [5, -1])

item_seq_emb = tf.nn.embedding_lookup(lut, item_seq_index)


dnn = tf.keras.layers.Dense(4, activation=tf.nn.leaky_relu)
output = dnn(item_seq_emb)
apres_split = tf.split(output, 2, axis=2)
apres_concat = tf.concat(apres_split, axis=0)
apres_matmul = tf.matmul(apres_concat, tf.transpose(apres_concat, [0, 2, 1]))
apres_pool = tf.nn.max_pool(apres_matmul, [1, 4, 1], [1 for _ in range(len(apres_matmul.shape))], padding="VALID")
apres_norm = tf.nn.l2_normalize(apres_pool, axis=2)


print(item_seq_index.shape)
print(item_seq_emb.shape)
print(output.shape)
print(apres_split[0].shape)
print(apres_concat.shape)
print(apres_matmul.shape)
print(apres_pool.shape)
print(apres_pool[0, :])
print(apres_norm[0, :])


class Transformer(tf.keras.models.Model):
    def __init__(self, target_vocabular_size):
        super(Transformer, self).__init__()
        self.linear_layer = tf.keras.layers.Dense(target_vocabular_size)

    def call(self, input_a, input_b):
        input_a = self.linear_layer(input_a)
        input_b = self.linear_layer(input_b)
        input_a = tf.expand_dims(input_a, 1)
        input_b = tf.expand_dims(input_b, 1)
        output = tf.matmul(input_a, input_b, transpose_b=True)

        return output


import numpy as np
x_a = np.random.uniform((5, 20))
x_b = np.random.uniform((5, 20))
y = np.random.randint(0, 2, size=(5, 1))

transformer = Transformer(target_vocabular_size=10)
transformer.fit(x=[x_a, x_b], y=y)
print(transformer.summary())
train_step_signature = [
    tf.TensorSpec(shape=(None, 20), dtype=tf.float32),
    tf.TensorSpec(shape=(None, 20), dtype=tf.float32),
    tf.TensorSpec(shape=(None, 1), dtype=tf.int8)
]

optimizer = tf.keras.optimizers.Adam(learning_rate=10**-4, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
training_loss = tf.keras.metrics.Mean(name='training_loss')
training_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='training_accuracy')


@tf.function(input_signature=train_step_signature)
def train_step(input_a, input_b, target):
    # Run training step
    with tf.GradientTape() as tape:
        output = transformer(input_a, input_b)
    total_loss = tf.keras.metrics.binary_accuracy(y_true=target, y_pred=output)
    gradients = tape.gradient(total_loss, transformer.trainable_variables)
    optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))
    training_loss(total_loss)
    training_accuracy(target, output)


# for epoch in tqdm(range(20)):
#     training_loss.reset_states()
#     training_accuracy.reset_states()
#
#     for (batch, (input_language, target_language)) in enumerate(data_container.train_data):
#         train_step(input_language, target_language)
#
#     print('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch, training_loss.result(), training_accuracy.result()))

