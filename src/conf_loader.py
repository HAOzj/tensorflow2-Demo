MODEL_DIR = "../model/bst_column"

# 预处理
indexing_dir = "../indexing"
tfrecord_dir = "../tfrecord"
click_seq_dir = "../click_seq_json"
class_ratio = 4  # 负样本对比正样本的倍数

# 模型结构
user_max_len = 30
item_max_len = 30
emb_dim = 32
n_layer = 2
vocab_size = 400000

# 训练超参
n_epoch = 10
buffer_size = 1000
batch_size = 1000
lr = 0.001
l2_reg = 0.0
test_ratio = 0.15
val_ratio = 0.15
