# 项目 
这个项目是用于栏目智能排序的BST+DSSM模型的数据生成,预处理,训练和预测

# 数据处理步骤
1. `get_indexing.py`生成cid(vid)对应的视频序号(列表)
2. `get_tfrecord.py`生成训练样本并存成TFRecord文件
3. `get_dataset.py`生成训练样本的`tf.data.dataset`容器,用于训练

> 2步中原始数据是用户的点击序列,如果点击的是栏目的话,元素格式为 <视频id>_<栏目id>;否则是 <视频id>
# 训练 

`train.py`训练并保存模型

# 预测
`infer.py`文件载入模型,读取用户的行为序列和栏目的视频序号列表来排序栏目,并将结果存入.json.gz文件

# 环境部署
`pip install -r requirements.txt`

### tensorflow2安装
参考 https://tensorflow.google.cn/install?hl=zh-cn
1. 使用python3.5–3.8
> 号称支持3.8,但是报错 
> AttributeError: module 'tensorflow.python.keras.utils.generic_utils' has no attribute 'populate_dict_with_module_objects'
> 请参考 https://stackoverflow.com/questions/61137954/attributeerror-module-tensorflow-python-keras-utils-generic-utils-has-no-attr

2. 更新setuptools
```shell
pip3 install --ignore-installed setuptools
```
> 此处容易报错,参考 https://blog.csdn.net/sinat_40875078/article/details/105612356

3. 用豆瓣或清华的源安装tf-cpu
```
# Requires the latest pip
pip install --upgrade pip

# 用国内的源安装tf
pip install tensorflow -i https://pypi.tuna.tsinghua.edu.cn/simple/
```


# FAQ

### 为什么使用TFRecord和dataset
- TFRecord基于`protocol buffer`,后者是google开发的跨平台,跨语言,序列化和反序列化很高效的机制.
> 参考 https://www.tensorflow.org/tutorials/load_data/tfrecord

- `tf.data.dataset`是tf自己的数据管道构件,提供了方便的接口,支持多种数据载入和输入方法和高效的数据输入
> 参考 https://www.tensorflow.org/api_docs/python/tf/data/Dataset

### 模型多个输入如何处理
model的`call`方法的输入比较灵活,可以用dict或者tuple的方式来获得多个输入  

> 参考 https://blog.csdn.net/qq_39238461/article/details/109107250
> 参考 model_tf2.py
### 多输入时生成Dataset
`from_generator`时,output_types, output_shapes用dict来定义;
> 参考  tfds_try.py 

从TFRecord文件读取时`map`直接返回dict
> 参考 get_dataset.py

### 多输入时padding
`dataset`的`padded_batch`方法中`padded_shapes`用dict来定义
> 参考 https://github.com/tensorflow/tensorflow/issues/14704

### TFRecordDataset划分
不像`numpy.array`那样用`sklearn.model_selection.train_test_split`方便,需要用`take`, `skip`等方法操作dataset
> 参考 https://stackoverflow.com/questions/51125266/how-do-i-split-tensorflow-datasets  
> 注意,batch之后,

目前没有简单方法获得其大小,需要直接遍历`iterator`来获得
> 参考 https://github.com/tensorflow/tensorflow/issues/26966


# 踩过的坑 

- `tf.io.parse_sequence_example`方法解析TFRecord文件时,返回是tuple-3,而不是网上很多资料的tuple-2

- 用`tf.keras.backend.binary_crossentropy`损失函数时,`target`需要是float类型,int会报错

- `tf.data.dataset`用了batch方法后,`take`和`skip`方法的操作单元都变成了mini_batch,而不是sample.比如
```
dataset = dataset.batch(2)
take(2)  # 前2个batch,共4个sample
```

- tensorflow2.3.1`load_model`后模型的一些部分缺损,报错
`ValueError: Could not find matching function to call loaded from the SavedModel`

subclassed model适合用保存权重的方式.  
- 因为`saved_model`对于subclassed model保存的是`call`函数,不方便序列化.尤其是自定的`call`方法参数不是`inputs`和`training`时
- 对functional或sequential model保存的是data structure

> 参考 https://stackoverflow.com/questions/58339137/tensorflow-2-0-save-and-load-a-model-that-contains-a-lstm-layer-while-the-load

# tensorflow2的技巧
### 查看梯度  
eager execution下要用`tf.GradientTape`

```python
import tensorflow as tf 
  
x = tf.constant(4.0) 
  
# Using GradientTape 
with tf.GradientTape() as gt: 
    gt.watch(x) 
    y = x * x * x 
  
# Computing first order gradient 
first_order = gt.gradient(y, x) 
  
# Computing Second order gradient 
second_order  = gt.gradient(first_order, x)  
  
# Printing result 
print("first_order: ", first_order) 
print("second_order: ", second_order) 
```

### 自定义Layer
`call`方法最好有`inputs`和`training`参数

### 查看模型中的变量

- 如果是`tf.Variable`,则用`read_value()`
- 如果是`tf.keras.layers.Layer`,则用`get_weights()`

```python 
import tensorlfow as tf 
model = tf.keras.models.load_model(filepath)
embeddings = model.tag_embedding

# tf.Variable
tag_mat = tag_embedding.read_value().numpy().tolist()

# tf.keras.layers.Layer
wk = model.attn_list[0].Wk.get_weights()[0].tolist()
```


### 载入模型时有自定的函数
用`custom_objects`参数

```python 
import tensorlfow as tf 
model = tf.keras.models.load_model(filepath, custom_objects={"loss_fn": loss_fn})
```

### 制作mask

```python
import tensorflow as tf
import numpy as np 
item_inputs = np.random.randint(low=0, high=100, size=(20, 20)).astype(dtype=float)
one_item = tf.ones_like(item_inputs, dtype=tf.float32)
zero_item = tf.zeros_like(item_inputs, dtype=tf.float32)
item_mask = tf.where(item_inputs == 0, x=one_item, y=zero_item)
```



