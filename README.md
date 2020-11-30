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



