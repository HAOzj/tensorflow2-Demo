"""
     > File Name: trans_data_tfrecord.py
     > Author: Jiangchangjun
     > Mail: Jiangchangjun@ffalcon.com 
     > Created Time: Tue 20 Aug 2019 10:57:54 AM CST
 """
import argparse
import tensorflow as tf
import os
import json
import gzip
from multiprocessing import Pool


def _int64_feature(value):
    """
    int64 is used for numeric values
    """
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _float_feature(value):
    """
    _float is used for string/char values
    """
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _float_feature_weight(value):
    """
    _float is used for feature weight values
    """
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _bytes_feature(value):
    """
    _bytes is used for sequence values
    """
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def _read_dir(dir_path, output_dir):
    """
    read input dir data
    """
    files = os.listdir(dir_path)
    process_pool = Pool(8)  # define 8 process to execute this task
    for fi in files:
        data_file = dir_path + '/' + fi
        tfrecord_file = output_dir + '/' + fi.split('.')[0] + '.tfrecord'
        # thread_pool.submit(_trans_2_tfrecord, data_file, tfrecord_file)
        process_pool.apply_async(_trans_2_tfrecord, args=(data_file, tfrecord_file,))
        #_trans_2_tfrecord(data_file, tfrecord_file)
    print('waiting for all subprocesses done...')
    process_pool.close()
    process_pool.join()
    print('all subprocesses done')


def _trans_2_tfrecord(input_file, output_file):
    """
    read data from gzip file
    write data into tfrecord file
    """
    print("\nStart to convert %s to %s\n" %(input_file, output_file))
    with gzip.open(input_file, 'rb') as f:
        writer = tf.io.TFRecordWriter(output_file)
        for line in f:
            try:
                info = json.loads(line)
                column_sequence = info.get("column_sequence", [])
                behavior_sequence = [item.encode() for item in info.get('behavior_sequence', None)]
                if not column_sequence:
                    print('[column_sequence is None]')
                    continue
                if behavior_sequence is None:
                    print('[behavior_sequence is None]')
                    continue
                label = info.get("label", -1)

                example = tf.train.Example(features=tf.train.Features(
                        feature={
                            'label':
                                _int64_feature(label),
                            'behaviorSequenceFeature':
                                _bytes_feature(behavior_sequence),
                            "column_sequence":
                                _bytes_feature(column_sequence)
                            }
                        )
                    ) 
                writer.write(example.SerializeToString())
            except json.decoder.JSONDecodeError as e:
                print(e.msg)
                continue
    writer.close()


def parse_args():
    """
    Parse the trans file relate params
    """
    parser = argparse.ArgumentParser(description="Trans dataFile format into Tfrecords parms")

    parser.add_argument('--input_dir', 
                        type=str,
                        default='/data/jiangchangjun/ctr_model/feature/train/movie/',
                        help='Input data dir')
    parser.add_argument('--num_files',
                        type=int,
                        default=10,
                        help='Write tfRecord file num')
    parser.add_argument('--output_dir',
                        type=str,
                        default='/data/jiangchangjun/ctr_model/tf_record_dataset/movie/',
                        help='Write tfRecord file dir')
    flags, unsued_flags = parser.parse_known_args()
    return flags


if __name__ == '__main__':
    params = parse_args()
    _read_dir(params.input_dir, params.output_dir)
