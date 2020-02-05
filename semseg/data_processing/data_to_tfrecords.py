import os
import multiprocessing
import io
from absl import flags, app
from copy import deepcopy

import tensorflow as tf
import numpy as np
import yaml
import PIL
from tqdm import tqdm

from semseg.data_processing.data_iterator import data_iter


flags.DEFINE_string('data_dir',
                    '/home/filippo/datasets/carla_semseg_data/',
                    'data directory path')
flags.DEFINE_integer('num_shards',
                     256,
                     'number of tfrecord files')
flags.DEFINE_boolean('debug',
                     False,
                     'debug for a few samples')
flags.DEFINE_string('data_type',
                    'trainvaltest',
                    'data types to write into tfrecords')

FLAGS = flags.FLAGS
tf.logging.set_verbosity(tf.logging.INFO)


def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def int64_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def bytes_list_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def create_tf_example(data_type, file_id, img_fp, label_fp):
    with tf.gfile.GFile(img_fp, 'rb') as f:
        img_encoded = f.read()
        img_encoded_io = io.BytesIO(img_encoded)
        img = np.array(PIL.Image.open(img_encoded_io))
    with tf.gfile.GFile(label_fp, 'rb') as f:
        label_encoded = f.read()
        label_encoded_io = io.BytesIO(label_encoded)
        label = np.array(PIL.Image.open(label_encoded_io))

    feature_dict = {
        'data_type': bytes_feature(data_type.encode()),
        'file_id': bytes_feature(file_id.encode()),
        'img': bytes_feature(img_encoded),
        'label': bytes_feature(label_encoded)
    }
    meta_data = {
        'img': {'dtype': img.dtype, 'shape': img.shape},
        'label': {'dtype': label.dtype, 'shape': label.shape}
    }

    example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
    return example, meta_data


def pool_create_tf_example(args):
    return create_tf_example(*args)


def write_tfrecords(path, dataiter, num_shards=256, nmax=-1):
    writers = [
        tf.python_io.TFRecordWriter('{}{:05d}_{:05d}.tfrecord'.format(path, i, num_shards)) for i in range(num_shards)
    ]
    print('\nWriting to output path: {}'.format(path))
    pool = multiprocessing.Pool()
    counter = 0
    for i, (tf_example, meta_data) in tqdm(enumerate(pool.imap(pool_create_tf_example,
                                                               [(deepcopy(data['data_type']),
                                                                 deepcopy(data['file_id']),
                                                                 deepcopy(data['img_fp']),
                                                                 deepcopy(data['label_fp'])
                                                                 ) for data in dataiter]))):
        if tf_example is not None:
            writers[i % num_shards].write(tf_example.SerializeToString())
            with open('{}{:05d}_{:05d}.tfrecord.metadata'.format(path, i, num_shards), 'w') as yaml_file:
                yaml.dump(meta_data, yaml_file, default_flow_style=False)

            counter += 1
        if 0 < nmax < i:
            break
    pool.close()
    for writer in writers:
        writer.close()
    print('Recorded {} data elements'.format(counter))


def create_tfrecords(data_dir, num_shards=256, debug=False, data_type='trainval'):
    np.random.seed(0)

    output_path = os.path.join(data_dir, 'tfrecords/')
    if not tf.gfile.IsDirectory(output_path):
        tf.gfile.MakeDirs(output_path)

    train_it = data_iter(data_dir, data_type='train')
    val_it = data_iter(data_dir, data_type='val')
    test_it = data_iter(data_dir, data_type='test')

    # Write data to tfrecords format
    nmax = 300 if debug else -1
    if 'train' in data_type:
        print('\nWriting train tfrecords ...')
        train_path = os.path.join(output_path, 'train/')
        if not tf.gfile.IsDirectory(train_path):
            tf.gfile.MakeDirs(train_path)
        write_tfrecords(train_path, train_it, num_shards, nmax=nmax)

    if 'val' in data_type:
        print('\nWriting val tfrecords ...')
        val_path = os.path.join(output_path, 'val/')
        if not tf.gfile.IsDirectory(val_path):
            tf.gfile.MakeDirs(val_path)
        write_tfrecords(val_path, val_it, num_shards, nmax=nmax)

    if 'test' in data_type:
        print('\nWriting test tfrecords ...')
        test_path = os.path.join(output_path, 'test/')
        if not tf.gfile.IsDirectory(test_path):
            tf.gfile.MakeDirs(test_path)
        write_tfrecords(test_path, test_it, num_shards, nmax=nmax)


def main(_):
    create_tfrecords(FLAGS.data_dir, num_shards=FLAGS.num_shards, debug=FLAGS.debug, data_type=FLAGS.data_type)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    app.run(main)
