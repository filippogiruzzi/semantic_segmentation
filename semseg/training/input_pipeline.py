import os
import argparse
import glob
import logging
from functools import partial

import tensorflow as tf
import numpy as np
import cv2
import yaml

from semseg.data_processing.data_viz import mask_to_rgb


logger = logging.getLogger(__name__)


def get_dataset(tfrecords,
                batch_size,
                epochs,
                input_size=(512, 512),
                shuffle=False,
                fake_input=False):
    meta_data = None
    for tfrecord_name in tfrecords:
        try:
            with open(tfrecord_name + '.metadata', 'r') as yaml_file:
                meta_data = yaml.load(yaml_file, Loader=yaml.Loader)
                break
        except Exception as e:
            logger.error("Ignoring exception {}".format(e))
            pass

        if meta_data is None:
            raise ValueError("meta data missing")

    def parse_func(example_proto, meta_data=None):
        feature_dict = {
            'data_type': tf.FixedLenFeature([], tf.string),
            'file_id': tf.FixedLenFeature([], tf.string),
            'img': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.string)
        }
        parsed_feature = tf.parse_single_example(example_proto, feature_dict)

        features, labels = {}, {}
        for key, val in parsed_feature.items():
            if key in ['img', 'label']:
                val = tf.image.decode_png(val, channels=3)
                val = tf.reshape(val, shape=meta_data[key]['shape'])
            if key == 'label':
                val = tf.image.resize_images(val, size=list(input_size), method=1)
                val = tf.slice(val, begin=[0, 0, 0], size=[input_size[0], input_size[1], 1])
                val = tf.squeeze(val, axis=-1)
                labels[key] = val
            else:
                features[key] = val
        return features, labels

    files = tf.data.Dataset.list_files(tfrecords)
    dataset = files.apply(tf.contrib.data.parallel_interleave(tf.data.TFRecordDataset, cycle_length=10))
    if shuffle:
        dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(buffer_size=256, count=epochs))
    else:
        dataset = dataset.repeat(epochs)

    cur_parse_func = partial(parse_func, meta_data=meta_data)
    dataset = dataset.map(cur_parse_func, num_parallel_calls=8)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=64)
    if fake_input:
        dataset = dataset.take(1).cache().repeat()
    return dataset


def data_input_fn(tfrecords,
                  batch_size,
                  epochs,
                  input_size=(512, 512),
                  shuffle=False,
                  fake_input=False):
    def _input_fn():
        dataset = get_dataset(tfrecords,
                              batch_size,
                              epochs,
                              input_size,
                              shuffle,
                              fake_input)

        it = dataset.make_one_shot_iterator()
        next_batch = it.get_next()

        img_input = next_batch[0]['img']
        label = next_batch[1]['label']

        with tf.name_scope('preprocess_img_input'):
            img_input = tf.cast(img_input, dtype=tf.float32)
            img_input = tf.divide(img_input, 255.0)
            # img_input = tf.subtract(img_input, 0.5)
            img_input = tf.image.resize_images(img_input, size=list(input_size))

        features = {'img_input': img_input}
        labels = {'label': label}
        return features, labels
    return _input_fn


def main():
    parser = argparse.ArgumentParser(description='visualize input pipeline')
    parser.add_argument('--data-dir', '-d', type=str, default='/home/filippo/datasets/carla_semseg_data/tfrecords/')
    parser.add_argument('--data-type', '-t', type=str, default='train')
    args = parser.parse_args()

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    tfrecords = glob.glob('{}{}/*.tfrecord'.format(args.data_dir, args.data_type))
    dataset = get_dataset(tfrecords,
                          batch_size=32,
                          epochs=1,
                          input_size=(256, 256),
                          shuffle=False,
                          fake_input=False)
    print('\nDataset out types {}'.format(dataset.output_types))

    batch = dataset.make_one_shot_iterator().get_next()
    sess = tf.Session()
    try:
        batch_nb = 0
        while True:
            data = sess.run(batch)
            batch_nb += 1

            data_type = data[0]['data_type']
            file_id = data[0]['file_id']
            img = data[0]['img']
            label = data[1]['label']

            print(img.shape, label.shape)

            print('\nBatch nb {}'.format(batch_nb))
            for i in range(len(img)):
                cv2.imshow('RGB', cv2.cvtColor(img[i], cv2.COLOR_RGB2BGR))
                cv2.imshow('SEMSEG', mask_to_rgb(np.expand_dims(label[i], axis=-1), ax=0))
                cv2.waitKey(0)

    except tf.errors.OutOfRangeError:
        pass


if __name__ == '__main__':
    main()
