import os
import glob
import argparse
import logging
import sys
from datetime import datetime

import numpy as np
import tensorflow as tf
import cv2

from semseg.training.input_pipeline import data_input_fn
from semseg.training.estimator import SemsegEstimator
from semseg.data_processing.data_viz import mask_to_rgb


def main():
    parser = argparse.ArgumentParser(description='train UNet for Semantic segmentation')
    parser.add_argument('--data-dir', '-d', type=str, default='/home/filippo/datasets/carla_semseg_data/tfrecords/',
                        help='tf records data directory')
    parser.add_argument('--model-dir', type=str, default='', help='pretrained model directory')
    parser.add_argument('--ckpt', type=str, default='', help='pretrained checkpoint directory')
    parser.add_argument('--mode', '-m', type=str, default='train', help='train, eval or predict')
    parser.add_argument('--model', type=str, default='unet', help='model name')
    parser.add_argument('--input-size', type=str, default='512x512', help='image input size')
    parser.add_argument('--batch-size', '-bs', type=int, default=8, help='batch size')
    parser.add_argument('--epochs', '-e', type=int, default=100, help='train epochs')
    parser.add_argument('--n-classes', '-n', type=int, default=13, help='number of classes in output')
    parser.add_argument('--learning-rate', '-lr', type=float, default=0.001, help='initial learning rate')
    parser.add_argument('--fake-input', action='store_true', default=False, help='debug with 1 batch training')
    args = parser.parse_args()

    assert args.model in ['unet'], 'Wrong model name'
    assert len(args.input_size.split('x')) == 2, '2 values required for --input-size'
    input_size = tuple([int(x) for x in args.input_size.split('x')])

    tfrecords_train = glob.glob('{}train/*.tfrecord'.format(args.data_dir))
    tfrecords_val = glob.glob('{}val/*.tfrecord'.format(args.data_dir))
    tfrecords_test = glob.glob('{}test/*.tfrecord'.format(args.data_dir))

    logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)
    logger = logging.getLogger(__name__)
    np.random.seed(0)
    tf.set_random_seed(0)
    tf.logging.set_verbosity(tf.logging.INFO)

    if not args.model_dir:
        save_dir = '{}models/{}/{}/'.format(args.data_dir, args.model, datetime.now().isoformat())
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    else:
        save_dir = args.model_dir

    params = {
        'model': args.model,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'input_size': input_size,
        'n_classes': args.n_classes,
        'lr': args.learning_rate,
    }

    train_config = tf.estimator.RunConfig(save_summary_steps=10,
                                          save_checkpoints_steps=500,
                                          keep_checkpoint_max=10,
                                          log_step_count_steps=10)

    ws = None
    if args.ckpt:
        ws = tf.estimator.WarmStartSettings(ckpt_to_initialize_from=args.ckpt, vars_to_warm_start='.*')

    # Create TensorFlow estimator object
    estimator_obj = SemsegEstimator(params)
    estimator = tf.estimator.Estimator(model_fn=estimator_obj.model_fn,
                                       model_dir=save_dir,
                                       config=train_config,
                                       params=params,
                                       warm_start_from=ws)

    mode_keys = {
        'train': tf.estimator.ModeKeys.TRAIN,
        'eval': tf.estimator.ModeKeys.EVAL,
        'predict': tf.estimator.ModeKeys.PREDICT
    }
    mode = mode_keys[args.mode]

    # Training & Evaluation on Train / Val set
    if mode == tf.estimator.ModeKeys.TRAIN:
        train_input_fn = data_input_fn(tfrecords_train,
                                       batch_size=params['batch_size'],
                                       epochs=1,
                                       input_size=input_size,
                                       shuffle=True,
                                       fake_input=args.fake_input)
        eval_input_fn = data_input_fn(tfrecords_val,
                                      batch_size=params['batch_size'],
                                      epochs=1,
                                      input_size=input_size,
                                      shuffle=False,
                                      fake_input=args.fake_input)

        for epoch_num in range(params['epochs']):
            logger.info("Training for epoch {} ...".format(epoch_num))
            estimator.train(input_fn=train_input_fn)
            logger.info("Evaluation for epoch {} ...".format(epoch_num))
            estimator.evaluate(input_fn=eval_input_fn)

    # Evaluation on Test set
    elif mode == tf.estimator.ModeKeys.EVAL:
        test_input_fn = data_input_fn(tfrecords_test,
                                      batch_size=params['batch_size'],
                                      epochs=1,
                                      input_size=input_size,
                                      shuffle=False,
                                      fake_input=args.fake_input)

        logger.info("Evaluation of test set ...")
        estimator.evaluate(input_fn=test_input_fn)

    # Prediction visualization on Test set
    elif mode == tf.estimator.ModeKeys.PREDICT:
        test_input_fn = data_input_fn(tfrecords_test,
                                      batch_size=1,
                                      epochs=1,
                                      input_size=input_size,
                                      shuffle=False,
                                      fake_input=args.fake_input)

        predictions = estimator.predict(input_fn=test_input_fn)
        for n, pred in enumerate(predictions):
            img_input = pred['img_input']
            pred = pred['semseg']

            # print(np.where(np.argmax(pred, axis=-1) == 0))

            # cv2.imshow('RGB', cv2.resize(cv2.cvtColor(img_input, cv2.COLOR_RGB2BGR), (225, 150)))
            # cv2.imshow('SEMSEG', cv2.resize(cv2.medianBlur(mask_to_rgb(pred, ax=-1), 5), (225, 150)))
            cv2.imshow('RGB', cv2.cvtColor(img_input, cv2.COLOR_RGB2BGR))
            cv2.imshow('SEMSEG', cv2.medianBlur(mask_to_rgb(pred, ax=-1), 5))
            cv2.waitKey(0)


if __name__ == '__main__':
    main()
