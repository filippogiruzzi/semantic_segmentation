import os
import argparse

import cv2

from semseg.data_processing.data_viz import mask_to_rgb


DIR_IDS = {
    'train': ['A', 'B', 'C'],
    'val': ['D'],
    'test': ['E']
}


def data_iter(data_dir, dir_ids=DIR_IDS, data_type='train'):
    for dir_id in dir_ids[data_type]:
        sub_dir = 'data{}/data{}/'.format(dir_id, dir_id.upper())
        img_dir = os.path.join(data_dir, sub_dir, 'CameraRGB/')
        labels_dir = os.path.join(data_dir, sub_dir, 'CameraSeg/')

        img_fns, labels_fns = os.listdir(img_dir), os.listdir(labels_dir)
        for fn in img_fns:
            img_fp = os.path.join(img_dir, fn)
            label_fp = os.path.join(labels_dir, fn)

            yield {
                'data_type': data_type,
                'file_id': fn.split('.')[0],
                'img_fp': img_fp,
                'label_fp': label_fp
            }


def main():
    parser = argparse.ArgumentParser(description='data iterator to loop through data')
    parser.add_argument('--data-dir', type=str, default='/home/filippo/datasets/carla_semseg_data/')
    args = parser.parse_args()

    train_iter = data_iter(args.data_dir, dir_ids=DIR_IDS)
    for data in train_iter:
        img_fp, label_fp = data['img_fp'], data['label_fp']
        img, label = cv2.imread(img_fp), cv2.imread(label_fp)
        print(img.shape, label.shape)

        cv2.imshow('RGB', img)
        cv2.imshow('SEMSEG', mask_to_rgb(label))
        cv2.waitKey(0)


if __name__ == '__main__':
    main()
