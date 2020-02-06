import os
import argparse

import numpy as np
import cv2


CLASSES = {
    '0': (0, 0, 0),  # Unlabeled
    '1': (70, 70, 70),  # Building
    '2': (190, 153, 153),  # Fence
    '3': (250, 170, 160),  # Other
    '4': (220, 20, 60),  # Pedestrian
    '5': (153, 153, 153),  # Pole
    '6': (157, 234, 50),  # Road line
    '7': (128, 64, 128),  # Road
    '8': (244, 35, 232),  # Sidewalk
    '9': (107, 142, 35),  # Vegetation
    '10': (0, 0, 142),  # Car
    '11': (102, 102, 156),  # Wall
    '12': (220, 220, 0)  # Traffic sign
}


def mask_to_rgb(semseg_mask, ax=-1):
    mask = semseg_mask[:, :, ax]
    if semseg_mask.shape[-1] > 3:
        mask = np.argmax(semseg_mask, axis=-1)
    semseg_rgb = np.zeros((semseg_mask.shape[0], semseg_mask.shape[1], 3), dtype=np.uint8)
    for class_id in CLASSES.keys():
        class_locs = np.where(mask == int(class_id))
        semseg_rgb[class_locs[0], class_locs[1], :] = CLASSES[class_id]
    return semseg_rgb


def main():
    parser = argparse.ArgumentParser(description='visualize raw data')
    parser.add_argument('--data-dir', type=str, default='/home/filippo/datasets/carla_semseg_data/')
    args = parser.parse_args()

    dir_ids = ['A', 'B', 'C', 'D', 'E']
    dir_id = dir_ids[0]
    sub_dir = 'data{}/data{}/'.format(dir_id, dir_id.upper())
    img_dir = os.path.join(args.data_dir, sub_dir, 'CameraRGB/')
    labels_dir = os.path.join(args.data_dir, sub_dir, 'CameraSeg/')

    img_fns, labels_fns = os.listdir(img_dir), os.listdir(labels_dir)

    print(len(img_fns), len(labels_fns))
    for fn in img_fns:
        img_fp = os.path.join(img_dir, fn)
        label_fp = os.path.join(labels_dir, fn)

        img, label = cv2.imread(img_fp), cv2.imread(label_fp)
        overlayed = cv2.addWeighted(img, 0.8, mask_to_rgb(label), 0.5, 0)
        # cv2.imshow('RGB', cv2.resize(img, (225, 150)))
        # cv2.imshow('SEMSEG', cv2.resize(mask_to_rgb(label), (225, 150)))
        cv2.imshow('RGB', img)
        cv2.imshow('SEMSEG', mask_to_rgb(label))
        cv2.imshow('OVERLAY', overlayed)
        cv2.waitKey(0)


if __name__ == '__main__':
    main()
