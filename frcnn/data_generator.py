import numpy as np
import cv2
import random


def get_new_image_size(width, height, config):
    min_img_side = config.min_img_size
    if(width < height):
        f = float(min_img_side / width)
        resized_height = int(f * height)
        resized_width = min_img_side
    else:
        f = float(min_img_side / height)
        resized_width = int(f * width)
        resized_height = min_img_side

    return resized_width, resized_height


def get_anchor_generator(all_img_data, class_count, config, img_length_calc_fn, backend, mode='train'):
    if mode == 'train':
        random.shuffle(all_img_data)

    for image_data in all_img_data:
        img = cv2.imread(all_img_data['filepath'])

        height, width, _ = img.shape

        resized_width, resized_height = get_new_image_size(width, height, config)

        resized_img = cv2.resize(img, (resized_width, resized_height), interpolation=cv2.INTER_CUBIC)

        y_rpn_cls, y_rpn_regr = calc_rpn()