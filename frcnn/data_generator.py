import numpy as np


def get_anchor_generator(all_img_data, class_count, config, img_length_calc_fn, backend, mode='train'):
    for img in all_img_data:
        pass