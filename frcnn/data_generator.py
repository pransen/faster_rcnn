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


def union(b1, b2, area_intersection):
    area_b1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    area_b2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    area_union = area_b1 + area_b2 - area_intersection
    return area_union

def intersection(b1, b2):
    x_min = max(b1[0], b2[0])
    y_min = max(b1[1], b2[1])
    w = min(b1[2], b2[2]) - x_min
    h = min(b1[3], b2[3]) - y_min
    if w < 0 or h < 0:
        return 0
    return w * h

def iou(b1, b2):
    # To avoid divide by zero
    delta = 1e-6
    area_intersection = intersection(b1, b2)
    area_union = union(b1, b2, area_intersection)
    return float(area_intersection/ area_union + delta)

def calc_rpn(config, image_data, width, height, resized_width, resized_height, img_length_calc_fn):
    downscale = float(config.rpn_stride)
    anchor_box_scales = config.anchor_box_scales
    anchor_box_ratios = config.anchor_box_ratios

    # At each location we propose a fixed number of anchors
    num_anchors = len(anchor_box_scales) * len(anchor_box_ratios)

    # Output feature map size
    feature_map_width, feature_map_height = img_length_calc_fn(width, height)

    num_anchor_ratios = len(anchor_box_ratios)
    num_anchor_scales = len(anchor_box_scales)

    # initialize empty objectives
    y_rpn_regr = np.zeros((feature_map_height, feature_map_width, num_anchors * 4))
    y_rpn_overlap = np.zeros((feature_map_height, feature_map_width, num_anchors))
    y_is_box_valid = np.zeros((feature_map_height, feature_map_width, num_anchors))

    num_bboxes = len(image_data['bboxes'])

    # This data structures track the anchors with the highest IOU with a bounding box
    # for condition 1 of section 3.1.2 of paper
    num_anchors_for_bbox = np.zeros(num_bboxes).astype(int)
    best_anchor_for_bbox = -1 * np.ones((num_bboxes, 4)).astype(int)
    best_iou_for_anchor_box = np.zeros(num_bboxes).astype(np.float32)
    best_x_for_bbox = np.zeros((num_bboxes, 4)).astype(int)
    best_dx_for_bbox = np.zeros((num_bboxes, 4)).astype(np.float32)

    gt_boxes = np.zeros((num_bboxes, 4))

    for bbox_num, bbox in enumerate(image_data['bboxes']):
        gt_boxes[bbox_num, 0] = float(bbox['x1'] * resized_width / width)
        gt_boxes[bbox_num, 1] = float(bbox['x2'] * resized_width / width)
        gt_boxes[bbox_num, 2] = float(bbox['y1'] * resized_height / height)
        gt_boxes[bbox_num, 3] = float(bbox['y2'] * resized_height / height)


    for anchor_scale_idx in range(num_anchor_scales):
        for anchor_ratio_idx in range(num_anchor_ratios):
            anchor_w = anchor_box_scales[anchor_scale_idx] * anchor_box_ratios[anchor_ratio_idx][0]
            anchor_h = anchor_box_scales[anchor_scale_idx] * anchor_box_ratios[anchor_ratio_idx][1]

            for ix in range(feature_map_width):
                x1_anchor = downscale * (ix + 0.5) - anchor_w / 2
                x2_anchor = downscale * (ix + 0.5) + anchor_w / 2

                if x1_anchor < 0 or x2_anchor > resized_width:
                    continue
                for jy in range(feature_map_height):
                    y1_anchor = downscale * (jy + 0.5) - anchor_h / 2
                    y2_anchor = downscale * (jy + 0.5) + anchor_h / 2

                    if y1_anchor < 0 or y2_anchor > resized_height:
                        continue

                    bbox_type = 'neg'
                    best_iou_for_loc = 0.0

                    for box_idx in range(num_bboxes):
                        bbox = gt_boxes[box_idx]
                        anchor_box = (x1_anchor, y1_anchor, x2_anchor, y2_anchor)
                        bounding_box = (bbox[0], bbox[2], bbox[1], bbox[3])
                        curr_iou = iou(bounding_box, anchor_box)
                        if curr_iou > best_iou_for_anchor_box[box_idx] or curr_iou > config.rpn_max_overlap:
                            cx_gt = float((bbox[0] + bbox[1])/2.0)
                            cy_gt = float((bbox[2] + bbox[3]) / 2.0)
                            cx_anchor = float((x1_anchor + x2_anchor)/2)
                            cy_anchor = float((y1_anchor + y2_anchor) / 2)

                            tx = float((cx_gt - cx_anchor)/ anchor_w)
                            ty = float((cy_gt - cy_anchor) / anchor_h)
                            tw = np.log(float((bbox[1] - bbox[0])/ anchor_w))
                            th = np.log(float((bbox[3] - bbox[2]) / anchor_h))

                        # An anchor can be positive, negative or neutral
                        # positive for best iou with a gt or iou > 0.7 with any gt boxes
                        # negative if iou < 0.3
                        # neutral if iou is in between 0.3 and 0.7
                        if image_data['bboxes'][box_idx]['class'] != 'bg':
                            # Keep track of the best anchor all the anchor boxes
                            if curr_iou > best_iou_for_anchor_box[box_idx]:
                                best_anchor_for_bbox[box_idx] = [jy, ix, anchor_ratio_idx, anchor_scale_idx]
                                best_iou_for_anchor_box[box_idx] = curr_iou
                                best_x_for_bbox[box_idx] = [x1_anchor, x2_anchor, y1_anchor, y2_anchor]
                                best_dx_for_bbox[box_idx] = [tx, ty, tw, th]

                            if curr_iou > config.rpn_max_overlap:
                                bbox_type = 'pos'
                                num_anchors_for_bbox[box_idx] += 1
                                if curr_iou > best_iou_for_loc:
                                    best_iou_for_loc = curr_iou
                                    best_regr = (tx, ty, tw, th)

                            if config.rpn_min_overlap < curr_iou < config.rpn_max_overlap:
                                if bbox_type == 'pos':
                                    bbox_type = 'neutral'

                    if bbox_type == 'neg':
                        y_is_box_valid[jy, ix, anchor_ratio_idx + num_anchor_ratios * anchor_scale_idx] = 1
                        y_rpn_overlap[jy, ix, anchor_ratio_idx + num_anchor_ratios * anchor_scale_idx] = 0
                    elif bbox_type == 'neutral':
                        y_is_box_valid[jy, ix, anchor_ratio_idx + num_anchor_ratios * anchor_scale_idx] = 0
                        y_rpn_overlap[jy, ix, anchor_ratio_idx + num_anchor_ratios * anchor_scale_idx] = 0
                    elif bbox_type == 'pos':
                        y_is_box_valid[jy, ix, anchor_ratio_idx + num_anchor_ratios * anchor_scale_idx] = 1
                        y_rpn_overlap[jy, ix, anchor_ratio_idx + num_anchor_ratios * anchor_scale_idx] = 1
                        start = 4 * (anchor_ratio_idx + num_anchor_ratios * anchor_scale_idx)
                        y_rpn_regr[jy, ix, start: start + 4] = best_regr

    for idx in range(num_bboxes):
        if num_anchors_for_bbox[idx] == 0:
            if best_anchor_for_bbox[idx] == -1:
                # no anchor with iou greater than 0 for current bbox
                continue
            jy, ix, anchor_ratio_idx, anchor_scale_idx = best_anchor_for_bbox[box_idx]
            best_regr = best_dx_for_bbox[box_idx]

            y_is_box_valid[jy, ix, anchor_ratio_idx + num_anchor_ratios * anchor_scale_idx] = 1
            y_rpn_overlap[jy, ix, anchor_ratio_idx + num_anchor_ratios * anchor_scale_idx] = 1
            start = 4 * (anchor_ratio_idx + num_anchor_ratios * anchor_scale_idx)
            y_rpn_regr[jy, ix, start: start + 4] = best_regr

    y_rpn_overlap = np.transpose(y_rpn_overlap, (2, 0, 1))
    y_rpn_overlap = np.expand_dims(y_rpn_overlap, axis = 0)

    y_rpn_regr = np.transpose(y_rpn_regr, (2, 0, 1))
    y_rpn_regr = np.expand_dims(y_rpn_regr, axis=0)

    y_is_box_valid = np.transpose(y_is_box_valid, (2, 0, 1))
    y_is_box_valid = np.expand_dims(y_is_box_valid, axis=0)

    pos_locs = np.where(np.logical_and(y_rpn_overlap[0, :, :, :] == 1, y_is_box_valid[0, :, :, :] == 1))
    neg_locs = np.where(np.logical_and(y_rpn_overlap[0, :, :, :] == 0, y_is_box_valid[0, :, :, :] == 1))

    # section 3.1.3 of paper
    # RPN has much more negative locations as compared to positive locations
    # Hence the total number of regions is constrained to 256
    num_regions = 256

    num_pos = len(pos_locs[0])
    # If number of positive regions is > 128, make it equal to 128
    if num_pos > num_regions / 2:
        val_locs = random.sample(range(num_pos), num_regions / 2)
        y_is_box_valid[0, pos_locs[0][val_locs], pos_locs[1][val_locs], pos_locs[2][val_locs]] = 0
        num_pos = num_regions / 2
    if len(neg_locs[0]) + num_pos > num_regions:
        val_locs = random.sample(range(len(neg_locs[0])), num_regions - num_pos)
        y_is_box_valid[0, neg_locs[0][val_locs], neg_locs[1][val_locs], neg_locs[2][val_locs]] = 0

    y_rpn_cls = np.concatenate([y_is_box_valid, y_rpn_overlap], axis=1)
    y_rpn_regr = np.concatenate([np.repeat(y_rpn_overlap, 4, axis=1), y_rpn_regr], axis=1)

    return np.copy(y_rpn_cls), np.copy(y_rpn_regr)

def get_anchor_generator(all_img_data, class_count, config, img_length_calc_fn, backend, mode='train'):
    if mode == 'train':
        random.shuffle(all_img_data)

    for image_data in all_img_data:
        img = cv2.imread(all_img_data['filepath'])

        height, width, _ = img.shape

        resized_width, resized_height = get_new_image_size(width, height, config)

        resized_img = cv2.resize(img, (resized_width, resized_height), interpolation=cv2.INTER_CUBIC)

        y_rpn_cls, y_rpn_regr = calc_rpn(config, image_data, width, height, resized_width, resized_height, img_length_calc_fn)

        resized_img = resized_img[:, :, (2, 1, 0)]  # BGR -> RGB
        resized_img = resized_img.astype(np.float32)
        resized_img[:, :, 0] -= config.img_channel_mean[0]
        resized_img[:, :, 1] -= config.img_channel_mean[1]
        resized_img[:, :, 2] -= config.img_channel_mean[2]
        resized_img /= config.img_scaling_factor

        resized_img = np.transpose(resized_img, (2, 0, 1))
        resized_img = np.expand_dims(resized_img, axis=0)

        y_rpn_regr[:, y_rpn_regr.shape[1] // 2:, :, :] *= config.std_scaling

        if backend == 'tf':
            resized_img = np.transpose(resized_img, (0, 2, 3, 1))
            y_rpn_cls = np.transpose(y_rpn_cls, (0, 2, 3, 1))
            y_rpn_regr = np.transpose(y_rpn_regr, (0, 2, 3, 1))

        yield np.copy(resized_img), [np.copy(y_rpn_cls), np.copy(y_rpn_regr)], image_data