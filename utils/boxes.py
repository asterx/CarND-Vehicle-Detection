# coding=utf-8
import cv2
import numpy as np

DEFAULT_COLOR = (255, 0, 0)
DEFAULT_THICKNESS = 5


def draw_boxes(img, boxes, color = DEFAULT_COLOR, thickness = DEFAULT_THICKNESS):
    res = np.copy(img)
    for box in boxes:
        cv2.rectangle(res, box[0], box[1], color, thickness)
    return res


def draw_labeled_boxes(img, labels, color = DEFAULT_COLOR, thickness = DEFAULT_THICKNESS):
    for car_num in range(1, labels[1]+1):
        n_zero_y, n_zero_x = list(map(np.array, (labels[0] == car_num).nonzero()))
        box = ((np.min(n_zero_x), np.min(n_zero_y)), (np.max(n_zero_x), np.max(n_zero_y)))
        cv2.rectangle(img, box[0], box[1], color, thickness)
    return img


def add_heat(heatmap, box_list):
    for box in box_list:
        heatmap[ box[0][1]:box[1][1], box[0][0]:box[1][0] ] += 1
    return heatmap


def apply_threshold(heatmap, threshold):
    heatmap[ heatmap <= threshold ]  = 0
    return heatmap
