import os
import os.path as osp
import json
import math
from glob import glob
from pprint import pprint
import streamlit as st

import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import cv2
import albumentations as A
import lanms
from albumentations.pytorch import ToTensorV2
from imageio import imread

from model import EAST
from detect import detect


def draw_bbox(
    image,
    bbox,
    color=(0, 0, 255),
    thickness=1,
    thickness_sub=None,
    double_lined=False,
    write_point_numbers=False,
):
    """이미지에 하나의 bounding box를 그려넣는 함수"""
    thickness_sub = thickness_sub or thickness * 3
    basis = max(image.shape[:2])
    fontsize = basis / 1500
    x_offset, y_offset = int(fontsize * 12), int(fontsize * 10)
    color_sub = (255 - color[0], 255 - color[1], 255 - color[2])

    points = [(int(np.rint(p[0])), int(np.rint(p[1]))) for p in bbox]

    for idx in range(len(points)):
        if double_lined:
            cv2.line(
                image,
                points[idx],
                points[(idx + 1) % len(points)],
                color_sub,
                thickness=thickness_sub,
            )
        cv2.line(
            image,
            points[idx],
            points[(idx + 1) % len(points)],
            color,
            thickness=thickness,
        )

    if write_point_numbers:
        for idx in range(len(points)):
            loc = (points[idx][0] - x_offset, points[idx][1] - y_offset)
            if double_lined:
                cv2.putText(
                    image,
                    str(idx),
                    loc,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    fontsize,
                    color_sub,
                    thickness_sub,
                    cv2.LINE_AA,
                )
            cv2.putText(
                image,
                str(idx),
                loc,
                cv2.FONT_HERSHEY_SIMPLEX,
                fontsize,
                color,
                thickness,
                cv2.LINE_AA,
            )


def draw_bboxes(
    image,
    bboxes,
    color=(0, 0, 255),
    thickness=1,
    thickness_sub=None,
    double_lined=False,
    write_point_numbers=False,
):
    """이미지에 다수의 bounding box들을 그려넣는 함수"""
    for bbox in bboxes:
        draw_bbox(
            image,
            bbox,
            color=color,
            thickness=thickness,
            thickness_sub=thickness_sub,
            double_lined=double_lined,
            write_point_numbers=write_point_numbers,
        )


def gray_mask_to_heatmap(x):
    x = cv2.cvtColor(cv2.applyColorMap(x, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    return x


def get_superimposed_image(image, score_map, heatmap=True, w_image=None, w_map=None):
    """이미지에 map/mask 등을 중첩하여 그려주는 함수

    Args:
        image (ndarray): (H, W, C) shaped, float32 or uint8 dtype is allowed.
        score_map (ndarray): (H, W) shaped, float32 or uint8 dtype is allowed.
        heatmap (boot): Wheather to convert `score_map` into a heatmap.
        w_image (float)
        w_map (float)

    Blending weights(`w_image` and `w_map`) are default to (0.4, 0.6).
    """

    assert w_image is None or (w_image > 0 and w_image < 1)
    assert w_map is None or (w_map > 0 and w_map < 1)

    if image.dtype != np.uint8:
        image = (255 * np.clip(image, 0, 1)).astype(np.uint8)

    if score_map.dtype != np.uint8:
        score_map = (255 * np.clip(score_map, 0, 1)).astype(np.uint8)
    if heatmap:
        score_map = gray_mask_to_heatmap(score_map)
    elif score_map.ndim == 2 or score_map.shape[2] != 3:
        score_map = cv2.cvtColor(score_map, cv2.COLOR_GRAY2RGB)

    if w_image is None and w_map is None:
        w_image, w_map = 0.4, 0.6
    elif w_image is None:
        w_image = 1 - w_map
    elif w_map is None:
        w_map = 1 - w_image

    return cv2.addWeighted(image, w_image, score_map, w_map, 0)
