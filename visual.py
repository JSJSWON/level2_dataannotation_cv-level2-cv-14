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
from visual_function import *


@st.cache
def load_model(CHECKPOINT_DIR, checkpoint):
    model = EAST(pretrained=False)
    model.eval()
    model.load_state_dict(torch.load(CHECKPOINT_DIR + checkpoint, map_location="cpu"))
    return model


def main(
    DATASET_DIR="../input/data/ICDAR17_Korean/", CHECKPOINT_DIR="./trained_models/"
):
    """
    Args:
        DATASET_DIR (str, optional): ufo,images폴더의 상위폴더를 넣어주세요.
        CHECKPOINT_DIR (str, optional): 저장된 모델의 checkpoint 경로를 넣어주세요
    """
    DATA_JSON = st.selectbox("확인할 data_json파일을 선택하세요", os.listdir(DATASET_DIR + "ufo/"))
    ufo_fpath = osp.join(DATASET_DIR, "ufo/", DATA_JSON)
    with open(ufo_fpath, "r") as f:
        ufo_anno = json.load(f)
    sample_ids = sorted(ufo_anno["images"])

    SAMPLE_IDX = st.slider("SAMPLE_IDX", 0, len(sample_ids), 0, 1)
    sample_id = sample_ids[SAMPLE_IDX]  # `sample_id`가 곧 이미지 파일명
    image_fpath = osp.join(DATASET_DIR, "images", sample_id)
    image = imread(image_fpath)

    tab1, tab2 = st.tabs(["img_View", "developing..."])
    with tab1:
        bboxes, labels = [], []
        for word_info in ufo_anno["images"][sample_id]["words"].values():
            bboxes.append(np.array(word_info["points"]))
            labels.append(int(not word_info["illegibility"]))
        bboxes, labels = np.array(bboxes, dtype=np.float32), np.array(
            labels, dtype=np.float32
        )

        st.write("Image shape:\t{}".format(image.shape))
        st.write("Bounding boxes shape:\t{}".format(bboxes.shape))
        st.write("Labels shape:\t{}".format(labels.shape))

        vis = image.copy()
        draw_bboxes(
            vis,
            bboxes,
            double_lined=True,
            thickness=2,
            thickness_sub=5,
            write_point_numbers=True,
        )
        st.image(vis)

    with tab2:
        checkpoint = st.selectbox(
            "model의 checkpoint를 골라주세요.", os.listdir(CHECKPOINT_DIR)
        )
        if "checkpoint" not in st.session_state:
            st.session_state.checkpoint = checkpoint
            st.session_state.model = load_model(CHECKPOINT_DIR, checkpoint)
        if checkpoint != st.session_state.checkpoint:
            st.session_state.checkpoint = checkpoint
            st.session_state.model = load_model(CHECKPOINT_DIR, checkpoint)

        bboxes = detect(st.session_state.model, [image], 1024)[0]
        vis = image.copy()
        draw_bboxes(vis, bboxes, thickness=2)
        st.image(vis)


if __name__ == "__main__":
    main()
