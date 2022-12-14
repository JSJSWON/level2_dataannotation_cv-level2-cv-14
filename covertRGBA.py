import os
import os.path as osp

from glob import glob
from PIL import Image


def convert_rgba(PATH):
    cnt = 0
    for image in os.listdir(PATH):
        image_fpath = osp.join(PATH, image)
        im = Image.open(image_fpath)
        # If is png image
        if im.format == 'PNG':
            # and is not RGBA
            if im.mode != 'RGBA':
                cnt += 1
                print(image_fpath)
                im.convert("RGBA").save(image_fpath)

    print(f"{cnt} images converted!")


if __name__ == "__main__":
    # 이미지가 들어있는 경로 설정
    PATH = "/opt/ml/input/data/ICDAR_1719_v2/images"
    convert_rgba(PATH)