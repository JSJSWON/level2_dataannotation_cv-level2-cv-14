import numpy as np
import random
from PIL import Image, ImageDraw
from straug.straug.pattern import VGrid


def my_aug(img, vertices):
    """
    straug 폴더에서 원하는 aug를 가져와서 augs의 리스트에 추가하여 준다.
    그 중에서 랜덤으로 하나의 aug를 뽑은 후 0.5의 확률로 aug를 적용
    Args:
        img (PIL img): aug가 적용되기 이전의 img
        vertices: box의 좌표
    Returns:
        PIL img: aug가 적용된 img 반환 (n,8)
    """
    rng = np.random.default_rng(2022)
    augs = [
        VGrid(rng),
    ]
    aug = random.choice(augs)
    aug_img = aug(img, prob=0.5)

    if img.mode != "RGB":
        img = img.convert("RGB")
    if aug_img.mode != "RGB":
        aug_img = aug_img.convert("RGB")

    img, aug_img = np.array(img), np.array(aug_img)
    mask_img = Image.new("1", (img.shape[1], img.shape[0]), 1)

    for vertice in vertices:
        poligon = [(i, j) for i, j in zip(vertice[::2], vertice[1::2])]
        ImageDraw.Draw(mask_img).polygon(poligon, outline=1, fill=0)
    mask = np.array(mask_img)
    # Image.fromarray(mask,"L").save("./valid.jpg")
    for i in range(3):
        img[:, :, i] = img[:, :, i] * mask
        aug_img[:, :, i] = aug_img[:, :, i] * (1 - mask)
    # Image.fromarray(img + aug_img,"RGB").save("./valid2.jpg")
    return img + aug_img
