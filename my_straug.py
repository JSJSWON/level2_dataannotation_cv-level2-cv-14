import numpy as np
import random
from straug.straug.pattern import VGrid


def my_aug(img):
    """
    straug 폴더에서 원하는 aug를 가져와서 augs의 리스트에 추가하여 준다.
    그 중에서 랜덤으로 하나의 aug를 뽑은 후 0.5의 확률로 aug를 적용
    Args:
        img (PIL img): aug가 적용되기 이전의 img

    Returns:
        PIL img: aug가 적용된 img 반환
    """
    rng = np.random.default_rng(2022)
    augs = [
        VGrid(rng),
    ]
    aug = random.choice(augs)
    return aug(img, prob=0.5)
