import numpy as np
from src.data import CENTER, NORM_MEAN, NORM_STD

def get_avg_value_and_distance(img_yx):
    assert img_yx.shape[0] == img_yx.shape[1]
    img_yx = img_yx.copy()
    img_yx[img_yx < 0] = 0

    coor_y = 0.5 + np.arange(img_yx.shape[0])

    def get_offset_and_var(axis):
        """pass axis=1 to get stats for y and axis=0 for x"""
        # assuming axis == 1
        p_y = img_yx.sum(axis)
        p_y /= p_y.sum()
        avg_y = (p_y * coor_y).sum()
        var_y = (p_y * (coor_y - avg_y)**2).sum()
        offset_y = avg_y - coor_y.size / 2
        return offset_y, var_y

    offset_x, var_x = get_offset_and_var(0)
    offset_y, var_y = get_offset_and_var(1)


    return img_yx.sum(), offset_x, offset_y, offset_x**2 + offset_x**2, var_x + var_y

def normalize(img):
    return (img / 255 - NORM_MEAN) / NORM_STD

def denormalize(img):
    return (255 * (NORM_STD * img + NORM_MEAN)).astype(int)