# @Time : 2021/10/10 14:29 
# @Author : Kate
# @File : chp4.py 
# @Software: PyCharm

"""
1. 高通、低通、带阻、带通
"""

import math
import numpy as np
from my_opencv.img_io import *
import matplotlib.pyplot as plt


def zone_plate(size, lim):
    """
    获得 zone plate图像，用于测试过滤频率
    :param size: 图像尺寸
    :param lim: 范围
    :return: zone plate图像
    """
    arr = np.linspace(-lim, lim, size)
    shift = size // 2
    shift = arr[shift]
    plate_img = np.zeros((size, size))
    for x, i in zip(np.linspace(-lim, lim, size), range(size)):
        for y, j in zip(np.linspace(-lim, lim, size), range(size)):
            a = x - shift
            b = y - shift
            plate_img[i, j] = 0.5 * (1 + np.cos(a * a + b * b))
    return cv2.convertScaleAbs(plate_img, alpha=255)

# --------------------------------------------------------------------------------------- #


# 8. 高通、低通、带阻、带通
def filters_test():
    img = zone_plate(599, 10)
    show_img('gray', img)
