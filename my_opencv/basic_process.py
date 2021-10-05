# @Time : 2021/9/26 14:48 
# @Author : Kate
# @File : basic_process.py 
# @Software: PyCharm
from math import floor

import cv2
from my_opencv import img_io
import numpy as np
import re


def get_channel(img, channel, show=False):
    """
    通道分离，split比numpy操作慢
    :param img: 输入图像
    :param channel: 选择什么通道返回
    :param show: 是否要展示各个通道图像
    :return:
    """
    b, g, r = cv2.split(img)
    zeros = np.zeros(img.shape[:2], dtype="uint8")
    if show:
        img_io.show_img("BLUE", cv2.merge([b, zeros, zeros]))
        img_io.show_img("GREEN", cv2.merge([zeros, g, zeros]))
        img_io.show_img("RED", cv2.merge([zeros, zeros, r]))
        img_io.show_img('whole', cv2.merge([b, g, r]))
    channels = [cv2.merge([b, zeros, zeros]), cv2.merge([zeros, g, zeros]), cv2.merge([zeros, zeros, r])]
    return channels[channel]


def set_border(img):
    """
    扩展边缘，4个数分别是上下左右
    :param img: 输入图像
    :return:
    """
    paras = ['cv2.' + nm for nm in dir(cv2) if 'BORDER' in nm]
    for string in paras:
        para = eval(string)
        new_img = cv2.copyMakeBorder(img, 100, 200, 300, 400, para)
        img_io.show_img(string, new_img)


def resize_img(img):
    """
    图像的放缩
    :param img: 输入图像
    """
    paras = ['cv2.' + i for i in dir(cv2) if 'INTER_' in i and len(i.split('_')) == 2]
    for string in paras:
        para = eval(string)
        new_img = cv2.resize(img, None, fx=3, fy=3, interpolation=para)
        new_img1 = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=para)
        img_io.show_img(string, new_img)
        img_io.show_img(string, new_img1)


def affine_transformation(img):
    """
    仿射变换 -> 传3个点  cv2.getAffineTransform()
    透视变换 -> 传4个点  cv2.getPerspectiveTransform()
               这4个点中任意3个都不能共线
    :param img: 输入图像
    """
    rows, cols = img.shape[:2]
    src_3_points = np.float32([[5, 5], [20, 5], [30, 50]])
    tar_3_points = np.float32([[10, 20], [20, 10], [40, 70]])
    M = cv2.getAffineTransform(src_3_points, tar_3_points)
    res = cv2.warpAffine(img, M, (rows, cols))
    img_io.show_img('affine transform', res)


def rotate(img):
    """
    旋转图像-> 中心，角度，放缩
    :param img: 输入图像
    """
    rows, cols = img.shape[:2]
    img_io.show_img('origin', img)
    scale = 1.1
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 10, scale)
    dst = cv2.warpAffine(img, M, (floor(scale * cols), floor(scale * rows)))
    img_io.show_img('rotate', dst)


def threshold(img):
    """
    阈值化
    -->自适应的用
    # 11：邻域大小    2：C 常数，被平均或加权平均值减去，然后就的到阈值
     th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C , cv2.THRESH_BINARY,11,2 )
     th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C , cv2.THRESH_BINARY,11,2)
    :param img: 输入图像
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    pattern = re.compile(r'^(THRESH_)')
    paras = ['cv2.' + nm for nm in dir(cv2) if pattern.match(nm)]
    img_io.show_img('origin', gray, 'o')
    for string in paras:
        para = eval(string)
        if string == 'cv2.THRESH_OTSU':
            ret, new_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            ret, new_img = cv2.threshold(gray, 127, 255, para)
        img_io.show_img(string, new_img)


def smooth(img):
    """
    图像平滑
    :param img: 输入图像
    """
    kernel = np.ones((5, 5), np.float32) / 25
    img2 = cv2.filter2D(img, -1, kernel)
    img3 = cv2.blur(img, (5, 5))              # 和上面那个效果一样的
    img4 = cv2.GaussianBlur(img, (5, 5), 0)
    img5 = cv2.medianBlur(img, 5)             # 这个适合过滤椒盐噪声
    img6 = cv2.bilateralFilter(img, 9, 75, 75)  # 后面两个参数是空间高斯函数标准差和灰度值相似性高斯函数标准差
    img_io.show_img('origin', img)
    img_io.show_img('smooth', img2)
    img_io.show_img('blur', img3)
    img_io.show_img('GaussianBlur', img4)
    img_io.show_img('medianBlur', img5)
    img_io.show_img('bilateralFilter', img6)


