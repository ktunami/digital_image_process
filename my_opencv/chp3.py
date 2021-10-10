# @Time : 2021/10/5 11:07 
# @Author : Kate
# @File : chp3.py
# @Software: PyCharm

"""
1. 不同的仿射变换矩阵得到不同的图像几何变换结果
2. 基本的灰度变换
    图像反转
    对数变换
    r变换
    分段函数变换
    比特平面分层
3. 画直方图，直方图均衡
4. 各种滤波: 均值滤波，中值滤波，高斯滤波
5. 图像增强-laplacian operators-分别用90度和45度各向同性的核
6. 非锐化掩蔽和高提升滤波
7. 边缘检测：
     Roberts cross-gradient operators.
     Sobel operators
     Prewitt operators
     Canny operators
"""

import math
import numpy as np
from my_opencv.img_io import *
import matplotlib.pyplot as plt


def show_affine_result(mat, img, title):
    """
    展示应用仿射变换的结果
    :param mat: 仿射变换矩阵
    :param img: 原始图像
    :param title: 标题
    """
    rows, cols, channel = img.shape
    trans_mat = np.array(mat, dtype=np.float32)
    img2 = cv2.warpPerspective(img, trans_mat, (2 * cols, 2 * rows))
    show_img(title, img2)


def gamma(img, v):
    """
    gama变换
    :param img:原始图像
    :param v: 参数（指数）
    :return:
    """
    lut = np.zeros(256, dtype=np.float32)
    for i in range(256):
        lut[i] = 255 / (255 ** v) * i ** v
    output_img = cv2.LUT(img, lut)
    output_img = np.uint8(output_img + 0.5)
    return output_img


def piecewise_linear_contrast_stretch(img, pt1, pt2):
    """
    线性分段函数变换：对比度拉伸
    :param img: 原始图像
    :param pt1: 第一个点（输入，输出）
    :param pt2: 第二个点（输入，输出）
    :return: 返回变换图像
    """
    if pt1[0] >= pt2[0]:
        print('pt1的输入必须小于pt2的输入')
    elif pt1[0] == 0:
        print('pt1的输入不能是0')
    else:
        lut = np.arange(256, dtype=np.float32)
        lut[:pt1[0] + 1] = lut[:pt1[0] + 1] * pt1[1] / pt1[0]
        lut[pt1[0] + 1: pt2[0] + 1] = lut[pt1[0] + 1: pt2[0] + 1] * (pt2[1] - pt1[1]) / (pt2[0] - pt1[0])
        lut[pt2[0] + 1:] = lut[pt2[0] + 1:] * (255 - pt2[1]) / (255 - pt2[0])
        output_img = cv2.LUT(img, lut)
        output_img = np.uint8(output_img)
        return output_img


def piecewise_linear_intensity_slicing(img, rg, val, remain=None):
    """
    线性分段函数变换：灰度级分割
    :param img: 原始图像
    :param rg: 灰度提取范围（含2个数的元祖）
    :param val: 提取部分的值
    :param remain: 其余部分取值（为None则保持原状）
    :return: 处理后图像
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lut = np.arange(256, dtype=np.float32)
    lut[rg[0]:rg[1] + 1] = val
    if remain is not None:
        lut[:rg[0]] = remain
        lut[rg[1] + 1:] = remain
    output_img = cv2.LUT(gray, lut)
    output_img = np.uint8(output_img)
    return output_img


def img_hist(img, ti, is_gray=False):
    """
    画三通道图像的直方图
    """
    if is_gray:
        plt.hist(img.ravel(), 256, [0, 256])
        plt.title(ti)
        plt.show()
    else:
        color = ('b', 'g', 'r')
        for i, color in enumerate(color):
            hist = cv2.calcHist([img], [i], None, [256], [0, 256])
            plt.plot(hist, color)
            plt.xlim([0, 256])
            plt.title(ti)
        plt.show()


def normalize_hist(img):
    """
    直方图均衡
    """
    (b, g, r) = cv2.split(img)
    nb = cv2.equalizeHist(b)
    ng = cv2.equalizeHist(g)
    nr = cv2.equalizeHist(r)
    new_img = cv2.merge((nb, ng, nr))
    return new_img


def max_min_filtering(img, ksize, is_min):
    """
    最大最小值滤波器
    :param img: 图像
    :param ksize: 核尺寸
    :param is_min: 是否是最小滤波器
    """
    n = (ksize // 2)
    rows, cols, channels = img.shape
    new_img = np.zeros(img.shape, dtype=np.uint8)  # 注意：必须要记得指明类型！
    for i in range(channels):
        if is_min:
            extended = np.full((rows + n * 2, cols + n * 2), 255, dtype=np.uint8)
            extended[n:rows + n, n:cols + n] = img[:, :, i]
            for y in range(rows):
                for x in range(cols):
                    new_img[y, x, i] = np.min(extended[y:y + ksize, x:x + ksize])
        else:
            extended = np.zeros((rows + n * 2, cols + n * 2), dtype=np.uint8)
            extended[n:rows + n, n:cols + n] = img[:, :, i]
            for y in range(rows):
                for x in range(cols):
                    new_img[y, x, i] = np.max(extended[y:y + ksize, x:x + ksize])
    return new_img


def get_abs_filter2d(img, kernel):
    """
    做图像相关，负值转为正值
    """
    s_img = cv2.filter2D(img, cv2.CV_16S, kernel)
    u_img = cv2.convertScaleAbs(s_img)
    return u_img


def edge_detect_algo(img, kernel_x, kernel_y, title):
    """
    边缘检测通用模版
    :param img: 待处理图像
    :param kernel_x: x核
    :param kernel_y: y核
    :param title: 展示标题
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edge_x = get_abs_filter2d(gray, kernel_x)
    edge_y = get_abs_filter2d(gray, kernel_y)
    add_direct = cv2.add(edge_x, edge_y)
    add_weighted = cv2.addWeighted(edge_x, 0.5, edge_y, 0.5, 0)  # 用这个
    show_img('gray', gray)
    show_img(title + ' x', edge_x)
    show_img(title + ' y', edge_y)
    show_img(title + ' add direct', add_direct)
    show_img(title + ' add weighted', add_weighted)


# ----------------------------------------------------------------------------- #
# 1. 不同的仿射变换矩阵得到不同的图像几何变换结果
def effect_of_affine_mat(img):
    """
    不同的仿射变换矩阵得到不同的图像几何变换结果  5 affine.png
    warpAffine 接受的是2*2的变换矩阵
    warpPerspective 接受的是3*3的
    变换矩阵类型是浮点数
    """
    rows, cols, channel = img.shape

    # 1. scaling
    mat1 = [[2, 0, 0],
            [0, 1, 0],
            [0, 0, 1]]
    # show_affine_result(mat1, img, 'scale(2x)')

    # 2. Translation
    mat2 = [[1, 0, cols],
            [0, 1, rows],
            [0, 0, 1]]
    # show_affine_result(mat2, img, 'Translation')

    # 3. rotation 以左上角为中心旋转
    ang = -45
    mat3 = [[np.cos(ang), -np.sin(ang), 0],
            [np.sin(ang), np.cos(ang), math.floor(0.5 * rows * 1.414)],
            [0, 0, 1]]
    # show_affine_result(mat3, img, 'rotation(45)')

    # 4. rotation inner 可自定义中心旋转
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 45, 1)
    print(mat3)
    print(M)
    dst = cv2.warpAffine(img, M, (2 * cols, 2 * rows))
    # show_img('rotation inner', dst)

    # 5. Shear(horizontal)
    mat5 = [[1, 0.5, 0],
            [0, 1, 0],
            [0, 0, 1]]
    show_affine_result(mat5, img, 'Shear(horizontal)')

    # 6. Shear(vertical)
    mat6 = [[1, 0, 0],
            [0.5, 1, 0],
            [0, 0, 1]]
    show_affine_result(mat6, img, 'Shear(vertical)')


# 2. 基本的灰度变换
# 图像反转
def image_negatives(img):
    """
    图像取反 5 affine.png
    """
    show_img('Img inverse', 255 - img)


# 对数变换
def log_transform(img):
    """
    图像取对数 6 fourier.png
    """
    new_img = 32 * np.log(1.0 + img)
    new_img = np.uint8(new_img + 0.5)
    show_img('log transform', new_img)


# r变换
def gama_transform(img):
    """
    图像的gama变换 7 girl.png
    """
    img1 = gamma(img, 0.4)
    show_img('gama transform r=0.4', img1)
    img2 = gamma(img, 2.5)
    show_img('gama transform r=2.5', img2)


# 分段函数变换
def piecewise_linear_transform(img):
    """
    线性分段函数变换 7 girl.png
    """
    img1 = piecewise_linear_contrast_stretch(img, (30, 10), (225, 245))
    show_img('piecewise_linear_contrast_stretch 30,10', img1)
    img2 = piecewise_linear_contrast_stretch(img, (40, 10), (215, 245))
    show_img('piecewise_linear_contrast_stretch 40,10', img2)
    img3 = piecewise_linear_intensity_slicing(img, (100, 150), 255)
    show_img('piecewise_linear_intensity_slicing 100, 150 REMAIN', img3)
    img4 = piecewise_linear_intensity_slicing(img, (100, 150), 255, 0)
    show_img('piecewise_linear_intensity_slicing 100, 150', img4)


# 比特平面分层
def bit_plane_slicing(img):
    """
    比特平面图 7 girl.png
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    row, col = gray.shape
    new_img = np.zeros((row, col, 8))
    for i in range(row):
        for j in range(col):
            n = str(np.binary_repr(gray[i, j], 8))
            for k in range(8):
                new_img[i, j, k] = n[k]
    show_img('origin gray pic', gray)
    for i in range(8):
        show_img('bit_plane_slicing' + str(8 - i), new_img[:, :, i])


# 3. 画直方图，直方图均衡
def img_hist_op(img):
    """
    直方图相关 7 girl.png
    """
    img_hist(img, "original pic")
    new_img = normalize_hist(img)
    img_hist(new_img, "normalize_hist")
    show_img('original pic', img)
    show_img('normalize_hist', new_img)


# 4. 各种滤波: 均值滤波，中值滤波，高斯滤波
def smooth(img):
    """
    图像平滑  'rectangle.png'   # 2
    """
    kernel = np.ones((5, 5), np.float32) / 25
    img2 = cv2.filter2D(img, -1, kernel)
    img3 = cv2.blur(img, (5, 5))  # 和上面那个效果一样的
    img4 = cv2.GaussianBlur(img, (5, 5), 0)
    img5 = cv2.medianBlur(img, 5)  # 这个适合过滤椒盐噪声
    img6 = cv2.bilateralFilter(img, 9, 75, 75)  # 后面两个参数是空间高斯函数标准差和灰度值相似性高斯函数标准差
    img7 = max_min_filtering(img, 3, True)
    img8 = max_min_filtering(img, 3, False)
    show_img('origin', img)
    show_img('smooth', img2)
    show_img('blur', img3)
    show_img('GaussianBlur', img4)
    show_img('medianBlur', img5)
    show_img('bilateralFilter', img6)
    show_img('min filter', img7)
    show_img('max filter', img8)


# 5. 图像增强-laplacian-分别用90度和45度各向同性的核
# 'mri.png',  # 10
# 'lines.png',  # 11
# 'pie.png',  # 1
def laplacian_enhancement(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    show_img('gray', gray)
    kernel90 = np.array([[0, 1, 0],
                         [1, -4, 1],
                         [0, 1, 0]])
    kernel45 = np.array([[1, 1, 1],
                         [1, -8, 1],
                         [1, 1, 1]])
    la_img_90 = cv2.filter2D(gray, -1, kernel90)
    sharp_90 = cv2.subtract(gray, la_img_90)
    la_img_45 = cv2.filter2D(gray, -1, kernel45)
    sharp_45 = cv2.subtract(gray, la_img_45)
    show_img('laplacian 90', la_img_90)
    show_img('sharp 90', sharp_90)
    show_img('laplacian 45', la_img_45)
    show_img('sharp 45', sharp_45)


# 6. 非锐化掩蔽和高提升滤波
def unsharp_mask_highboost(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (21, 21), 5)
    mask = cv2.subtract(gray, blur)
    unsharp_mask = cv2.add(gray, mask)
    factor = 8 * np.full((mask.shape[0], mask.shape[1]), 1, dtype=np.uint8)
    factor = cv2.multiply(mask, factor)
    highboost_filter = cv2.add(gray, factor)
    show_img('gray', gray)
    show_img('blur 21*21 5 gaussian', blur)
    show_img('mask', mask)
    show_img('unsharp_mask', unsharp_mask)
    show_img('highboost_filter 8', highboost_filter)


# 7. 边缘检测：
# 'pie.png',    # 1
# 'girl.png',   # 7
def edge_detection_robert(img):
    #      Roberts cross-gradient operators.
    robert_kernel_x = np.array([[0, 0, 0],
                                [0, -1, 0],
                                [0, 0, 1]])
    robert_kernel_y = np.array([[0, 0, 0],
                                [0, 0, -1],
                                [0, 1, 0]])
    # edge_detect_algo(img, robert_kernel_x, robert_kernel_y, 'Roberts')
    #      Sobel operators
    sobel_kernel_x = np.array([[-1, -2, -1],
                               [0, 0, 0],
                               [1, 2, 1]])
    sobel_kernel_y = np.array([[-1, 0, 1],
                               [-2, 0, 2],
                               [-1, 0, 1]])
    # edge_detect_algo(img, sobel_kernel_x, sobel_kernel_y, 'Sobel')
    #      Prewitt operators
    sobel_kernel_x = np.array([[-1, -1, -1],
                               [0, 0, 0],
                               [1, 1, 1]])
    sobel_kernel_y = np.array([[-1, 0, 1],
                               [-1, 0, 1],
                               [-1, 0, 1]])
    # edge_detect_algo(img, sobel_kernel_x, sobel_kernel_y, 'Prewitt')
    #      Canny operators
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    canny = cv2.Canny(gray, 20, 100)
    show_img('gray', gray)
    show_img('canny 20, 100', canny)



