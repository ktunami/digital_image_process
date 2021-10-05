# @Time : 2021/9/26 12:21 
# @Author : Kate
# @File : img_io.py
# @Software: PyCharm

import cv2
import os
from my_opencv import Path


def show_img(title, image, quit_val='q'):
    """
    展示图片，按q退出
    :param title: 标题
    :param image: 图片
    :param quit_val: 退出按钮
    """
    cv2.namedWindow(title)
    cv2.imshow(title, image)
    print('按q结束')
    while 1:
        k = cv2.waitKey(0)
        if k & 0xFF == ord(quit_val):
            break
    cv2.destroyWindow('title')


def read_img(image_name):
    """
    返回读取的图片
    :param image_name: 图片名
    :return: 图片
    """
    path = Path.read_img_path(image_name)
    img = cv2.imread(path)
    print('read img: ', path)
    print('size: ', img.size)
    print('shape: ', img.shape)
    print('dtype: ', img.dtype)
    return img


def save_img(img, filename):
    """
    存图片
    :param img: 图片
    :param filename: 图片名
    """
    path = Path.output_img_path(filename)
    cv2.imwrite(path, img)


def make_video_from_cam(name, callback=None):
    """
    摄像头录视频并存储
    :param name: 存储的视频文件名
    :param callback: 回调函数，用于对每帧视频进行处理
    :return:
    """
    cap = cv2.VideoCapture(0)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(Path.output_img_path(name), fourcc, 20.0, (640, 480))
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            if callback:
                frame = callback(frame)
            out.write(frame)
            cv2.imshow(name, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # q
            break
    cap.release()
    out.release()
    cv2.destroyAllWindows()