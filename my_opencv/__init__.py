# @Time : 2021/9/26 12:48 
# @Author : Kate
# @File : __init__.py.py 
# @Software: PyCharm

import os


class Path(object):
    __src_path = '/Users/kate/Desktop/'
    __tar_path = '/Users/kate/Desktop/'
    __src_folder = 'src_imgs'
    __tar_folder = 'output_imgs'

    @staticmethod
    def make_dir(path):
        if not os.path.exists(path):
            os.makedirs(path)
            print(path, ' created')
        else:
            print(path, ' already exists')

    @classmethod
    def init_path(cls):
        Path.make_dir(cls.__src_path + cls.__src_folder)
        Path.make_dir(cls.__tar_path + cls.__tar_folder)

    @classmethod
    def read_img_path(cls, image_name):
        return cls.__src_path + cls.__src_folder + os.sep + image_name

    @classmethod
    def output_img_path(cls, image_name):
        return cls.__tar_path + cls.__tar_folder + os.sep + image_name


Path.init_path()
__all__ = ['basic_process', 'img_io']