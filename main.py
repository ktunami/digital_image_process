from my_opencv import *


def test(img):
    return basic_process.get_channel(img, 1)


if __name__ == '__main__':
    img_list = ['dog.jpg',                     # 0
                'pie.png',                     # 1
                'rectangle.png',               # 2
                'gradient.png',                # 3
                'grandma.jpg',                 # 4
                'affine.png',                  # 5
                'fourier.png',                 # 6
                'girl.png',                    # 7
                'salt.png',                    # 8
                'test.png',                    # 9
                'mri.png',                     # 10
                'lines.png',                   # 11
                'lines2.png',                  # 12
                'dip.png',                     # 13
                ''
                ]
    img = img_io.read_img(img_list[7])
    # img_io.show_img('origin', img)
    chp4.filters_test()

# basic_process.threshold(img)
# basic_process.rotate(img)
# img_io.make_video_from_cam('name', test)
