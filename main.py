from my_opencv import *


def test(img):
    return basic_process.get_channel(img, 1)


if __name__ == '__main__':
    img_list = ['dog.jpg',                     # 0
                'pie.png',                     # 1
                'rectangle.png',               # 2
                'gradient.png',                # 3
                'grandma.jpg',                 # 4
                ''
                ]
    img = img_io.read_img(img_list[4])
    basic_process.smooth(img)

# basic_process.threshold(img)
# basic_process.rotate(img)
# img_io.make_video_from_cam('name', test)
