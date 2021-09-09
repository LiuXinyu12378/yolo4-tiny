#-------------------------------------#
#       对单张图片进行预测
#-------------------------------------#
from yolo import YOLO
from PIL import Image

yolo = YOLO()


# img = input('Input image filename:')
img = "VOCdevkit/fall_detection/images/fall_1.jpg"
try:
    image = Image.open(img)
except:
    print('Open Error! Try again!')
else:
    r_image = yolo.detect_image(image)
    r_image.show()
