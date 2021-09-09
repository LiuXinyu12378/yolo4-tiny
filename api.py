import requests
import json
import base64
import os
import numpy as np
import math
import cv2
import sys
import time

from PIL import ImageFont

API_CATEGORY = os.environ.get("API_CATEGORY", "cv")
API_NAME = os.environ.get("API_NAME", "yolov4_tiny")
API_VERSION = os.environ.get("API_VERSION", "1.0")
DEUBG_MODE = os.environ.get("DEBUG_MODE", "True")

API_URI = '/%s/%s/%s' % (API_CATEGORY, API_NAME, API_VERSION)
Healthy_URI = '%s/%s' % (API_URI, "healthy")

# print(API_URI)

def pic2base64(image_path):

    with open(image_path, 'rb') as f:
        image = f.read()
    image_base64 = base64.b64encode(image)
    image_base64 = image_base64.decode()

    return image_base64


def post_request(imageId,image_path,url=None):

    url_ = 'http://127.0.0.1:8080'+API_URI
    base64Data = pic2base64(image_path)
    format = os.path.splitext(image_path)[-1].replace(".","")
    url = ""

    data = {"imageId": imageId,
            "base64Data": base64Data,
            "format": format,
            "url": url}

    data = json.dumps(data)
    res = requests.post(url_, data=data).text
    res = json.loads(res)

    boxes_class = eval(res["result"])
    img = cv2.imread(image_path)

    if boxes_class:

        for y1,x1,y2,x2,cls,score in boxes_class:
            cv2.rectangle(img,(x1,y1),(x2,y2),color=(0,0,255),thickness=2)
            cv2.rectangle(img,(x1,y1-40),(x1+150,y1),color=(0,0,255),thickness=-1)
            cv2.putText(img, cls, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)

    cv2.namedWindow("img",cv2.WINDOW_NORMAL)
    cv2.imshow("img",img)
    cv2.imwrite("result.jpg",img)
    cv2.waitKey(0)
    print(boxes_class,type(boxes_class))


if __name__ == '__main__':
    post_request("00001","img/street.jpg")
