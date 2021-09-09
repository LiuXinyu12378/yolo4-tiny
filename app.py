from yolo import YOLO
from PIL import Image
from flask import Flask,request,jsonify
import json
import os
import base64
import cv2
import numpy as np

yolo = YOLO()


CUDA = False
app = Flask(__name__)
API_CATEGORY = os.environ.get("API_CATEGORY", "cv")
API_NAME = os.environ.get("API_NAME", "yolov4_tiny")
API_VERSION = os.environ.get("API_VERSION", "1.0")
DEUBG_MODE = os.environ.get("DEBUG_MODE", "True")

API_URI = '/%s/%s/%s' % (API_CATEGORY, API_NAME, API_VERSION)
Healthy_URI = '%s/%s' % (API_URI, "healthy")

def base64_2_pic(base64data):
    base64data = base64data.encode(encoding="utf-8")
    data = base64.b64decode(base64data)
    imgstring = np.array(data).tostring()
    imgstring = np.asarray(bytearray(imgstring), dtype="uint8")
    image = cv2.imdecode(imgstring, cv2.IMREAD_COLOR)

    return image


@app.route(API_URI, methods=["POST"])
def interface():
    data = request.get_json()
    if data:
        pass
    else:
        data = request.get_data()
        data = json.loads(data)

    imageId = data["imageId"]
    base64Data = data["base64Data"]
    format = data["format"]
    url = data["url"]

    image = base64_2_pic(base64Data)
    image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    result = yolo.detect_image_not_draw(image)
    data = {
        "status": 0,
        "message": "success",
        "result":str(result)
    }
    return data

if __name__ == '__main__':
    app.run(host="0.0.0.0",port=8080)
