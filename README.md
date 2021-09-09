# 来源

https://github.com/LiuXinyu12378/yolov4-tiny-pytorch.git

# 特点

速度快，轻量级目标检测及分类，且模型部分做了onnx转换，onnx转换成mnn后可直接在arm嵌入式上支持推理

# 效果



![img1](result.jpg)

# 应用

## 1.应用场景

ARM开发板或CPU快速目标检测分类推理

## 2.输入输出

#### 输入：

{"imageId": "xxxx", "base64Data":"xxxx", "format": "jpg", "url":"xxxxx"}

#### 输出：

{"status":0, "message": "success", "result": [[y1,x1,y2,x2,"类别","类别分数"],[...],[...],,,], 'target': 'yes'}}

##### 

### 3.数据集

coco2017

train2017_img http://images.cocodataset.org/zips/train2017.zip

train2017_ann http://images.cocodataset.org/annotations/annotations_trainval2017.zip

val2017_img http://images.cocodataset.org/zips/val2017.zip

val2017_ann http://images.cocodataset.org/annotations/stuff_annotations_trainval2017.zip

test2017_img http://images.cocodataset.org/zips/test2017.zip

test2017_list http://images.cocodataset.org/annotations/image_info_test2017.zip

