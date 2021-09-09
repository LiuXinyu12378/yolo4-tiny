import torch
import torch.nn as nn
from collections import OrderedDict
from nets.CSPdarknet53_tiny import darknet53_tiny
import json
import numpy as np

# -------------------------------------------------#
#   卷积块
#   CONV+BATCHNORM+LeakyReLU
# -------------------------------------------------#
from utils.utils2 import DecodeBox, non_max_suppression

anchors = [[10,14],[23,27],[37,58],[81,82],[135,169],[344,319]]



class BasicConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(BasicConv, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, kernel_size // 2, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x


# ---------------------------------------------------#
#   卷积 + 上采样
# ---------------------------------------------------#
class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Upsample, self).__init__()

        self.upsample = nn.Sequential(
            BasicConv(in_channels, out_channels, 1),
            nn.Upsample(scale_factor=2, mode='nearest')
        )

    def forward(self, x, ):
        x = self.upsample(x)
        return x


# ---------------------------------------------------#
#   最后获得yolov4的输出
# ---------------------------------------------------#
def yolo_head(filters_list, in_filters):
    m = nn.Sequential(
        BasicConv(in_filters, filters_list[0], 3),
        nn.Conv2d(filters_list[0], filters_list[1], 1),
    )
    return m

def process(input,anchors):
    # input为bs,3*(1+4+num_classes),13,13

    # 一共多少张图片
    batch_size = input.size(0)
    # 13，13
    input_height = input.size(2)
    input_width = input.size(3)

    # 计算步长
    # 每一个特征点对应原来的图片上多少个像素点
    # 如果特征层为13x13的话，一个特征点就对应原来的图片上的32个像素点
    # 416/13 = 32
    stride_h = 416 / input_height
    stride_w = 416 / input_width

    # 把先验框的尺寸调整成特征层大小的形式
    # 计算出先验框在特征层上对应的宽高
    scaled_anchors = []
    for anchor_width, anchor_height in anchors:
        anchor_width = torch.tensor(anchor_width,dtype=torch.float64)
        anchor_height = torch.tensor(anchor_height,dtype=torch.float64)
        anchor_width_ = anchor_width.div(stride_w)
        anchor_height_ = anchor_height.div(stride_h)
        # anchor_width_ = float(anchor_width) / stride_w
        # anchor_height_ = float(anchor_height) / stride_h
        scaled_anchors.append([anchor_width_, anchor_height_])
    # scaled_anchors = [(anchor_width / stride_w, anchor_height / stride_h) for anchor_width, anchor_height in self.anchors]

    # bs,3*(5+num_classes),13,13 -> bs,3,13,13,(5+num_classes)
    prediction = input.view(batch_size, 3,
                            85, input_height, input_width).permute(0, 1, 3, 4, 2).contiguous()

    # 先验框的中心位置的调整参数
    x = torch.sigmoid(prediction[..., 0])
    y = torch.sigmoid(prediction[..., 1])
    # 先验框的宽高调整参数
    w = prediction[..., 2]  # Width
    h = prediction[..., 3]  # Height

    # 获得置信度，是否有物体
    conf = torch.sigmoid(prediction[..., 4])
    # 种类置信度
    pred_cls = torch.sigmoid(prediction[..., 5:])  # Cls pred.

    FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor

    # 生成网格，先验框中心，网格左上角 batch_size,3,13,13

    grid_x = torch.tensor(range(input_width))
    grid_y = torch.tensor(range(input_height))
    grid_x = grid_x.repeat(input_width, 1).repeat(batch_size * 3, 1, 1).view(x.shape).type(FloatTensor)
    grid_y = grid_y.repeat(input_height, 1).t().repeat(batch_size * 3, 1, 1).view(y.shape).type(FloatTensor)
    # grid_x = torch.linspace(0, input_width - 1, input_width).repeat(input_width, 1).repeat(batch_size * 3, 1, 1).view(x.shape).type(FloatTensor)
    # grid_y = torch.linspace(0, input_height - 1, input_height).repeat(input_height, 1).t().repeat(batch_size * 3, 1, 1).view(y.shape).type(FloatTensor)

    # if input_width == 13:
    #     grid_x = grid[0]
    #     grid_y = grid[1]
    # if input_width == 26:
    #     grid_x = grid2[0]
    #     grid_y = grid2[1]

    # 生成先验框的宽高
    anchor_w = FloatTensor(scaled_anchors).index_select(1, LongTensor([0]))
    anchor_h = FloatTensor(scaled_anchors).index_select(1, LongTensor([1]))
    anchor_w = anchor_w.repeat(batch_size, 1).repeat(1, 1, input_height * input_width).view(w.shape)
    anchor_h = anchor_h.repeat(batch_size, 1).repeat(1, 1, input_height * input_width).view(h.shape)

    # 计算调整后的先验框中心与宽高
    pred_boxes = FloatTensor(prediction[..., :4].shape)
    pred_boxes[..., 0] = x.data + grid_x
    pred_boxes[..., 1] = y.data + grid_y
    pred_boxes[..., 2] = torch.exp(w.data) * anchor_w
    pred_boxes[..., 3] = torch.exp(h.data) * anchor_h

    # 用于将输出调整为相对于416x416的大小
    _scale = torch.Tensor([stride_w, stride_h] * 2).type(FloatTensor)
    output = torch.cat((pred_boxes.view(batch_size, -1, 4) * _scale,
                        conf.view(batch_size, -1, 1), pred_cls.view(batch_size, -1, 80)), -1)
    return output

# ---------------------------------------------------#
#   yolo_body
# ---------------------------------------------------#
class YoloBody(nn.Module):
    def __init__(self, num_anchors, num_classes):
        super(YoloBody, self).__init__()
        #  backbone
        self.backbone = darknet53_tiny(None)

        self.conv_for_P5 = BasicConv(512, 256, 1)
        self.yolo_headP5 = yolo_head([512, num_anchors * (5 + num_classes)], 256)

        self.upsample = Upsample(256, 128)
        self.yolo_headP4 = yolo_head([256, num_anchors * (5 + num_classes)], 384)

    def forward(self, x):
        #  backbone
        feat1, feat2 = self.backbone(x)
        P5 = self.conv_for_P5(feat2)
        out0 = self.yolo_headP5(P5)

        P5_Upsample = self.upsample(P5)
        P4 = torch.cat([feat1, P5_Upsample], axis=1)

        out1 = self.yolo_headP4(P4)
        anchor0 = torch.tensor([[81,82],[135,169],[344,319]])
        anchor1 = torch.tensor([[23,27],[37,58],[81,82]])

        out0 = process(out0,anchor0)
        out1 = process(out1,anchor1)
        # out0 = DecodeBox(anchor0, 80,(416, 416))(out0)
        # out1 = DecodeBox(anchor1, 80,(416, 416))(out1)


        # output = torch.cat([out0,out1], 1)
        # batch_detections = non_max_suppression(output, 80,
        #                                        conf_thres=0.5,
        #                                        nms_thres=0.3)
        # res = batch_detections[0]

        return out0,out1

