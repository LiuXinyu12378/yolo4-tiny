import os, sys
import cv2
sys.path.append(os.getcwd())
import onnxruntime
import onnx
import torchvision.transforms as transforms
import torch
import numpy as np

class ONNXModel():
    def __init__(self, onnx_path):
        """
        :param onnx_path:
        """
        self.onnx_session = onnxruntime.InferenceSession(onnx_path)
        self.input_name = self.get_input_name(self.onnx_session)
        self.output_name = self.get_output_name(self.onnx_session)
        print("input_name:{}".format(self.input_name))
        print("output_name:{}".format(self.output_name))

    def get_output_name(self, onnx_session):
        """
        output_name = onnx_session.get_outputs()[0].name
        :param onnx_session:
        :return:
        """
        output_name = []
        for node in onnx_session.get_outputs():
            output_name.append(node.name)
        return output_name

    def get_input_name(self, onnx_session):
        """
        input_name = onnx_session.get_inputs()[0].name
        :param onnx_session:
        :return:
        """
        input_name = []
        for node in onnx_session.get_inputs():
            input_name.append(node.name)
        return input_name

    def get_input_feed(self, input_name, image_tensor):
        """
        input_feed={self.input_name: image_tensor}
        :param input_name:
        :param image_tensor:
        :return:
        """
        input_feed = {}
        for name in input_name:
            input_feed[name] = image_tensor
        return input_feed

    def forward(self, image_tensor):
        '''
        image_tensor = image.transpose(2, 0, 1)
        image_tensor = image_tensor[np.newaxis, :]
        onnx_session.run([output_name], {input_name: x})
        :param image_tensor:
        :return:
        '''
        # 输入数据的类型必须与模型一致,以下三种写法都是可以的
        # scores, boxes = self.onnx_session.run(None, {self.input_name: image_tensor})
        # scores, boxes = self.onnx_session.run(self.output_name, input_feed={self.input_name: image_tensor})
        input_feed = self.get_input_feed(self.input_name, image_tensor)
        scores, boxes = self.onnx_session.run(self.output_name, input_feed=input_feed)
        return scores, boxes


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


r_model_path = "./Epoch99-Total_Loss2.4070-Val_Loss3.2981.onnx"


img = cv2.imread("../VOCdevkit/fall_detection/images/fall_1.jpg")
print(img.shape)
img = cv2.resize(img, (416, 416))
print(img.shape)
to_tensor = transforms.ToTensor()
img = to_tensor(img)
img = torch.unsqueeze(img,0)
print(img.shape)


rnet1 = ONNXModel(r_model_path)
out = rnet1.forward(to_numpy(img))

torch.set_printoptions(profile="full")
print(out[0].shape)
print(out[1].shape)
out0 = torch.from_numpy(out[0])
out1 = torch.from_numpy(out[1])
print(type(out0))
print(type(out1))
# print(out1,type(out1))

with open("out0_onnx.txt","w") as f_write:
    f_write.write(str(out0))

# print(out[1])