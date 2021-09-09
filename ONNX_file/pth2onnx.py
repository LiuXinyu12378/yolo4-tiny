import torch
from nets.yolo4_tiny import YoloBody

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch_model = torch.load("../logs/Epoch99-Total_Loss2.4070-Val_Loss3.2981.pth",map_location=device) # pytorch模型加载


model = YoloBody(3,1).eval()
model.load_state_dict(torch_model)

batch_size = 1  #批处理大小
input_shape = (3, 416, 416)   #输入数据,改成自己的输入shape

# #set the model to inference mode
model.eval()

x = torch.randn(batch_size, *input_shape)	# 生成张量
export_onnx_file = "./Epoch99-Total_Loss2.4070-Val_Loss3.2981.onnx"			# 目的ONNX文件名
torch.onnx.export(model,
                    x,
                    export_onnx_file,
                    opset_version=11,
                    do_constant_folding=True,	# 是否执行常量折叠优化
                    input_names=["input"],	# 输入名
                    output_names=["output"],	# 输出名
                    dynamic_axes={"input":{0:"batch_size"},  # 批处理变量
                                    "output":{0:"batch_size"}})