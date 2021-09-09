import torch
from nets.yolo4_tiny import YoloBody

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch_model = torch.load("traffic.pt",map_location=device) # pytorch模型加载
torch_model2 = torch.load("../logs/Epoch100-Total_Loss5.2986-Val_Loss8.4492.pth",map_location=device) # pytorch模型加载

print("module_list_len:",len(torch_model['model']))
print("module_list2_len:",len(torch_model2))
new_state_dict = {}
for layer_name,layer_name2 in zip(torch_model['model'],torch_model2):
    # print(layer_name,np.shape(torch_model['model'][layer_name]),"\t",layer_name2,np.shape(torch_model2[layer_name2]))
    new_state_dict[layer_name2]=torch_model['model'][layer_name]

# print("--"*100)
model = YoloBody(3,1).eval()
print(model)
model.load_state_dict(new_state_dict)


batch_size = 1  #批处理大小
input_shape = (3, 416, 416)   #输入数据,改成自己的输入shape

# #set the model to inference mode
model.eval()

x = torch.randn(batch_size, *input_shape)	# 生成张量
export_onnx_file = "traffic.onnx"			# 目的ONNX文件名
torch.onnx.export(model,
                    x,
                    export_onnx_file,
                    opset_version=11,
                    do_constant_folding=True,	# 是否执行常量折叠优化
                    input_names=["input"],	# 输入名
                    output_names=["output0","output1"],	# 输出名
                    dynamic_axes={"input":{0:"batch_size"},  # 批处理变量
                                    "output0":{0:"batch_size"},
                                    "output1":{0:"batch_size"}
                                  })