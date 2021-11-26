import torch
from torch import tensor
import torch.nn as nn

import torchvision

import numpy as np
import matplotlib.image as mpimg

from models import Generator

import PIL
def load_model(path, model_strcuture):
    # path为模型参数.pth文件的储存路径
    # model_strcuture为继承自nn.Module的模型结构，需要显式地加载，eg.model = Generator(3, 3)
    # 返回模型参数 model_para，type(model_para) == <class 'collections.OrderedDict'>
    # 返回加载好参数的完整模型 model_strcuture
    model_para = torch.load(path)
    model_strcuture.load_state_dict(model_para)
    return model_para, model_strcuture

if __name__ == "__main__":

    path = "./output/20211103/netG_B2A.pth"
    model = Generator(3, 3)
    # 加载模型
    model_para, model = load_model(path, model)
    layer_name = []
    # print(model)输出的结果就是 print(model.state_dict)
    for param_tensor in model.state_dict():
        # model.state_dict() type 为 <class 'collections.OrderedDict'>有序字典
        #打印 key value字典
        #print(param_tensor,'\t',model.state_dict()[param_tensor].size())
        layer_name.append(param_tensor)

    print(type(model_para))
    exp = nn.Conv2d(3,64,7)
    # tensor要先转为nn.Parameter才能赋给Conv.weight
    exp.weight = nn.Parameter(model_para[layer_name[0]])
    exp.bias = nn.Parameter(model_para[layer_name[1]])
    # 读取图片
    input = torch.from_numpy(mpimg.imread("./datasets/noise2denoise/train/B/negative0.jpg"))
    input = input.float()
    print(input.size())
    input = input.permute([2,0,1])
    input = torch.unsqueeze(input, 0)
    input = input.cuda()
    #print(input.size())
    #output0 = exp(input)
    # 后处理
    #output = output0.permute([1,0,2,3])
    #print(output.size())
    #torchvision.utils.save_image(output,"./output/output0.png")
    # 自动化处理
    input_copy = input
    for i in range(int(len(layer_name) / 2)):
        # 输入features预处理
        print("input: ", input_copy.size())
        print(layer_name[2 * i],'\t',model_para[layer_name[2 * i]].size())
        # 创建Conv层
        conv_size = list(model_para[layer_name[2 * i]].size())
        conv_Layer = nn.Conv2d(conv_size[1], conv_size[0], conv_size[2])
        conv_Layer.weight = nn.Parameter(model_para[layer_name[2 * i]])
        conv_Layer.bias = nn.Parameter(model_para[layer_name[2 * i + 1]])
        # 卷积
        tmp = conv_Layer(input_copy)
        input_copy = tmp
        #input_copy = tmp.permute([1,0,2,3])
        print("output: ", input_copy.size())
        torchvision.utils.save_image(input_copy.permute([1,0,2,3]),"./output/output"+ str(i) +".png")