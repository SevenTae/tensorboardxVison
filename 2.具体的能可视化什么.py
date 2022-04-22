'''from http://t.csdn.cn/UVHBV'''
from tensorboardX import SummaryWriter

#一次实验建立一个writer就够了,然后add乱七八糟的东西
writer = SummaryWriter("./visonlog/exp2")#前提这个路径要存在
#y
# 1.数字
# writer.add_scalar()#关于数的
'''Add scalar data to summary.'''
for  i in range(100):
    writer.add_scalar("y=x2",i*i,i)
# writer.add_image()#关于图的
#2.图片
'''
tag (string): 数据名称
img_tensor (torch.Tensor / numpy.array): 图像数据
global_step (int, optional): 训练的 step
walltime (float, optional): 记录发生的时间，默认为 time.time()
dataformats (string, optional): 图像数据的格式，默认为 'CHW'，即 Channel x Height x Width，还可以是 'CHW'、'HWC' 或 'HW' 等

'''
# add_image 方法只能一次插入一张图片
import os
import cv2 as cv
impath = r"D:\CV\Segment\CGNet-master\tensorboardxVison\img"
img = os.listdir(impath)
print(img)
for i in range(1, 6):
    writer.add_image('countdown',
                     cv.cvtColor(cv.imread(r'D:\CV\Segment\CGNet-master\tensorboardxVison\img\{}.jpg'.format(i)), cv.COLOR_BGR2RGB),
                     global_step=i,
                     dataformats='HWC')


#3.运行图（也就是你的神经网络的那个model）
'''
model (torch.nn.Module): 待可视化的网络模型
input_to_model (torch.Tensor or list of torch.Tensor, optional): 待输入神经网络的变量或一组变量
'''
from utils.summary import summary
from model import CGNet
import torch
input_to_model=torch.randn([1,3,640,640])
model = CGNet.Context_Guided_Network(19, M=3, N=21)
writer.add_graph(model, input_to_model=input_to_model, verbose=False)

writer.close()
