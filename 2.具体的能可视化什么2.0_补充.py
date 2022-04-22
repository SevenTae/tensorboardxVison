'''from http://t.csdn.cn/UVHBV
http://t.csdn.cn/4gVvX
'''
from tensorboardX import SummaryWriter

#一次实验建立一个writer就够了,然后add乱七八糟的东西
writer = SummaryWriter("./visonlog/exp3")#前提这个路径要存在

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
                     '''

'''上边这种图像，这个样子的图像不在一张图上，需要鼠标选，下面使用pytorch制作网格图像
orchvision.utils.make_grid
功能：制作网格图像
• tensor：图像数据, BCHW形式
• nrow：行数（列数自动计算）
• padding：图像间距（像素单位）
• normalize：是否将像素值标准化
• range：标准化范围
• scale_each：是否单张图维度标准化
• pad_value：padding的像素值

'''
#vutils.make_grid接受的数据是一个batch的形式所以得先弄个自己的dataset
'''
import torchvision.utils as vutils
import  torchvision.transforms as transforms
from torch.utils.data import DataLoader ,Dataset
import os
from PIL import Image

class MyDataset(Dataset):

    def __init__(self,data_dir ,transform_compose):
        self.rootpath = data_dir
        self.img = os.listdir(self.rootpath)
    def __len__(self):
         return  len(self.img)
    def __getitem__(self, index):
        imgpath = os.path.join(self.rootpath, self.img[index])
        img = Image.open(imgpath)
        label = 0
        imgae = transform_compose(img)
        return  imgae ,label

train_dir= r"D:\CV\Segment\CGNet-master\tensorboardxVison\img"
transform_compose = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
train_data = MyDataset(data_dir=train_dir, transform_compose=transform_compose)
img ,labl = train_data[0]
print(type(img)) #<class 'torch.Tensor'>
print(img.shape) #torch.Size([3, 224, 224])
print(labl) #0
train_loader = DataLoader(dataset=train_data, batch_size=2, shuffle=False)
data_batch, label_batch = next(iter(train_loader))  # 取一个batch
# img_grid = vutils.make_grid(data_batch, nrow=4, normalize=True, scale_each=True)
img_grid = vutils.make_grid(data_batch, nrow=4, normalize=False, scale_each=False)
writer.add_image("input img", img_grid, 0)
writer.close()

'''


###！！！特征图可视化！！
import torchvision.utils as vutils
import  torchvision.transforms as transforms
from torch.utils.data import DataLoader ,Dataset
import os
from PIL import Image
import  torchvision.models as models
# 数据
path_img = r"D:\CV\Segment\CGNet-master\tensorboardxVison\img\1.jpg"     # your path to image
normMean = [0.49139968, 0.48215827, 0.44653124]
normStd = [0.24703233, 0.24348505, 0.26158768]
norm_transform = transforms.Normalize(normMean, normStd)
img_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    norm_transform])
# 数据读取
img_pil = Image.open(path_img).convert('RGB')
if img_transforms is not None:
    img_tensor = img_transforms(img_pil)
img_tensor.unsqueeze_(0)    # chw --> bchw

# 模型
alexnet = models.alexnet(pretrained=True)

# forward
convlayer1 = alexnet.features[0]
fmap_1 = convlayer1(img_tensor)
#它这一次层出来是 bchw=(1, 64, 55, 55)的形状
'''
为了方便，我们每次可视化一张图的中间特征图就行，也就是batch = 1
中间的输出是(1 channe h，w )
我们要把它变成 （channle 1 h， w） 要不然他会不认得
#如果图是在太小的话，比如到了32*32或者更小7*7这么小的的看看能不能先上采样放大一点 ，，（这个先不考虑了）
'''
# 预处理
fmap_1.transpose_(0, 1)  # bchw=(1, 64, 55, 55) --> (64, 1, 55, 55)
fmap_1_grid = vutils.make_grid(fmap_1, normalize=True, scale_each=True, nrow=8)
writer.add_image('feature map in conv1', fmap_1_grid, global_step=322)
writer.close()

