import os
import cv2 as cv
from captcha.image import ImageCaptcha
import matplotlib.pyplot as plt
import numpy as np
import random
import string
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.data.dataset as Dataset
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms

path=r"D:\PythonProjectData\【Pytorch】验证码识别"
#
# # 我们首先来尝试生成验证码
# # characters是验证码上的字符集，包含了10个数字和二十六个大写英文字母
# characters=string.digits+string.ascii_uppercase
# # 0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ
#
# # 我们也可以对验证码的宽度width和高度height进行设置，以及字体的type
# width,height,n_len,n_class=170,80,4,len(characters)
# generator=ImageCaptcha(width=width,height=height)
#
# # 随机生成字符的字符串
# random_str="".join([random.choice(characters) for i in range(4)])
# # 生成验证码
# img=generator.generate_image(random_str)
#
# plt.imshow(img)
# plt.show()
#
# # 在路径下生成训练集
# def random_captcha_text(size=4):
#     characters=string.digits
#     return "".join([random.choice(characters) for i in range(size)])
#
# def gen_captcha_dataset(path,num=1000):
#     path=path
#     image=ImageCaptcha(width=160,height=60)
#     text=[]
#     for i in range(num):
#         # 直接调用writer方法写入文件
#         captcha_text=random_captcha_text()
#         text.append(captcha_text)
#         image.write(captcha_text,path+"/%s"%i+".jpg")
#     # 作为标签写入
#     with open(path+"/label.txt","w") as f:
#         for i in text:
#             f.write(i+"\n")
#
# # gen_captcha_dataset(path)
#
# # step 1. 读取数据！
# # 我们需要自定义数据集，方便加载我们的验证码和label标签
# # 上述的类共有10类，我们可以将图片对应的输出映射为： 4 * 10(one-hot)
# # 通过交叉熵来实现精度验证
# print()
# class dataSet(torch.utils.data.Dataset):
#     def __init__(self,root_dir,label_file,transform=transforms.ToTensor()):
#         self.root_dir=root_dir
#         self.label=torch.Tensor([[int(_) for _ in i ] for i in np.loadtxt(label_file,dtype=str)]).int() # 这玩意默认是float类型
#         # 需要将float类型转化为int型后分开
#         self.transform=transform
#
#     # 这个是Dataset类的迭代器
#     def __getitem__(self, idx):
#         # 获取图片和对应的索引
#         img_name=os.path.join(self.root_dir,"%d.jpg"%idx)
#         img=Image.open(img_name) # 无论啥都行，能打开就行
#         # 对应标签
#         labels=self.label[idx]
#         if self.transform:
#             img=self.transform(img)
#         return img,labels
#
#     def __len__(self):
#         return (len(self.label))
#
# # 尝试读取我们的数据吧
# dataset=dataSet(path,path+"/label.txt")
# i,j=dataset.__getitem__(0)
# print(i.shape)
# print(j)
#
#
# # Step2. 搭建卷积神经网络！
# # 我们在之前并没有指定ImageCaptcha的大小，所以最终的维度是(3,60,160)
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv=nn.Sequential(
            nn.Conv2d(3,32,kernel_size=4,stride=1,padding=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2,inplace=True),
            nn.MaxPool2d(kernel_size=2),
            # out: (bs,32,30,80)

            nn.Conv2d(32,64,kernel_size=4,stride=1,padding=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2,inplace=True),
            nn.MaxPool2d(kernel_size=2),
            # out: (bs,64,15,40)

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(kernel_size=2)
            # out: (bs,64,7,20)
        )

        self.fc1=nn.Linear(64*7*20,500)
        self.fc2=nn.Linear(500,40)
        # 每个图片有四个数字，每个数字有十个分类

    def forward(self,x):
        x=self.conv(x)
        x=x.view(x.size(0),-1) # (bs,64*7*20)
        output=self.fc2(self.fc1(x))
        return output
#
# # Step 3. 自定义损失函数！
# # 我们采用交叉熵(适用于多分类)作为损失函数
# # 首先在output中提取出第一个数字对应的十个维度
# # 将每个数字的十维度进行堆叠，形成四个样本数据，计算其损失
# class nCrossEntropLoss(torch.nn.Module):
#     def __init__(self,n=4,device=None):
#         super(nCrossEntropLoss, self).__init__()
#         self.n=n
#         self.total_loss=0
#
#         self.loss=nn.CrossEntropyLoss()
#         if device:
#             self.loss=self.loss.to(device)
#
#     def forward(self,output,label):
#         # output: [bs,40]
#         output=output.cpu()
#         output_fir=output[:,0:10]
#         label=Variable(torch.LongTensor(label.data.cpu().numpy()))
#         label_fir=label[:,0]
#         # 将形状堆叠，变成[4*_,10]
#         for i in range(1,self.n):
#             output_fir=torch.cat((output_fir,output[:,10*i:10*i+10]),0)
#             label_fir=torch.cat((label_fir,label[:,i]),0)
#         # 思路是将一张图平均剪切为四张小图，进行四个多分类
#             self.total_loss=self.loss(output_fir,label_fir)
#         return self.total_loss
#
#
# # Step 4. 模型训练！
# import time
# net=ConvNet()
# optimizer=torch.optim.Adam(net.parameters(),lr=0.001)
#
# ds=dataSet(path,path+"/label.txt")
# dataSet_size=len(ds)
# dataLoader=torch.utils.data.DataLoader(
#     dataset=ds,
#     batch_size=10,
#     shuffle=True
# )
# # 模型参数
# best_model_wts=net.state_dict()
# best_acc=0.0
# since=time.time()
# EPOCH=50
# device=torch.device("cuda:0")
# net=net.to(device)
# loss_func=nCrossEntropLoss(4,device)
# loss_func=loss_func.to(device)
#
# for epoch in range(EPOCH):
#     instance_loss=0.0
#     instance_correct=0
#
#     for step,(bx,by) in enumerate(dataLoader):
#         bs=bx.shape[0]
#         pred=torch.LongTensor(bs,1).zero_()
#         bx,by=bx.to(device),by.to(device)
#         bx=Variable(bx) # (bs,3,60,160)
#         by=Variable(by) # (bs,4)
#         optimizer.zero_grad()
#         out=net(bx) # (bs,40)
#
#         loss=loss_func(out,by)
#
#         for i in range(4):
#             pre=F.log_softmax(out[:,10*i:10*i+10],dim=1)
#             pred=torch.cat((pred,pre.data.max(1,keepdim=True)[1].cpu()),dim=1)
#         loss.backward()
#         optimizer.step()
#
#         instance_loss+=loss.data*bx.size()[0]
#
#         instance_correct+=np.sum(np.equal(pred.numpy()[:,1:],by.data.cpu().numpy().astype(int)))
#
#     epoch_loss=instance_loss/dataSet_size
#     epoch_acc=instance_correct/dataSet_size/4*100
#
#     if epoch_acc>best_acc:
#         best_acc=epoch_acc
#         best_model_wts=net.state_dict()
#
#     if epoch==EPOCH-1:
#         torch.save(best_model_wts,path+"/best_model.pkl")
#
#     print()
#     time_elapsed = time.time() - since
#     print('Training complete in {:.0f}m {:.0f}s'.format(
#         time_elapsed // 60, time_elapsed % 60))
#     print('Train Loss:{:.4f} Acc: {:.4f}%'.format(epoch_loss, epoch_acc))

# Step 5. 实际运用！
# 俺们生成一个小玩意
characters=string.digits
width,height,n_len,n_class=160,60,4,len(characters)
generator=ImageCaptcha(width=width,height=height)
# 随机生成字符的字符串
random_str="".join([random.choice(characters) for i in range(4)])
# 生成验证码
img=generator.generate_image(random_str)
plt.imshow(img)
plt.show()
img.save(path+"/test.jpg")

# 读入数据
data=Image.open(path+"/test.jpg")
data=transforms.ToTensor()(data)
data=torch.unsqueeze(data,0)
print(data.shape)

# 读入模型
net=ConvNet()
net.load_state_dict(torch.load(path+"/best_model.pkl"))
pre=net(data)
print(pre.shape)
pre_t=pre[:,:10]
for i in range(1,4):
    pre_t=torch.cat((pre_t,pre[:,10*i:10*i+10]),dim=0)
print(pre_t)
print("Predict value is %s"%(torch.argmax(pre_t,dim=1)))
