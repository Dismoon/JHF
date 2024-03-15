import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data_process import *
from model_torch import *
import numpy as np
import random
from data_process import imgdata
from tensorboardX import SummaryWriter
from utils_torch import *
from PIL import Image
import time
import cv2
from UNET import UNet

os.environ["CUDA_VISIBLE_DEVICES"]='0'
def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
print(torch.version.cuda)
toPIL=transforms.ToPILImage()
#设置随机种子
set_seed()
#参数设置

Epoch2=64

batch_size2=128
lr_sa=1e-4
lr=lr_sa
c_dim=4
D=10
C=6
G=32
G0=64
ks=3
#加载数据
data_num = 2
# vaild_data2=vailddata2('./poldata/vaild70','./poldata/vaild_label70')
# vaild_loader2=DataLoader(vaild_data2,1)
filename = './cyclegan_2/dataset_%s/test_result/RDN/'%data_num
vaild_data1=vailddata2('./cyclegan_2/dataset_%s/testA'%data_num,'./cyclegan_2/dataset_%s/testB'%data_num,'*.png')
vaild_loader1=DataLoader(vaild_data1,1)

#加载模型
net=model(c_dim,G0,ks,C,D,G,False)
net.initialize_weight()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
net.to(device)

#损失函数




#summary
data = '22.6.1'
data2=10.11

epoch = 60
category='self_1'
if os.path.isdir('./checkpoint/'+ data):
    checkpoint=torch.load('./checkpoint/'+data+'/checkpoint_epoch_60.pkl')
    net.load_state_dict(checkpoint['model_state_dict'])


if not os.path.isdir(filename+data+'/%s/'%epoch):
    os.makedirs(filename+data+'/%s/'%epoch)
net.eval()
with torch.no_grad():
    for j,datas in enumerate(vaild_loader1):
        # 前向传播forward
        img,label,names = datas
        name = names[0]
        print(name)
        img = img.to(device)
        label = label.to(device)
        output2 = net(img)
        dofp2 = todofp(output2)
        save_img(dofp2, 'dofp_out', name, epoch, data, filename)
        dofp3 = todofp(label)
        save_img(dofp3, 'dofp_label', name, epoch, data,filename)
        dofp4 = todofp(img)
        save_img(dofp4, 'dofp_input', name, epoch, data,filename)

        S0_in, S1_in, S2_in = cal_s(img)
        input_dolp = cal_dolp(S0_in, S1_in, S2_in)
        input_aop = cal_aop(S1_in, S2_in)
        S0, S1, S2 = cal_s(output2)
        img_dolp = cal_dolp(S0, S1, S2)
        img_aop = cal_aop(S1, S2)
        S0_gt, S1_gt, S2_gt = cal_s(label)
        label_dolp = cal_dolp(S0_gt, S1_gt, S2_gt)
        label_aop = cal_aop(S1_gt, S2_gt)
        save_img(img_dolp, 'dolp_out', name, epoch,data, filename)
        save_img(img_aop, 'aop_out', name, epoch, data, filename)
        save_img(S0, 'S0_out', name, epoch, data, filename)
        save_img(label_dolp, 'dolp_label', name, epoch, data,filename)
        save_img(label_aop, 'aop_label', name, epoch, data,filename)
        save_img(S0_gt, 'S0_label', name, epoch, data, filename)
        save_img(input_dolp, 'dolp_input', name, epoch, data,filename)
        save_img(input_aop, 'aop_input', name, epoch, data,filename)
        save_img(S0_in, 'S0_in', name, epoch, data, filename)
        # img = toBGR(img)
        # save_img(img, 'input', j, 60, data,'./cyclegan_2/dataset_6/test_result/RDN/')
        # output2 = toBGR(output2)
        # save_img(output2, 'output', j, 60, data,'./cyclegan_2/dataset_6/test_result/RDN/')
        # label = toBGR(label)
        # save_img(label, 'label', j, 60, data,'./cyclegan_2/dataset_6/test_result/RDN/')










