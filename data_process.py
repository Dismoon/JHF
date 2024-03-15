import torch
from torchvision import transforms
from torch.utils.data import Dataset
import glob
import os
from PIL import Image
from torchvision import transforms
import cv2
import numpy as np

class imgdata(Dataset):
    def __init__(self,img_dir,label_dir):
        self.img_info=self.get_img_info(img_dir)
        self.label_info=self.get_img_info(label_dir)


    def __getitem__(self, item):
        img=cv2.imread(self.img_info[item],-1)#读取数据
        #转换为tensor，并归一化到（0,1）
        label=cv2.imread(self.label_info[item],-1)
        #根据一定概率进行翻转和旋转
        random=np.random.rand(2)

        img=self.augementation(img,random)
        label=self.augementation(label,random)


        return  img,label

    def __len__(self):
        return len(self.img_info)

    def get_img_info(self,data_dir):#获取图片的路径列表
        data_info=sorted(glob.glob(os.path.join(data_dir,'*.png')))


        return data_info
    def augementation(self,data,random):#图像增强函数
        flip_H=transforms.RandomHorizontalFlip(p=1)
        flip_V=transforms.RandomVerticalFlip(p=1)
        totensor = transforms.ToTensor()
        data=totensor(data)
        if random[0] < 0.3:
            img_flip = flip_V(data)
        elif random[0] > 0.7:
            img_flip = flip_H(data)
        else:
            img_flip = data


        if random[1] < 0.5:
            img_rot = torch.rot90(img_flip, 1, [1, 2])
        else:
            img_rot = img_flip

        return img_rot
class imgdata2(Dataset):
    def __init__(self,img_dir,label_dir,format='*.png'):
        self.img_info=self.get_img_info(img_dir,format)
        # self.middle_info = self.get_img_info(middle_dir,format)
        self.label_info=self.get_img_info(label_dir,format)


    def __getitem__(self, item):
        img=cv2.imread(self.img_info[item],-1)#读取数据
        #转换为tensor，并归一化到（0,1）
        # middle=cv2.imread(self.middle_info[item],-1)
        label=cv2.imread(self.label_info[item],-1)
        #根据一定概率进行翻转和旋转
        random=np.random.rand(2)

        img=self.augementation(img,random)
        # middle = self.augementation(middle, random)
        label=self.augementation(label,random)


        return  img,label

    def __len__(self):
        return len(self.img_info)

    def get_img_info(self,data_dir,format):#获取图片的路径列表
        data_info=sorted(glob.glob(os.path.join(data_dir,format)))


        return data_info
    def augementation(self,data,random):#图像增强函数
        flip_H=transforms.RandomHorizontalFlip(p=1)
        flip_V=transforms.RandomVerticalFlip(p=1)
        # clip = transforms.CenterCrop((data.shape[0]//16*16, data.shape[1]//16*16))
        totensor = transforms.ToTensor()
        data=totensor(data)
        if random[0] < 0.3:
            img_flip = flip_V(data)
        elif random[0] > 0.7:
            img_flip = flip_H(data)
        else:
            img_flip = data


        if random[1] < 0.5:
            img_rot = torch.rot90(img_flip, 1, [1, 2])
        else:
            img_rot = img_flip

        return img_rot
class vailddata(Dataset):
    def __init__(self,img_dir,label_dir):
        self.img_info=self.get_img_info(img_dir)
        self.label_info=self.get_img_info(label_dir)


    def __getitem__(self, item):
        img=cv2.imread(self.img_info[item],-1)#读取数据
        #转换为tensor，并归一化到（0,1）
        label=cv2.imread(self.label_info[item],-1)

        img=self.augementation(img)
        label=self.augementation(label)


        return  img,label

    def __len__(self):
        return len(self.img_info)

    def get_img_info(self,data_dir):#获取图片的路径列表
        data_info=sorted(glob.glob(os.path.join(data_dir,'*.png')))


        return data_info
    def augementation(self,data):#图像增强函数

        totensor = transforms.ToTensor()
        data=totensor(data)


        return data
class vailddata2(Dataset):
    def __init__(self,img_dir,label_dir,format='*.png'):
        self.img_info=self.get_img_info(img_dir,format)
        # self.middle_info = self.get_img_info(middle_dir,format)
        self.label_info=self.get_img_info(label_dir,format)
        self.A_size = len(self.img_info)  # get the size of dataset A
        self.B_size = len(self.label_info)

    def __getitem__(self, item):

        img=cv2.imread(self.img_info[item % self.A_size],-1)#读取数据
        #转换为tensor，并归一化到（0,1）
        # middle=cv2.imread(self.middle_info[item],-1)
        label=cv2.imread(self.label_info[item % self.B_size],-1)
        #根据一定概率进行翻转和旋转
        name = self.img_info[item % self.B_size].split('/')[-1].split('.')[0]
        img=self.augementation(img)
        # middle = self.augementation(middle)
        label=self.augementation(label)


        return  img,label,name

    def __len__(self):
        return max(self.A_size, self.B_size)

    def get_img_info(self,data_dir,format):#获取图片的路径列表
        data_info=sorted(glob.glob(os.path.join(data_dir,format)))


        return data_info
    def augementation(self,data):#图像增强函数
        # clip = transforms.CenterCrop((data.shape[0]//16*16, data.shape[1]//16*16))
        totensor = transforms.ToTensor()
        data=totensor(data)


        return data

def image_up(dir,target_size_h,target_size_w,stride,categoary):#分割数据，大小为target_size，步长为stride
    inputlist=sorted(glob.glob(os.path.join(dir,'*.png')))
    path_split=os.path.join(dir.split('/')[0],dir.split('/')[1],categoary+'_split')
    if not os.path.isdir(path_split):
        os.makedirs(path_split)
    for k in inputlist:
        image=cv2.imread(k,-1)
        image_shape=image.shape
        i=0
        for x in range(0,image_shape[0]-target_size_h+1,stride):
            if x+target_size_h>image.shape[0]:
                break
            for y in range(0,image_shape[1]-target_size_w+1,stride):
                if y+target_size_w > image.shape[1]:
                    break
                sub_img= image[x:x+target_size_h,y:y+target_size_w]
                print(os.path.join(path_split,k.split('/')[-1].split('.')[0]+'_'+str(i)+'.png'))
                cv2.imwrite(os.path.join(path_split,k.split('/')[-1].split('.')[0]+'_'+str(i)+'.png'),sub_img)
                i+=1
    return True
if __name__ =='__main__':
    image_up('./ysydata/label64',64,64,32,'label64')
    # import shutil

#     list=sorted(glob.glob(os.path.join('./ysydata/vaild_0.1ms','*.png')))
#     for j in list:
#         print(j.split('/')[-1])
#         if not os.path.isdir('./ysydata//vaild_label'):
#             os.makedirs('./ysydata//vaild_label')
#         shutil.copyfile('./ysydata/label_all/'+j.split('/')[-1],'./ysydata/vaild_label/'+j.split('/')[-1])
# #









