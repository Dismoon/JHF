import numpy as np
import glob  #查找符合特定规则的文件路径名
from PIL import Image

import os
import cv2
import math
# from Tool import cal_stokes_dolp
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
def adjust_gamma(imgs, gamma=3.0):
    table = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    # print(np.array(imgs, dtype = np.uint8))
    new_imgs = cv2.LUT(np.array(imgs, dtype = np.uint8), table)
    return new_imgs
alpha=np.random.rand(100)*0.1+0.9
beta=np.random.rand(100)*0.5+0.5
gamma=np.random.rand(100)*1.5+1.5
file='./val_label'
input_list = sorted(glob.glob(os.path.join(file, "*png")))
# print(input_list)
stds=np.random.rand(100)*5+20
for i,k in enumerate(input_list):
    img=cv2.imread(k,-1)
    print(img.dtype)
    A = img + np.random.normal(0, stds[i], size=(img.shape[0], img.shape[1], 4))
    print(A.dtype)
    A = np.clip(A, 0, 255)
    A = alpha[i]*np.float32(A)
    A = adjust_gamma(A, gamma[i]) * beta[i]
    print(A.dtype)


    # A = np.random.poisson(A)
    # A=np.clip(A,0,255)
    cv2.imwrite(os.path.join('./val_input', k.split("/")[-1].split('.')[0] ) + '.png',A)
    # im.save(os.path.join('./input3',k.split("/")[-1].split('.')[0]+'_'+str(round(stds[i],2)))+'.png')
file2='./input2'
input_list2=sorted(glob.glob(os.path.join(file2, "*png")))
def psnr1(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse < 1.0e-10:
        return 100
    return 10 * math.log10(255.0 ** 2 / mse)


def psnr2(img1, img2):
    mse = np.mean((img1 / 255. - img2 / 255.) ** 2)
    if mse < 1.0e-10:
        return 100
    PIXEL_MAX = 1
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
psnr_sum=0
# for i,k in zip(input_list,input_list2):
#     label = np.float32(np.array(Image.open(i)))
#     image=  np.float32(np.array(Image.open(k)))
#     psnr=psnr1(image,label)
#     print(psnr)
#     psnr_sum+=psnr
#
# print('psnr_mean: '+str(psnr_sum/len(input_list)))
# for j,i in enumerate(range(1,100,5)):
#     img =np.float32(np.array(Image.open('./test6/label/0.png')))
#     # label=np.float32(np.array(Image.open(file2)))
#     # img=img[:,:,np.newaxis]
#     A = img + np.random.normal(0, stds[i], size=(img.shape[0], img.shape[1], 4))
#     A = np.clip(A, 0, 255)
#     A = alpha[i]*np.float32(A)
#     A = adjust_gamma(A, gamma[i]) * beta[i]
#     A = A.astype(np.uint8)
#     im = Image.fromarray(A)
#     im.save('./test6/input/%s_'%j+'%s.png'%i)