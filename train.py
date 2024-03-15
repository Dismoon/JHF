import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model_torch import *
import numpy as np
import random
from data_process import imgdata2,vailddata2
from tensorboardX import SummaryWriter
from utils_torch import *
import time
from UNET import UNet
from torchvision import transforms
from torch.optim import lr_scheduler

os.environ["CUDA_VISIBLE_DEVICES"]='0'
def set_seed(seed=22):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
print(torch.version.cuda)
toPIL=transforms.ToPILImage()
trigger=False
data = '22.6.16'
data2=2.3
#设置随机种子
set_seed()
#参数设置
Epoch1=60
if trigger:
    Epoch2=64
    batch_size2 = 128
else:
    Epoch2 = 200
    batch_size2 = 64
batch_size1=32

lr_sa=1e-4
lr=lr_sa
c_dim=4
D=10
C=6
G=32
G0=64
ks=3
#加载数据
# train_data1=imgdata2('./input_split','./label_split','*.png')
# vaild_data1=vailddata2('./valid_split','./valid_label_split','*.png')
train_data1=imgdata2('./cyclegan_2/dataset_4/trainA','./cyclegan_2/dataset_4/trainB','*.png')
vaild_data1=vailddata2('./cyclegan_2/dataset_4/valA','./cyclegan_2/dataset_4/valB','*.png')

train_loader1=DataLoader(train_data1,batch_size1,shuffle=True)
vaild_loader1=DataLoader(vaild_data1,1)
# train_data2=imgdata2('./poldata/input7_split','./poldata/label7_split')
# vaild_data2=vailddata2('./poldata/vaild70','./poldata/vaild_label70')

# train_loader2=DataLoader(train_data2,batch_size2,shuffle=True)
# vaild_loader2=DataLoader(vaild_data2,1)

#加载模型
net=model(c_dim, G0, ks, C, D, G, bool(1-trigger))
# net=UNet()
net.initialize_weight()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
net.to(device)

#损失函数
class PALoss(nn.Module):
    def __init__(self):
        super(PALoss,self).__init__()


    def forward(self,image,label):
        S0, S1, S2 = cal_s(image)
        img_dolp = cal_dolp(S0,S1, S2)
        img_aop = cal_aop(S1, S2)
        S0_gt, S1_gt, S2_gt = cal_s(label)
        label_aop = cal_aop(S1_gt, S2_gt)
        label_dolp = cal_dolp(S0,S1, S2)
        loss_c=F.mse_loss(image,label)
        loss_d = F.mse_loss(img_dolp, label_dolp)
        loss_a=F.mse_loss(img_aop,label_aop)*0.2
        wc=loss_c/(loss_c+loss_a+loss_d)
        wa=loss_a/(loss_c+loss_a+loss_d)
        wd=loss_d/(loss_c+loss_a+loss_d)
        loss=wc*loss_c+wa*loss_a+wd*loss_d
        return loss

class PALoss2(nn.Module):
    def __init__(self):
        super(PALoss2,self).__init__()


    def forward(self,image,label):
        image0 = image[:, 0, :, :]
        image45 = image[:, 1, :, :]
        image90 = image[:, 2, :, :]
        label0 = label[:, 0, :, :]
        label45 = label[:, 1, :, :]
        label90 = label[:, 2, :, :]
        loss_0=F.l1_loss(image0,label0)
        loss_45 = F.l1_loss(image45, label45)
        loss_90=F.l1_loss(image90,label90)
        w0=loss_0/(loss_0+loss_45+loss_90)
        w45=loss_45/(loss_0+loss_45+loss_90)
        w90=loss_90/(loss_0+loss_45+loss_90)
        loss=w0*loss_0+w45*loss_45+w90*loss_90


        return loss

# Loss=PALoss2()
# Loss=nn.L1Loss()
# Loss1=nn.MSELoss()
Loss2=nn.MSELoss()
#优化器
optimizer=torch.optim.Adam(net.parameters(), lr)
scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
#summary

if not os.path.isdir('./log_dir/' + data):
    os.makedirs('./log_dir/' + data)
# if not os.path.isdir('./log_dir/%s_2'%data2):
#     os.makedirs('./log_dir/%s_2'%data2)
writer1=SummaryWriter('./log_dir/' + data, filename_suffix='123456')
# writer2=SummaryWriter('./log_dir/%s_2'%data2,filename_suffix='123456')
iter = 0
#断点训练
last_epoch=0
last_epoch2=0
# clip=transforms.CenterCrop((1024, 1024))
# temp=100
#
# if trigger:
#     if os.path.isdir('./checkpoint/8.18_1'):
#         checkpoint=torch.load('./checkpoint/8.18_1/checkpoint_epoch_74.pkl')
#         net.load_state_dict(checkpoint['model_state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # last_epoch=checkpoint['epoch']
    # Epoch1=Epoch1-last_epoch
    # iter=checkpoint['iter']
    # checkpoint2 = torch.load('./checkpoint/8.18_1/checkpoint_epoch_74.pkl')
    # temp=checkpoint2['temp']
    # lr = checkpoint['lr']
    # for param_group in optimizer.param_groups:
    #     param_group['lr'] = lr
#开始训练
time0=time.time()

count=0
for epoch in range(1, Epoch1+1):
    loss_mean=0
    loss_mean1 = 0
    net.train()
    for i,datas in enumerate(train_loader1):
        #前向传播forward
        img,label=datas
        img=img.to(device)
        # middle = middle.to(device)
        label=label.to(device)
        # output1,_=net(img)
        #
        # #后向传播backward
        # optimizer.zero_grad()
        # loss1 = Loss1(output1,middle)
        # loss1.backward()
        # loss_mean1 += loss1.item()
        # optimizer.step()
        optimizer.zero_grad()
        output2= net(img)
        loss=Loss2(output2,label)
        loss.backward()
        iter+=1
        loss_mean+=loss.item()

        #更新权重
        optimizer.step()
        if iter % 20 == 0:
            S0, S1, S2 = cal_s(output2)
            img_dolp=cal_dolp(S0,S1,S2)
            img_aop=cal_aop(S1,S2)
            S0_gt,S1_gt,S2_gt=cal_s(label)
            label_dolp=cal_dolp(S0_gt,S1_gt,S2_gt)
            label_aop=cal_aop(S1_gt,S2_gt)
            psnr=psnr1(output2,label).item()
            psnr_dolp=psnr1(img_dolp,label_dolp).item()
            psnr_aop=psnr1(img_aop,label_aop).item()
            print('Training: Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] time[{:.4f}min]'
                  'Loss[{:.7f}] PSNR[{:.4f}] PSNR_Dolp[{:.4f}] PSNR_aop[{:.4f}]'.format(epoch+last_epoch,Epoch1+last_epoch,i+1,len(train_loader1),(time.time()-time0)/60,loss_mean/20,
                                                                                        psnr,psnr_dolp,psnr_aop))
            loss_mean=0
            writer1.add_scalars('PSNR',{'DOFP':psnr,'DOLP':psnr_dolp,'AOP':psnr_aop},iter)
        writer1.add_scalars('Loss',{'Train':loss.item()},iter)
        #     psnr=psnr1(output2,label).item()
        #     print('Training: Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] time[{:.4f}min]'
        #           'Loss[{:.7f}] PSNR[{:.4f}] '.format(epoch+1+last_epoch,Epoch1+last_epoch,i+1,len(train_loader1),(time.time()-time0)/60,loss_mean/20,
        #                                                                                 psnr))
        #     loss_mean=0
        #     # loss_mean1 = 0
        #     writer1.add_scalars('PSNR',{'RGB':psnr},iter)
        # writer1.add_scalars('Loss',{'Train':loss.item()},iter)
        # writer.add_scalars('Loss_MSE', {'Train': loss1.item()}, iter)


    if not os.path.isdir('./cyclegan_2/results/RDN/'+data+'/%s'%(epoch+last_epoch )):
        os.makedirs('./cyclegan_2/results/RDN/'+data+'/%s'%(epoch+last_epoch ))
    net.eval()
    with torch.no_grad():
        val_loss_mean=0
        val_psnr =0
        val_psnr_dolp = 0
        val_psnr_aop = 0
        for j,datas in enumerate(vaild_loader1):
            # 前向传播forward
            img,label = datas
            img = img.to(device)
            label = label.to(device)
            output2 = net(img)
            loss = Loss2(output2, label)
            val_loss_mean+=loss.item()
            dofp2 = todofp(output2)
            save_img(dofp2, 'dofp_out', j, (epoch + last_epoch), data)
            dofp3 = todofp(label)
            save_img(dofp3, 'dofp_label', j, (epoch + last_epoch), data)
            dofp4 = todofp(img)
            save_img(dofp4, 'dofp_input', j, (epoch + last_epoch), data)

            S0_in, S1_in, S2_in = cal_s(img)
            input_dolp = cal_dolp(S0_in, S1_in, S2_in)
            input_aop = cal_aop(S1_in, S2_in)
            S0, S1, S2 = cal_s(output2)
            img_dolp = cal_dolp(S0, S1, S2)
            img_aop = cal_aop(S1, S2)
            S0_gt, S1_gt, S2_gt = cal_s(label)
            label_dolp = cal_dolp(S0_gt, S1_gt, S2_gt)
            label_aop = cal_aop(S1_gt, S2_gt)
            val_psnr += psnr1(output2, label).item()
            val_psnr_dolp += psnr1(img_dolp, label_dolp).item()
            val_psnr_aop += psnr1(img_aop, label_aop).item()
            save_img(img_dolp, 'dolp_out', j, (epoch + last_epoch), data)
            save_img(img_aop, 'aop_out', j, (epoch + last_epoch), data)
            save_img(label_dolp, 'dolp_label', j, (epoch + last_epoch), data)
            save_img(label_aop, 'aop_label', j, (epoch + last_epoch), data)
            save_img(input_dolp, 'dolp_input', j, (epoch + last_epoch), data)
            save_img(input_aop, 'aop_input', j, (epoch + last_epoch), data)
            img = toBGR(img)
            save_img(img, 'input', j, (epoch + last_epoch), data)
            output2 = toBGR(output2)
            save_img(output2, 'output', j, (epoch + last_epoch), data)
            label = toBGR(label)
            save_img(label, 'label', j, (epoch + last_epoch), data)


            writer1.add_scalars('Loss', {'Valid': val_loss_mean / len(vaild_loader1)}, iter)
            writer1.add_scalars('Valid_PSNR', {'DOFP': val_psnr / len(vaild_loader1), 'DOLP': val_psnr_dolp / len(vaild_loader1), 'AOP': val_psnr_aop / len(vaild_loader1)}, iter)
        print('Validation: Epoch[{:0>3}/{:0>3}]  Loss[{:.7f}] PSNR[{:.4f}] PSNR_Dolp[{:.4f}] PSNR_aop[{:.4f}]'.format(epoch+last_epoch, Epoch1+last_epoch,val_loss_mean / len(vaild_loader1),
                                                                                                                      val_psnr / len(vaild_loader1),val_psnr_dolp / len(vaild_loader1),val_psnr_aop / len(vaild_loader1)))

        # val_loss_mean=0
        # val_psnr=0
        # for j,data in enumerate(vaild_loader1):
        #     # 前向传播forward
        #     img, label = data
        #     img = img.to(device)
        #     # middle = middle.to(device)
        #     label = label.to(device)
        #     output2 = net(img)
        #     loss = Loss2(output2, label)
        #     val_loss_mean+=loss.item()
        #     val_psnr += psnr1(output2, label).item()
        #     # img = img * (torch.mean(label) / torch.mean(img))
        #     img=toBGR(img)
        #     img.squeeze_(0)
        #     img = toPIL(img)
        #     img.save('./val_output/8.18/%s/'%(epoch+last_epoch)+'%s_in.png'%j)
        #     output2=toBGR(output2)
        #     output2.squeeze_(0)
        #     output2 = toPIL(output2)
        #     output2.save('./val_output/8.18/%s/'%(epoch+last_epoch)+'%s_out.png'%j)
        #     label = toBGR(label)
        #     label.squeeze_(0)
        #     label = toPIL(label)
        #     label.save('./val_output/8.18/%s/'%(epoch+last_epoch)+'%s_label.png'%j)



        # writer1.add_scalars('Loss', {'Valid': val_loss_mean / len(vaild_loader1)}, iter)
        # writer1.add_scalars('Valid_PSNR', {'RGB': val_psnr / len(vaild_loader1)}, iter)
        # print('Validation: Epoch[{:0>3}/{:0>3}]  Loss[{:.7f}] PSNR[{:.4f}] PSNR_Dolp[{:.4f}] PSNR_aop[{:.4f}]'.format(epoch+1+last_epoch2, Epoch2+last_epoch2,val_loss_mean / len(vaild_loader2),
        #                                                                                                                           val_psnr / len(vaild_loader2),val_psnr_dolp / len(vaild_loader2),val_psnr_aop / len(vaild_loader2)))



    if not os.path.isdir('./checkpoint/' + data):
        os.makedirs('./checkpoint/' + data)
    # if val_loss_mean>temp:
    #     count+=1
    #     if count>5:
    #         lr = lr * 0.1
    #         for param_group in optimizer.param_groups:
    #             param_group['lr'] = lr
    #             count=0
    # else:
    #     temp=val_loss_mean
    old_lr = optimizer.param_groups[0]['lr']
    scheduler.step()
    lr = optimizer.param_groups[0]['lr']
    print('learning rate %.12f -> %.12f' % (old_lr, lr))
    checkpoint = {"model_state_dict": net.state_dict(),
                  "optimizer_state_dict": optimizer.state_dict(),
                  "epoch": epoch + 1 + last_epoch,
                  # 'train_sequence': 1,
                  'iter': iter,
                  # 'temp': temp,
                  'lr': lr}
    path_checkpoint = './checkpoint/' + data + '/checkpoint_epoch_%s.pkl' % (epoch + last_epoch)
    torch.save(checkpoint, path_checkpoint)



# if os.path.isdir('./checkpoint/%s_2'%data2):
#     checkpoint=torch.load('./checkpoint/%s_2/checkpoint_epoch_39.pkl'%data2)
#     net.load_state_dict(checkpoint['model_state_dict'])
#     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#     last_epoch2=checkpoint['epoch']
#     Epoch2=Epoch2-last_epoch2
#     iter=checkpoint['iter']
#     temp=checkpoint['temp']
#     # lr = checkpoint['lr']
#     # for param_group in optimizer.param_groups:
#     #     param_group['lr'] = lr
#
#
#
# for epoch in range(Epoch2):
#     loss_mean=0
#     net.train()
#
#
#     for i,data in enumerate(train_loader2):
#         #前向传播forward
#         img,label=data
#         img=img.to(device)
#         label=label.to(device)
#         # img=img*(label.mean()/img.mean())
#
#         #后向传播backward
#
#         optimizer.zero_grad()
#         output2= net(img)
#         loss=Loss(output2,label)
#         loss.backward()
#         iter+=1
#         loss_mean+=loss.item()
#
#         #更新权重
#         optimizer.step()
#         if iter%5==0:
#             S0,S1,S2=cal_s(output2)
#             img_dolp=cal_dolp(S0,S1,S2)
#             img_aop=cal_aop(S1,S2)
#             S0_gt,S1_gt,S2_gt=cal_s(label)
#             label_dolp=cal_dolp(S0_gt,S1_gt,S2_gt)
#             label_aop=cal_aop(S1_gt,S2_gt)
#             psnr=psnr1(output2,label).item()
#             psnr_dolp=psnr1(img_dolp,label_dolp).item()
#             psnr_aop=psnr1(img_aop,label_aop).item()
#             print('Training: Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] time[{:.4f}min]'
#                   'Loss[{:.7f}] PSNR[{:.4f}] PSNR_Dolp[{:.4f}] PSNR_aop[{:.4f}]'.format(epoch+1+last_epoch2,Epoch2+last_epoch2,i+1,len(train_loader2),(time.time()-time0)/60,loss_mean/5,
#                                                                                         psnr,psnr_dolp,psnr_aop))
#             loss_mean=0
#             writer2.add_scalars('PSNR',{'DOFP':psnr,'DOLP':psnr_dolp,'AOP':psnr_aop},iter)
#         writer2.add_scalars('Loss',{'Train':loss.item()},iter)
#         # writer.add_scalars('Loss_MSE', {'Train': loss1.item()}, iter)
#
#
#     if not os.path.isdir('./poldata/val_output/%s'%data2+'/%s'%(epoch+last_epoch2)):
#         os.makedirs('./poldata/val_output/%s'%data2+'/%s'%(epoch+last_epoch2))
#     net.eval()
#     with torch.no_grad():
#         val_loss_mean=0
#         val_psnr=0
#         val_psnr_dolp = 0
#         val_psnr_aop = 0
#         for j,data in enumerate(vaild_loader2):
#             # 前向传播forward
#             img,label = data
#             img = img.to(device)
#             label = label.to(device)
#             # img = img * (label.mean() / img.mean())
#             output2 = net(img)
#             loss = Loss(output2, label)
#             val_loss_mean+=loss.item()
#             dofp2 = todofp(output2)
#             save_img(dofp2, 'dofp_out', j, (epoch + last_epoch2), data2)
#             dofp3 = todofp(label)
#             save_img(dofp3, 'dofp_label', j, (epoch + last_epoch2), data2)
#             dofp4 = todofp(img)
#             save_img(dofp4, 'dofp_input', j, (epoch + last_epoch2), data2)
#
#             S0_in, S1_in, S2_in = cal_s(img)
#             input_dolp = cal_dolp(S0_in, S1_in, S2_in)
#             input_aop = cal_aop(S1_in, S2_in)
#             S0, S1, S2 = cal_s(output2)
#             img_dolp = cal_dolp(S0, S1, S2)
#             img_aop = cal_aop(S1, S2)
#             S0_gt, S1_gt, S2_gt = cal_s(label)
#             label_dolp = cal_dolp(S0_gt, S1_gt, S2_gt)
#             label_aop = cal_aop(S1_gt, S2_gt)
#             val_psnr += psnr1(output2, label).item()
#             val_psnr_dolp += psnr1(img_dolp, label_dolp).item()
#             val_psnr_aop += psnr1(img_aop, label_aop).item()
#             save_img(img_dolp, 'dolp_out', j, (epoch + last_epoch2), data2)
#             save_img(img_aop, 'aop_out', j, (epoch + last_epoch2), data2)
#             save_img(label_dolp, 'dolp_label', j, (epoch + last_epoch2), data2)
#             save_img(label_aop, 'aop_label', j, (epoch + last_epoch2), data2)
#             save_img(input_dolp, 'dolp_input', j, (epoch + last_epoch2), data2)
#             save_img(input_aop, 'aop_input', j, (epoch + last_epoch2), data2)
#             img = toBGR(img)
#             save_img(img, 'input', j, (epoch + last_epoch2), data2)
#             output2 = toBGR(output2)
#             save_img(output2, 'output', j, (epoch + last_epoch2), data2)
#             label = toBGR(label)
#             save_img(label, 'label', j, (epoch + last_epoch2), data2)
#
#
#         writer2.add_scalars('Loss', {'Valid': val_loss_mean / len(vaild_loader2)}, iter)
#         writer2.add_scalars('Valid_PSNR', {'DOFP': val_psnr / len(vaild_loader2), 'DOLP': val_psnr_dolp / len(vaild_loader2), 'AOP': val_psnr_aop / len(vaild_loader2)}, iter)
#         print('Validation: Epoch[{:0>3}/{:0>3}]  Loss[{:.7f}] PSNR[{:.4f}] PSNR_Dolp[{:.4f}] PSNR_aop[{:.4f}]'.format(epoch+1+last_epoch2, Epoch2+last_epoch2,val_loss_mean / len(vaild_loader2),
#                                                                                                                       val_psnr / len(vaild_loader2),val_psnr_dolp / len(vaild_loader2),val_psnr_aop / len(vaild_loader2)))
#
#     if not os.path.isdir('./checkpoint/%s_2'%data2):
#         os.makedirs('./checkpoint/%s_2'%data2)
#     if val_loss_mean>temp:
#         count+=1
#         if count>5:
#             lr = lr * 0.1
#             for param_group in optimizer.param_groups:
#                 param_group['lr'] = lr
#                 count = 0
#     else:
#         temp=val_loss_mean
#         checkpoint = {"model_state_dict": net.state_dict(),
#                       "optimizer_state_dict": optimizer.state_dict(),
#                       "epoch": epoch + 1 + last_epoch2,
#                       'train_sequence': 2,
#                       'iter': iter,
#                       'temp': temp}
#         path_checkpoint = "./checkpoint/%s_2"%data2+"/checkpoint_epoch_%s.pkl" % (epoch + 1 + last_epoch2)
#         torch.save(checkpoint, path_checkpoint)
#         count=0














