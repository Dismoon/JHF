import torch.nn as nn
import torch
import torch.nn.functional as F
from UNET import UNet
from torchsummary import summary
from atten import *

class  UPN(nn.Module):
    def __init__(self,G0,c_dim):
        super(UPN,self).__init__()
        self.feature=nn.Sequential(
            nn.Conv2d(G0,64,5,padding=2),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(64, c_dim, 3, padding=1)
        )

    def forward(self,x):
        return self.feature(x)

    def initialize_weight(self,w_mean,w_std):
        for m in self.modules():
            classname=m.__class__.__name__
            if classname.find('Conv')==0:
                nn.init.normal_(m.weight.data,w_mean,w_std)

class RDBs_1(nn.Module):
    def __init__(self,C,ks,G):
        super(RDBs_1,self).__init__()
        self.rdbs_1=nn.ModuleList([])
        for j in range(1,C+1):
            self.rdbs_1.append(nn.Conv2d(G*j,G,ks,padding=int((ks-1)/2)))

    def forward(self,x):
        for layers in self.rdbs_1:
            tmp=layers(x)
            tmp=F.relu(tmp,True)
            x=torch.cat([x,tmp],1)

        return x
    def initialize_weight(self):
        for m in self.modules():
            classname=m.__class__.__name__
            if classname.find('Conv')==0:
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                # nn.init.normal(m.weight.data,w_mean,w_std)

class RDBs2(nn.Module):
    def __init__(self,C,ks,G,k):
        super(RDBs2,self).__init__()
        self.rdbs2=nn.ModuleList([])
        self.rdbs2.append(RDBs_1(C,ks,G*k))
        self.rdbs2.append(nn.Conv2d(G *k* (C+1), G*k, 1))

    def forward(self,input):
        rdb_in=input
        for i,layers in enumerate(self.rdbs2):
            if i%2 == 0:
                x=rdb_in
            x=layers(x)
            if str(layers).find('RDBs_1') != 0:
                rdb_in=torch.add(x,rdb_in)



        return rdb_in

    def initialize_weight(self,w_mean,w_std):
        for m in self.modules():
            classname=m.__class__.__name__
            if classname.find('Conv')==0:
                nn.init.normal(m.weight.data,w_mean,w_std)
class RDBs3(nn.Module):
    def __init__(self,C,D,ks,G):
        super(RDBs3,self).__init__()
        self.rdbs=nn.ModuleList([])
        for i in range(1, D + 1):
            self.rdbs.append(RDBs2(C,ks,G,i))
            self.rdbs.append(nn.Conv2d(G*i, G, 1))

    def forward(self,input):
        rdb_in=input
        for i,layers in enumerate(self.rdbs):
            if i%2 == 0:
                x=rdb_in
            x=layers(x)
            if str(layers).find('RDBs2') != 0:
                # rdb_in=torch.add(x,rdb_in)
                x = F.relu(x, True)
                rdb_in = torch.cat([rdb_in, x], 1)



        return rdb_in

    def initialize_weight(self,w_mean,w_std):
        for m in self.modules():
            classname=m.__class__.__name__
            if classname.find('Conv')==0:

                nn.init.normal(m.weight.data,w_mean,w_std)

class RDBs(nn.Module):
    def __init__(self,C,D,ks,G,grad=True):
        super(RDBs,self).__init__()
        self.rdbs=nn.ModuleList([])
        for i in range(1,D+1):
            self.rdbs.append(RDBs_1(C,ks,G))
            self.rdbs.append(nn.Conv2d(G * (C+1), G, 1))
            # if i==D/2:
            #     for p in self.parameters():
            #         p.requires_grad = grad

    def forward(self,input):
        rdb_in=input
        rdb_concat = list()
        for i,layers in enumerate(self.rdbs):
            if i%2 == 0:
                x=rdb_in
            x=layers(x)
            if str(layers).find('RDBs_1') != 0:
                rdb_in=torch.add(x,rdb_in)
                rdb_concat.append(rdb_in)


        return torch.cat(rdb_concat,1)

    def initialize_weight(self):
        for m in self.modules():
            classname=m.__class__.__name__
            if classname.find('Conv')==0:
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                # nn.init.normal(m.weight.data,w_mean,w_std)
            elif classname.find('RDBs_1')==0:
                m.initialize_weight()




class model(nn.Module):
    def __init__(self,c_dim,G0,ks,C,D,G,grad=True):
        super(model,self).__init__()
        self.conv1=nn.Conv2d(c_dim,G0,ks,padding=int((ks-1)/2))
        self.conv2 =nn.Conv2d(G0,G,ks,padding=int((ks-1)/2))
        # for p in self.parameters():
        #     p.requires_grad = grad
        self.RDB =RDBs(C,D,ks,G,grad=grad)
        self.conv3 =nn.Conv2d(G*D, G0, 1, padding=0)
        self.conv4 =nn.Conv2d(G0, G0, ks, padding=int((ks-1)/2))
        self.UPN =UPN(G0,c_dim)
        self.conv5 =nn.Conv2d(c_dim,c_dim,ks,padding=int((ks-1)/2))
        # for p in self.parameters():
        #     print(p.requires_grad)
        # if grad==True:
        #     print("No trans")
        # else:
        #     print("Trans")
        # self.body=nn.ModuleDict({
        #     'conv1':nn.Conv2d(c_dim,G0,ks,padding=int((ks-1)/2)),
        #     'conv2':nn.Conv2d(G0,G,ks,padding=int((ks-1)/2)),
        #     'RDB':RDBs(C,D,ks,G),
        #     'conv3': nn.Conv2d(G*D, G0, 1, padding=0),
        #     'conv4': nn.Conv2d(G0, G0, ks, padding=int((ks-1)/2)),
        #     'UPN': UPN(G0,c_dim),
        #     'conv5':nn.Conv2d(c_dim,c_dim,ks,padding=int((ks-1)/2))
        # }
        # )



    def forward(self,x):
        F_1=self.conv1(x)
        F0=self.conv2(F_1)
        FD=self.RDB(F0)
        FGF1=self.conv3(FD)
        FGf2=self.conv4(FGF1)

        FDF=torch.add(FGf2,F_1)

        FU=self.UPN(FDF)
        map=self.conv5(FU)
        denoise= x - map
        # F_1=self.body['conv1'](x)
        # F0=self.body['conv2'](F_1)
        # FD=self.body['RDB'](F0)
        # FGF1=self.body['conv3'](FD)
        # FGf2=self.body['conv4'](FGF1)
        #
        # FDF=torch.add(FGf2,F_1)
        #
        # FU=self.body['UPN'](FDF)
        # map=self.body['conv5'](FU)
        # denoise=x+map

        return denoise

    def initialize_weight(self,w_mean=0,w_std=0.01):
        for m in self.modules():
            classname=m.__class__.__name__
            if classname.find('Conv')==0:
                nn.init.normal_(m.weight.data,w_mean,w_std)
            elif classname.find('RDB')==0:
                m.initialize_weight()
            elif classname.find('UPN')==0:
                m.initialize_weight(w_mean,w_std)


class model2(nn.Module):
    def __init__(self,c_dim,G0,ks,C,D,G):
        super(model2,self).__init__()
        self.body=nn.ModuleDict({
            'RDN1':model(c_dim, G0, ks, C, D, G),
            'UNet':UNet(),
            'RDN2':model(c_dim,G0,ks,C,D,G),
        }
        )
        for p in self.parameters():
            print(p.requires_grad)



    def forward(self,x):
        x1 = self.body['RDN1'](x)
        x2 = x1+self.body['UNet'](x1)
        x3 = self.body['RDN2'](x2)


        return x3

    def initialize_weight(self,w_mean=0,w_std=0.01):
        for m in self.modules():
            classname=m.__class__.__name__

            if classname.find('Conv')==0:
                nn.init.normal_(m.weight.data,w_mean,w_std)
if __name__ =='__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net=model(c_dim=3,G0=64,ks=3,C=6,D=16,G=32)
    net.to(device)
    net.initialize_weight(0,0.01)
    summary(net,(3,64,64),32,device='cuda')



