import torch.nn as nn
import torch
import torch.nn.functional as F
from torchsummary import summary

class UNet(nn.Module):
    def __init__(self,grad=True):
        super(UNet,self).__init__()
        self.con1=nn.Conv2d(3,32,3,padding=1)
        self.con1_2 = nn.Conv2d(32, 32, 3, padding=1)
        self.con2=nn.Conv2d(32,64,3,padding=1)
        self.con2_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.con3=nn.Conv2d(64,128,3,padding=1)
        self.con3_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.con4=nn.Conv2d(128,256,3,padding=1)
        self.con4_2 = nn.Conv2d(256,256, 3, padding=1)
        self.con5=nn.Conv2d(256,512,3,padding=1)
        self.con5_2 = nn.Conv2d(512,512, 3, padding=1)
        for p in self.parameters():
            p.requires_grad = grad
        self.up6=nn.ConvTranspose2d(512,256,2,2)
        self.con6 = nn.Conv2d(512, 256, 3, padding=1)
        self.con6_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.up7=nn.ConvTranspose2d(256,128,2,2)
        self.con7 = nn.Conv2d(256, 128, 3, padding=1)
        self.con7_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.up8=nn.ConvTranspose2d(128,64,2,2)
        self.con8 = nn.Conv2d(128, 64, 3, padding=1)
        self.con8_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.up9=nn.ConvTranspose2d(64,32,2,2)
        self.con9 = nn.Conv2d(64, 32, 3, padding=1)
        self.con9_2 = nn.Conv2d(32, 32, 3, padding=1)
        self.con10 = nn.Conv2d(32, 3, 1, padding=0)


    def forward(self,x):
        x1=F.leaky_relu(self.con1_2(F.leaky_relu(self.con1(x),0.2)),0.2)
        x2=F.max_pool2d(x1,2)
        x2=F.leaky_relu(self.con2_2(F.leaky_relu(self.con2(x2),0.2)),0.2)
        x3 =F.max_pool2d(x2,2)
        x3=F.leaky_relu(self.con3_2(F.leaky_relu(self.con3(x3),0.2)),0.2)
        x4 =F.max_pool2d(x3,2)
        x4=F.leaky_relu(self.con4_2(F.leaky_relu(self.con4(x4),0.2)),0.2)
        x5 =F.max_pool2d(x4,2)
        x5=F.leaky_relu(self.con5_2(F.leaky_relu(self.con5(x5),0.2)),0.2)

        x6=torch.cat([self.up6(x5),x4],1)
        x6=F.leaky_relu(self.con6_2(F.leaky_relu(self.con6(x6),0.2)),0.2)
        x7=torch.cat([self.up7(x6),x3],1)
        x7=F.leaky_relu(self.con7_2(F.leaky_relu(self.con7(x7),0.2)),0.2)
        x8=torch.cat([self.up8(x7),x2],1)
        x8=F.leaky_relu(self.con8_2(F.leaky_relu(self.con8(x8),0.2)),0.2)
        x9=torch.cat([self.up9(x8),x1],1)
        x9=F.leaky_relu(self.con9_2(F.leaky_relu(self.con9(x9),0.2)),0.2)

        out=self.con10(x9)
        return out

    def initialize_weight(self,w_mean=0,w_std=0.01):
        for m in self.modules():
            classname=m.__class__.__name__
            if classname.find('Conv')==0:
                nn.init.normal_(m.weight.data,w_mean,w_std)


if __name__ =='__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net=UNet()
    net.to(device)
    # net.initialize_weight(0,0.01)
    net.eval()
    with torch.no_grad():
        summary(net,(4,1024,1216),32,device='cuda')







