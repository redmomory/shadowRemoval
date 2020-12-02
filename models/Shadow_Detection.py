import torch
import torch.nn as nn
from torch.autograd import Variable

class conv_deconv(nn.Module):
    def __init__(self):
        super(conv_deconv,self).__init__()
        #Convolution 1
        self.conv1=nn.Conv2d(in_channels=3,out_channels=8, kernel_size=5, stride=1, padding=2)
        nn.init.kaiming_normal_(self.conv1.weight) #Xaviers Initialisation
        self.conv1_bn = nn.BatchNorm2d(8)
        self.swish1= nn.ReLU()

        #Max Pool 1
        self.maxpool1= nn.MaxPool2d(kernel_size=2,return_indices=True)

        #Convolution 2
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=5, padding=2)
        nn.init.kaiming_normal_(self.conv2.weight)
        self.conv2_bn = nn.BatchNorm2d(16)
        self.swish2 = nn.ReLU()

        #Max Pool 2
        self.maxpool2 = nn.MaxPool2d(kernel_size=2,return_indices=True)

        #Convolution 3
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        nn.init.kaiming_normal_(self.conv3.weight)
        self.conv3_bn = nn.BatchNorm2d(32)
        self.swish3 = nn.ReLU()
		
        #Max Pool 3
        self.maxpool3 = nn.MaxPool2d(kernel_size=2,return_indices=True)		
		
		#Convolution 4
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=8, kernel_size=3, padding=1)
        nn.init.kaiming_normal_(self.conv4.weight)
        self.conv4_bn = nn.BatchNorm2d(8)
        self.swish4 = nn.ReLU()		
		
		#FC1
        self.fc1 = nn.Linear(15*20*8, 15*20*2)
        self.fc1_bn = nn.BatchNorm1d(num_features=15*20*2)
        self.fswish1 = nn.ReLU()
		
		#FC2
        self.fc2 = nn.Linear(15*20*2, 15*10)
        self.fc2_bn = nn.BatchNorm1d(num_features=15*10)
        self.fswish2 = nn.ReLU()
		
		#DFC1
        self.dfc1 = nn.Linear(15*10, 15*20*2)
        self.dfc1_bn = nn.BatchNorm1d(num_features=15*20*2)
        self.dfswish1 = nn.ReLU()
		
		#DFC2
        self.dfc2 = nn.Linear(15*20*2, 15*20*8)
        self.dfc2_bn = nn.BatchNorm1d(num_features=15*20*8)
        self.dfswish2 = nn.ReLU()
		
		#De Convolution 1
        self.deconv1=nn.ConvTranspose2d(in_channels=16,out_channels=32,kernel_size=3, padding = 1)
        nn.init.kaiming_normal_(self.deconv1.weight)
        self.deconv1_bn = nn.BatchNorm2d(32)
        self.dswish1=nn.ReLU()
		
		#Max UnPool 1
        self.maxunpool1=nn.MaxUnpool2d(kernel_size=2)
		
        #De Convolution 2
        self.deconv2=nn.ConvTranspose2d(in_channels=64,out_channels=16,kernel_size=3, padding = 1)
        nn.init.kaiming_normal_(self.deconv2.weight)
        self.deconv2_bn = nn.BatchNorm2d(16)
        self.dswish2=nn.ReLU()

        #Max UnPool 2
        self.maxunpool2=nn.MaxUnpool2d(kernel_size=2)

        #De Convolution 3
        self.deconv3=nn.ConvTranspose2d(in_channels=16,out_channels=8,kernel_size=5, padding = 2)
        nn.init.kaiming_normal_(self.deconv3.weight)
        self.deconv3_bn = nn.BatchNorm2d(8)
        self.dswish3=nn.ReLU()

        #Max UnPool 3
        self.maxunpool3=nn.MaxUnpool2d(kernel_size=2)

        #DeConvolution 4
        self.deconv4=nn.ConvTranspose2d(in_channels=8,out_channels=1,kernel_size=5, padding = 2)
        nn.init.kaiming_normal_(self.deconv4.weight)
        self.deconv4_bn = nn.BatchNorm2d(1)
        self.dswish4 = nn.Sigmoid()

        #Last Conv
        self.conv5=nn.Conv2d(in_channels=1,out_channels=16, kernel_size=7, stride=1, padding=3)
        nn.init.kaiming_normal_(self.conv5.weight) #Xaviers Initialisation
        self.conv5_bn = nn.BatchNorm2d(16)
        self.swish5= nn.ReLU()

        self.conv6=nn.Conv2d(in_channels=16,out_channels=1, kernel_size=5, stride=1, padding=2)
        nn.init.kaiming_normal_(self.conv6.weight) #Xaviers Initialisation
        self.swish6= nn.Sigmoid()



    def forward(self,x):
        #Encoder
        out=self.conv1(x)
        out=self.conv1_bn(out)
        out=self.swish1(out)
        size1 = out.size()
        out,indices1=self.maxpool1(out)
        
        out=self.conv2(out)
        out=self.conv2_bn(out)
        out=self.swish2(out)
        size2 = out.size()
        out,indices2=self.maxpool2(out)
        
        out=self.conv3(out)
        out=self.conv3_bn(out)
        skip2 = self.swish3(out)
        size3 = out.size()
        out, indices3=self.maxpool3(skip2)
		
        out=self.conv4(out)
        out=self.conv4_bn(out)
        skip1 = self.swish4(out)
        out = skip1.view([-1, 8*15*20])
        out=self.fc1(out)
        out=self.fc1_bn(out)
        out=self.fswish1(out)
        out=self.fc2(out)
        out=self.fc2_bn(out)
        out=self.fswish2(out)
      
		#Decoder
        out=self.dfc1(out)
        out=self.dfc1_bn(out)
        out=self.dfswish1(out)
        out=self.dfc2(out)
        out=self.dfc2_bn(out)
        out=self.dfswish2(out)
        out = out.view([-1,8,15,20])
        out = torch.cat([out, skip1], dim=1)
		
        out=self.deconv1(out)
        out=self.deconv1_bn(out)
        out=self.dswish1(out)
        out=self.maxunpool1(out,indices3,size3)
        out = torch.cat([out, skip2], dim=1)

        out=self.deconv2(out)
        out=self.deconv2_bn(out)
        out=self.dswish2(out)
        out=self.maxunpool2(out,indices2,size2)
		
        out=self.deconv3(out)
        out=self.deconv3_bn(out)
        out=self.dswish3(out)
        out=self.maxunpool3(out,indices1,size1)
		
        out=self.deconv4(out)
        out=self.deconv4_bn(out)
        out=self.dswish4(out)

        out=self.conv5(out)
        out=self.conv5_bn(out)
        out=self.swish5(out)
        out=self.conv6(out)
        out=self.swish6(out)

        return(out)