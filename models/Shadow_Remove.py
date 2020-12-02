import torch
import torch.nn as nn
from torch.autograd import Variable

class Conv_deconv(nn.Module):
    def __init__(self):
        super(Conv_deconv,self).__init__()
		
		#Encoder!
		
        #Convolution 1
        self.conv1=nn.Conv2d(in_channels=4,out_channels=16, kernel_size=5, stride=1, padding = 0)
        nn.init.kaiming_normal_(self.conv1.weight) #Xaviers Initialisation
        self.conv1_bn = nn.BatchNorm2d(16)  
        self.swish1= nn.LeakyReLU()

        #Max Pool 1
        self.maxpool1= nn.MaxPool2d(kernel_size=2,return_indices=True)

        #Convolution 2
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5)
        nn.init.kaiming_normal_(self.conv2.weight)
        self.conv2_bn = nn.BatchNorm2d(32) 
        self.swish2 = nn.LeakyReLU()
        #Convolution 3
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3)
        nn.init.kaiming_normal_(self.conv3.weight)
        self.conv3_bn = nn.BatchNorm2d(32) 
        self.swish3 = nn.LeakyReLU()

        #Max Pool 1
        self.maxpool2= nn.MaxPool2d(kernel_size=2,return_indices=True)

        #Convolution 4
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        nn.init.kaiming_normal_(self.conv4.weight)
        self.conv4_bn = nn.BatchNorm2d(64) 
        self.swish4 = nn.LeakyReLU()
        #Convolution 5
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)
        nn.init.kaiming_normal_(self.conv5.weight)
        self.conv5_bn = nn.BatchNorm2d(64) 
        self.swish5 = nn.LeakyReLU()

		#Decoder!
        #De Convolution 1
        self.deconv1 = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3)
        nn.init.kaiming_normal_(self.deconv1.weight)
        self.deconv1_bn = nn.BatchNorm2d(64) 
        self.deswish1 = nn.LeakyReLU()

        #De Convolution 2
        self.deconv2 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3)
        nn.init.kaiming_normal_(self.deconv2.weight)
        self.deconv2_bn = nn.BatchNorm2d(32) 
        self.deswish2 = nn.LeakyReLU()
  
        #Max UnPool 1
        self.maxunpool1=nn.MaxUnpool2d(kernel_size=2) 

        #De Convolution 3
        self.deconv3 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3)
        nn.init.kaiming_normal_(self.deconv3.weight)
        self.deconv3_bn = nn.BatchNorm2d(32) 
        self.deswish3 = nn.LeakyReLU()

        #De Convolution 4
        self.deconv4 = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=5)
        nn.init.kaiming_normal_(self.deconv4.weight)
        self.deconv4_bn = nn.BatchNorm2d(16) 
        self.deswish4 = nn.LeakyReLU()
        

        #Max UnPool 2
        self.maxunpool2=nn.MaxUnpool2d(kernel_size=2) 

        #De Convolution 5
        self.deconv5=nn.ConvTranspose2d(in_channels=16,out_channels=3,kernel_size=5)
        nn.init.kaiming_normal_(self.deconv5.weight) 


    def forward(self,x):
        residual = torch.narrow(x, 1, 0, 3)
        out=self.conv1(x)
        out=self.conv1_bn(out)
        out=self.swish1(out)
        size1 = out.size()
        out,indices1=self.maxpool1(out)
        
        out=self.conv2(out)
        out = self.conv2_bn(out)
        out=self.swish2(out)
        out=self.conv3(out)
        out=self.conv3_bn(out)
        skip = self.swish3(out)
        size2 = skip.size()
        out,indices2=self.maxpool2(skip)

        out=self.conv4(out)
        out = self.conv4_bn(out)
        out=self.swish4(out)
        out=self.conv5(out)
        out=self.conv5_bn(out)
        out=self.swish5(out)


        out=self.deconv1(out)
        out=self.deconv1_bn(out)
        out=self.deswish1(out)
        out=self.deconv2(out)
        out=self.deconv2_bn(out)
        out=self.deswish2(out)
        out=self.maxunpool1(out,indices2,size2)
        out = torch.cat([out, skip], dim=1)

        out=self.deconv3(out)
        out=self.deconv3_bn(out)
        out=self.deswish3(out)
        out=self.deconv4(out)
        out=self.deconv4_bn(out)
        out=self.deswish4(out)
        out=self.maxunpool2(out,indices1,size1)
        out=self.deconv5(out)
        return(out + residual)