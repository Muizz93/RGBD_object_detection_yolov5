import torch
import torch.nn as nn
#Define the Convolutional Autoencoder
class ConvAutoencoder(nn.Module):
    def __init__(self, ch_rgb=3, ch_d=1):
        super(ConvAutoencoder, self).__init__()
       
        #Encoder
        self.conv1rgb = nn.Conv2d(ch_rgb, 32, 11, padding=2)
        self.conv1depth = nn.Conv2d(ch_d, 32, 11, padding=2)

        self.conv2 = nn.Conv2d(32, 64, 9, padding=2)
        self.conv3 = nn.Conv2d(64, 128, 7, padding=2)
        self.pool = nn.MaxPool2d(4, 4)
       
        #Decoder
        self.t_conv1 = nn.ConvTranspose2d(128, 64, 2, stride=4)
        self.t_conv2 = nn.ConvTranspose2d(64, 32, 2, stride=4)
        self.t_conv3rgb = nn.ConvTranspose2d(32, ch_rgb, 2, stride=4)
        self.t_conv3depth = nn.ConvTranspose2d(32, ch_d, 2, stride=4)
        
    def forward(self, rgb, depth):
        x = F.relu(self.conv1rgb(rgb))
        y = F.relu(self.conv1depth(depth))
        
        y += x
        x = F.relu(self.conv2(x))
        y = F.relu(self.conv2(y))

        x *= y
        x = F.relu(self.conv3(x))
        y = F.relu(self.conv3(y))

        y += x
        x = self.pool(x)
        y = self.pool(y)
        
        x *= y
        x = self.pool(x)
        y = self.pool(y)

        y += x        
        x = F.relu(self.t_conv1(x))
        y = F.relu(self.t_conv1(y))

        x *= y
        x = F.relu(self.t_conv2(x))
        y = F.relu(self.t_conv2(y))

        y += x
        x = F.relu(self.t_conv3rgb(x))
        y = F.relu(self.t_conv3depth(y))

        z = torch.cat(x,y)              
        return z
    
    #concatenate mymodel with  yolov5
    class MyEnsemble(nn.Module):
        def __init__(self, modelA, modelB):
            super(MyEnsemble).__init__():
            self.modelA = modelA
            self.modelB = modelB

        def forward(self, rgb, depth):
            x = self.sensor_fusion_model(rgb, depth)
            x = self.yolov5_4ch(x)
            return x