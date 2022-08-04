import torch
import torch.nn as nn
from tensorboardX import SummaryWriter

class vgg16(nn.Module):
    def __init__(self):
        super(vgg16, self).__init__()
        # 1
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # 2
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # 3
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # 4
        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # 5
        self.layer5 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
    
        self.conv=nn.Sequential(
                self.layer1,
                self.layer2,
                self.layer3,
                self.layer4,
                self.layer5
            )
        
        self.fc=nn.Sequential(
            # 6
            nn.Linear(512*7*7,4096),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5,),
            # 7
            nn.Linear(4096,4096),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5,),
            # 8
            nn.Linear(4096,1000),
            nn.ReLU(inplace=True)
        )
        
    def forward(self,x):
        x=self.conv(x)
        x = torch.flatten(x, 1)
        x = x.view(1, 512*7*7)
        x=self.fc(x)
        return x
    
vgg = vgg16()
data = torch.rand(1,3,224,224)

torch.onnx.export(vgg, data, 'D:/tmp/vgg_model2.onnx', export_params=True, opset_version=8)

with SummaryWriter(logdir="D:/tmp/network_visualization2") as w:
    w.add_graph(vgg, data)
