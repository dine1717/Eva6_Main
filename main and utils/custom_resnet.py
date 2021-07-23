import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, count, pre, stride=1):
        super(BasicBlock, self).__init__()

        # preprocessing first layer 
        if pre  == 'p':
            self.layer1_preprocessing = nn.Sequential(transforms.RandomCrop((32,32), padding  = 4))
        else:
            self.layer1_preprocessing = nn.Sequential()
        # first custom layer
        self.layer1 = nn.Sequential(nn.Conv2d(in_planes, planes, kernel_size=  3, stride = 1, padding =1),
                                    nn.MaxPool2d(kernel_size = (2,2)),
                                    nn.BatchNorm2d(planes),
                                    nn.ReLU())
       
        # residual block

        if count =='a':
            self.resblock = nn.Sequential(nn.Conv2d(planes, planes, kernel_size=  3, stride = 1, padding =1),
                                        nn.BatchNorm2d(planes),
                                        nn.ReLU(),
                                        nn.Conv2d(planes, planes, kernel_size= 3, stride = 1, padding =1),
                                        nn.BatchNorm2d(planes),
                                        nn.ReLU())
            
            # shortcut for the residual blocl
            # self.shortcut = nn.Sequential(nn.Conv2d(planes, self.expansion*planes,kernel_size=1, stride=stride, bias=False),
            #                               nn.BatchNorm2d(self.expansion*planes),
            #                               nn.ReLU()
            #                               )
            
            # self.dropout = nn.Sequential(nn.Dropout2d())
        else:
            self.resblock = nn.Sequential()
            # self.shortcut = nn.Sequential()

    def forward(self, x):
        out = self.layer1_preprocessing(x)
        out = self.layer1(out)
        res_out = self.resblock(out)
        # short_out = self.shortcut(out)
        # out = res_out + short_out       # with shortcut
        out = res_out + out  # without shortcut
        # print(out.shape)         
        return out # trial one without drop out 
        # return F.dropout2d(out)

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 128, num_blocks[0], count = 'a',pre = 'p' ,stride=1, )
        self.layer2 = self._make_layer(block, 256, num_blocks[1], count = 'b',pre = 'n' , stride=1)
        self.layer3 = self._make_layer(block, 512, num_blocks[2], count = 'a', pre = 'n' ,stride=1)        
        self.linear = nn.Linear(512*block.expansion, 1024)
        self.final = nn.Linear(1024,10 )

    def _make_layer(self, block, planes, num_blocks, count,pre,stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, count, pre,stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.max_pool2d(out,4)
        out = out.view(out.size(0),-1)
        out = self.linear(out)
        out = self.final(out)
        return out
    
def ResNet18new():
    return ResNet(BasicBlock, [1, 1, 1])

