import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_module import CBLModule
from torchsummary import summary


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)



class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=False):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        # diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        # diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        # x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
        #                 diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.conv(x))



class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # self.inc = DoubleConv(n_channels, 64)
        # self.down1 = Down(64, 128)
        # self.down2 = Down(128, 256)
        # self.down3 = Down(256, 512)
        # self.down4 = Down(512, 1024)
        # self.up1 = Up(1024, 512, bilinear)
        # self.up2 = Up(512, 256, bilinear)
        # self.up3 = Up(256, 128, bilinear)
        # self.up4 = Up(128, 64, bilinear)
        # self.outc = OutConv(64, n_classes)
        self.inc = DoubleConv(n_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        self.down4 = Down(256, 512)
        self.up1 = Up(512, 256, bilinear)
        self.cbl_1_8 = CBLModule(256, 256*5//4)
        self.bce_1_8 = OutConv(256*5//4, 1)

        self.up2 = Up(256, 128, bilinear)
        self.cbl_1_4 = CBLModule(128, 128*5//2)
        self.bce_1_4 = OutConv(128*5//2, 1)

        self.up3 = Up(128, 64, bilinear)
        self.cbl_1_2 = CBLModule(64, 64*5//2)
        self.bce_1_2 = OutConv(64*5//2, 1)

        self.up4 = Up(64, 32, bilinear)
        self.outc = OutConv(32, n_classes)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.normal_(m.weight, std=0.001)
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x1 = self.inc(x)
        # print('dddddd',x1)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        
        cbl_1_8 = self.cbl_1_8(x)
        bce_1_8 = self.bce_1_8(cbl_1_8)

        x = self.up2(x*(bce_1_8+1), x3)
        cbl_1_4 = self.cbl_1_4(x)
        bce_1_4 = self.bce_1_4(cbl_1_4)

        x = self.up3(x*(bce_1_4+1), x2)
        cbl_1_2 = self.cbl_1_2(x)
        bce_1_2 = self.bce_1_2(cbl_1_2)

        x = self.up4(x*(bce_1_2+1), x1)
        logits = self.outc(x)

        return logits, cbl_1_8, bce_1_8, cbl_1_4, bce_1_4, cbl_1_2, bce_1_2
    



# if __name__ == '__main__':

#     model = UNet(n_channels=1, n_classes=1).cuda()
    # summary(model,(1,256,256))

    # for name in model.state_dict():
    #     print(name)