
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)  # Or nn.Hardswish() if matching MobileNetV3

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.act(x)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            DepthwiseSeparableConv(in_channels, out_channels),
            DepthwiseSeparableConv(out_channels, out_channels)
        )

    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        # self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=1)  # channel reduction
        )
        self.conv = ConvBlock(out_channels + skip_channels, out_channels)

        self.conv = ConvBlock(out_channels + skip_channels, out_channels)

    def forward(self, x, skip, use_skip = True):
        x = self.up(x)
    
        if use_skip:
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)

        
        return self.conv(x)


class Model(nn.Module):
    def __init__(self, num_classes=1, backbone_name='mobilenetv3_large_100', K=16):
        super().__init__()
        self.encoder = timm.create_model(backbone_name, features_only=True, pretrained=True)

        enc_channels = self.encoder.feature_info.channels()
        self.enc1_channels = enc_channels[0]  # stage1
        self.enc2_channels = enc_channels[1]  # stage2
        self.enc3_channels = enc_channels[2]  # stage3
        self.enc4_channels = enc_channels[3]  # stage4
        self.enc5_channels = enc_channels[4]  # stage5 (deepest)

        
        # Decoder
        self.center = ConvBlock(self.enc5_channels, K*32)
        self.up4 = UpBlock(K*32, self.enc4_channels, K*16)
        self.up3 = UpBlock(K*16, self.enc3_channels, K*8)
        self.up2 = UpBlock(K*8, self.enc2_channels, K*4)
        self.up1 = UpBlock(K*4, self.enc1_channels, K*2)
        self.up0 = UpBlock(K*2, 0, K)


        self.final_conv = nn.Conv2d(K, num_classes, kernel_size=1)

    def forward(self, x):
        features = self.encoder(x)
        enc1, enc2, enc3, enc4, enc5 = features

        x = self.center(enc5)

        x = self.up4(x, enc4)
        x = self.up3(x, enc3)
        x = self.up2(x, enc2)
        x = self.up1(x, enc1)
        x = self.up0(x, x, use_skip=False)
        x = self.final_conv(x)

        return x

if __name__ == '__main__':
    model = Model(num_classes=1, K=8)
    dummy_input = torch.randn(1, 3, 640, 640)
    output = model(dummy_input)

    print(output.shape)