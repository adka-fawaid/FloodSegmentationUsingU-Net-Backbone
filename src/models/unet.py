import torch.nn as nn
import torch
import torchvision.models as models

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self,x): return self.net(x)

class UNet(nn.Module):
    """
    U-Net with flexible encoder options:
    - 'baseline': Standard U-Net encoder (default)
    - 'resnet50': ResNet50 encoder with ImageNet pretrained weights
    - 'efficientnet_b1': EfficientNet-B1 encoder with ImageNet pretrained weights
    """
    def __init__(self, in_ch=3, n_classes=1, base_c=32, encoder='baseline'):
        super().__init__()
        self.encoder_type = encoder
        
        if encoder == 'baseline':
            # Original U-Net encoder
            self.enc1 = DoubleConv(in_ch, base_c)
            self.pool = nn.MaxPool2d(2)
            self.enc2 = DoubleConv(base_c, base_c*2)
            self.enc3 = DoubleConv(base_c*2, base_c*4)
            self.enc4 = DoubleConv(base_c*4, base_c*8)
            self.bottleneck = DoubleConv(base_c*8, base_c*16)
            
            # Decoder (U-Net style)
            self.up4 = nn.ConvTranspose2d(base_c*16, base_c*8, 2, stride=2)
            self.dec4 = DoubleConv(base_c*16, base_c*8)
            self.up3 = nn.ConvTranspose2d(base_c*8, base_c*4, 2, stride=2)
            self.dec3 = DoubleConv(base_c*8, base_c*4)
            self.up2 = nn.ConvTranspose2d(base_c*4, base_c*2, 2, stride=2)
            self.dec2 = DoubleConv(base_c*4, base_c*2)
            self.up1 = nn.ConvTranspose2d(base_c*2, base_c, 2, stride=2)
            self.dec1 = DoubleConv(base_c*2, base_c)
            self.final = nn.Conv2d(base_c, n_classes, 1)
            
        elif encoder == 'resnet50':
            # ResNet50 encoder (pretrained on ImageNet)
            resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
            
            # Extract encoder layers
            # ResNet has: conv1+bn+relu (stride=2) -> maxpool (stride=2) -> layer1-4
            # Total downsampling: 2 (conv1) * 2 (maxpool) * 2 (layer1) * 2 (layer2) * 2 (layer3) * 2 (layer4) = 64x
            self.initial_conv = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)  # 64 channels, /2
            self.initial_pool = resnet.maxpool  # /2
            self.enc1 = resnet.layer1  # 256 channels, same size
            self.enc2 = resnet.layer2  # 512 channels, /2
            self.enc3 = resnet.layer3  # 1024 channels, /2
            self.enc4 = resnet.layer4  # 2048 channels, /2
            
            # Decoder (U-Net style) - 5 upsampling stages to match 5 downsampling stages
            self.up4 = nn.ConvTranspose2d(2048, 1024, 2, stride=2)
            self.dec4 = DoubleConv(2048, 1024)
            self.up3 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
            self.dec3 = DoubleConv(1024, 512)
            self.up2 = nn.ConvTranspose2d(512, 256, 2, stride=2)
            self.dec2 = DoubleConv(512, 256)
            self.up1 = nn.ConvTranspose2d(256, 64, 2, stride=2)
            self.dec1 = DoubleConv(128, 64)
            # Extra upsampling to restore original resolution
            self.up0 = nn.ConvTranspose2d(64, 64, 2, stride=2)
            self.dec0 = DoubleConv(64, 64)
            self.final = nn.Conv2d(64, n_classes, 1)
            
        elif encoder == 'efficientnet_b1':
            # EfficientNet-B1 encoder (pretrained on ImageNet)
            efficientnet = models.efficientnet_b1(weights=models.EfficientNet_B1_Weights.IMAGENET1K_V2)
            features = efficientnet.features
            
            # Extract encoder layers - EfficientNet has 8 stages
            # Stage 0-1: /2, Stage 2: /2, Stage 3: /2, Stage 4-5: /2, Stage 6-7: /2
            # Total: 5 downsampling stages (256->128->64->32->16->8)
            self.initial_features = features[0:2]  # 16 channels, /2
            self.enc1 = features[2]  # 24 channels, /2
            self.enc2 = features[3]  # 40 channels, /2
            self.enc3 = nn.Sequential(features[4], features[5])  # 112 channels, /2
            self.enc4 = nn.Sequential(features[6], features[7])  # 320 channels, /2
            
            # Decoder (U-Net style) - 5 upsampling stages to match 5 downsampling stages
            self.up4 = nn.ConvTranspose2d(320, 112, 2, stride=2)
            self.dec4 = DoubleConv(224, 112)
            self.up3 = nn.ConvTranspose2d(112, 40, 2, stride=2)
            self.dec3 = DoubleConv(80, 40)
            self.up2 = nn.ConvTranspose2d(40, 24, 2, stride=2)
            self.dec2 = DoubleConv(48, 24)
            self.up1 = nn.ConvTranspose2d(24, 16, 2, stride=2)
            self.dec1 = DoubleConv(32, 16)
            # Extra upsampling to restore original resolution
            self.up0 = nn.ConvTranspose2d(16, 16, 2, stride=2)
            self.dec0 = DoubleConv(16, 16)
            self.final = nn.Conv2d(16, n_classes, 1)

    def forward(self, x):
        # Encoder
        if self.encoder_type == 'baseline':
            e1 = self.enc1(x)
            e2 = self.enc2(self.pool(e1))
            e3 = self.enc3(self.pool(e2))
            e4 = self.enc4(self.pool(e3))
            b = self.bottleneck(self.pool(e4))
            
            # Decoder with skip connections
            d4 = self.up4(b)
            d4 = self.dec4(torch.cat([d4, e4], dim=1))
            d3 = self.up3(d4)
            d3 = self.dec3(torch.cat([d3, e3], dim=1))
            d2 = self.up2(d3)
            d2 = self.dec2(torch.cat([d2, e2], dim=1))
            d1 = self.up1(d2)
            d1 = self.dec1(torch.cat([d1, e1], dim=1))
            out = self.final(d1)
            
        elif self.encoder_type == 'resnet50':
            # ResNet encoder with initial downsampling
            x0 = self.initial_conv(x)  # /2
            x0_pooled = self.initial_pool(x0)  # /4
            e1 = self.enc1(x0_pooled)  # 256ch, /4
            e2 = self.enc2(e1)  # 512ch, /8
            e3 = self.enc3(e2)  # 1024ch, /16
            e4 = self.enc4(e3)  # 2048ch, /32
            
            # Decoder with skip connections
            d4 = self.up4(e4)  # /16
            d4 = self.dec4(torch.cat([d4, e3], dim=1))
            d3 = self.up3(d4)  # /8
            d3 = self.dec3(torch.cat([d3, e2], dim=1))
            d2 = self.up2(d3)  # /4
            d2 = self.dec2(torch.cat([d2, e1], dim=1))
            d1 = self.up1(d2)  # /2
            d1 = self.dec1(torch.cat([d1, x0], dim=1))
            d0 = self.up0(d1)  # /1 (original size)
            d0 = self.dec0(d0)
            out = self.final(d0)
            
        else:  # efficientnet_b1
            # EfficientNet encoder with initial downsampling
            x0 = self.initial_features(x)  # 16ch, /2
            e1 = self.enc1(x0)  # 24ch, /4
            e2 = self.enc2(e1)  # 40ch, /8
            e3 = self.enc3(e2)  # 112ch, /16
            e4 = self.enc4(e3)  # 320ch, /32
            
            # Decoder with skip connections
            d4 = self.up4(e4)  # /16
            d4 = self.dec4(torch.cat([d4, e3], dim=1))
            d3 = self.up3(d4)  # /8
            d3 = self.dec3(torch.cat([d3, e2], dim=1))
            d2 = self.up2(d3)  # /4
            d2 = self.dec2(torch.cat([d2, e1], dim=1))
            d1 = self.up1(d2)  # /2
            d1 = self.dec1(torch.cat([d1, x0], dim=1))
            d0 = self.up0(d1)  # /1 (original size)
            d0 = self.dec0(d0)
            out = self.final(d0)
        
        return out
