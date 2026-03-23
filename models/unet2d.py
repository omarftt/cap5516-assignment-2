import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(input_channels,output_channels,kernel_size=3,padding=1),
            nn.InstanceNorm2d(output_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(output_channels,output_channels,kernel_size=3,padding=1),
            nn.InstanceNorm2d(output_channels),
            nn.LeakyReLU(inplace=True),
        )
    def forward(self, x):
        return self.block(x)


class UNet2D(nn.Module):
    """
    2D U-Net for segmentation.
    This architecture has 4 encoder levels, a bottleneck, and 4 decoder levels.
    """

    def __init__(self, in_channels=4, out_channels=4, features=[32, 64, 128, 256]):
        super().__init__()

        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.upconvs = nn.ModuleList()

        # Encoder
        for feat in features:
            self.encoders.append(DoubleConv(in_channels, feat))
            self.pools.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_channels = feat

        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # Decoder path
        for f in reversed(features):
            self.upconvs.append(nn.ConvTranspose2d(f*2, f, kernel_size=2, stride=2))
            self.decoders.append(DoubleConv(f*2, f))

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skips = []

        for encoder, pool in zip(self.encoders, self.pools):
            x = encoder(x)
            skips.append(x)
            x = pool(x)

        x = self.bottleneck(x)
        skips = skips[::-1]

        for upconv, decoder, skip in zip(self.upconvs, self.decoders, skips):
            x = upconv(x)

            # Fix size mismatch if needed
            if x.shape[2:] != skip.shape[2:]:
                x = nn.functional.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=True)

            x = torch.cat([skip, x], dim=1)
            x = decoder(x)

        return self.final_conv(x)