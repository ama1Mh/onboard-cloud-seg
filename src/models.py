"""Models trained in this project.

TinyUNet(base=16) has 1.94M parameters with 4 input bands, the same size as the
unpruned U-Net of Cratere et al. (2025). base=8 -> 0.49M, base=4 -> 0.12M.
"""
import torch
import torch.nn as nn


def double_conv(c_in, c_out):
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
    )


class TinyUNet(nn.Module):
    """Plain U-Net with 4 down/up levels. Channels: base, 2b, 4b, 8b, 16b (bottleneck)."""

    def __init__(self, in_ch=4, n_classes=4, base=16):
        super().__init__()
        c = [base * 2 ** k for k in range(5)]
        self.enc = nn.ModuleList([double_conv(in_ch, c[0])] +
                                 [double_conv(c[k], c[k + 1]) for k in range(3)])
        self.bottleneck = double_conv(c[3], c[4])
        self.pool = nn.MaxPool2d(2)
        self.up = nn.ModuleList([nn.ConvTranspose2d(c[k + 1], c[k], 2, stride=2) for k in range(4)])
        self.dec = nn.ModuleList([double_conv(2 * c[k], c[k]) for k in range(4)])
        self.head = nn.Conv2d(c[0], n_classes, 1)

    def forward(self, x):
        skips = []
        for enc in self.enc:
            x = enc(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        for k in reversed(range(4)):
            x = self.up[k](x)
            x = self.dec[k](torch.cat([x, skips[k]], dim=1))
        return self.head(x)


def build_model(name, in_ch, n_classes=4):
    """Names: tinyunet4, tinyunet8, tinyunet16, tinyunet32, unet_mobilenetv2."""
    if name.startswith("tinyunet"):
        return TinyUNet(in_ch, n_classes, base=int(name.removeprefix("tinyunet")))
    if name == "unet_mobilenetv2":   # same architecture as UNetMobV2, trained by us
        import segmentation_models_pytorch as smp
        return smp.Unet(encoder_name="mobilenet_v2", encoder_weights=None,
                        in_channels=in_ch, classes=n_classes)
    raise ValueError(f"Unknown model {name}")
