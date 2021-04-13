import torch
import torch.nn as nn

class block(nn.Module):
    def __init__(self, in_channels, out_channels, identity_downsample=None, stride=1):
        super(block, self).__init__()
        self.expansion = 4
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn_1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn_2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, stride=1, padding=0)
        self.bn_3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.identity_downsample = identity_downsample
        self.stride = stride

    def forward(self, x):
        identity = x.copy()
        x = self.conv1(x)
        x = self.bn_1(x)
        x = nn.ReLU()(x)
        x = self.conv2(x)
        x = self.bn_2(x)
        x = nn.ReLU()(x)
        x = self.conv3(x)
        x = self.bn_3(x)

        if self.identity_downsample is not None:
            identity = self.identity_downsample(x)

        x += identity
        x = nn.ReLU()(x)
        return x

class ResNet(nn.Module):
    def __init__(self, block, layers, img_channels, num_classes):
        super(self, ResNet).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(img_channels, 64, 7, 2, padding=3) #
        self.bn_1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.layer_1 = self.layer(block, layers[0], out_channels=64, stride=1)
        self.layer_2 = self.layer(block, layers[1], out_channels=128, stride=2)
        self.layer_3 = self.layer(block, layers[2], out_channels=256, stride=2)
        self.layer_4 = self.layer(block, layers[3], out_channels=512, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(512 * 4, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn_1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.layer_4(x)
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.linear(x)
        return x

    def layer(self, block, num_res_block, out_channels, stride):
        identity_downsample = None
        layers = []

        if stride != 1 or self.in_channels != out_channels * 4:
            identity_downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    out_channels * 4,
                    kernel_size=1,
                    stride=stride
                ),
                nn.BatchNorm2d(out_channels * 4)
            )
        layers.append(block(self.in_channels, out_channels, identity_downsample, stride))

        self.in_channels = out_channels * 4

        for i in range(num_res_block - 1):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

def ResNet50(img_channel=3, num_class=1):
    return ResNet(block, [3, 4, 6, 3], img_channel, num_class)



