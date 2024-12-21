import torch
import torch.nn as nn
import torch.nn.functional as F

class baseCNN(nn.Module):
    def __init__(self):
        super(baseCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=4, stride=1, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=1, padding=0)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=1, padding=0)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(128 * 19 * 5, 64)
        self.fc2 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))
        return x

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out

class ResNet34(nn.Module):
    def __init__(self):
        super(ResNet34, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self.make_layer(BasicBlock, 64, 3)
        self.layer2 = self.make_layer(BasicBlock, 128, 4, stride=2)
        self.layer3 = self.make_layer(BasicBlock, 256, 6, stride=2)
        self.layer4 = self.make_layer(BasicBlock, 512, 3, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * BasicBlock.expansion, 1)
        self.sigmoid = nn.Sigmoid()

    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = self.sigmoid(x)
        return x

class MBConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, expand_ratio, stride, kernel_size):
        super(MBConvBlock, self).__init__()
        self.stride = stride
        self.use_residual = self.stride == 1 and in_channels == out_channels
        hidden_dim = in_channels * expand_ratio
        self.expand = in_channels != hidden_dim
        layers = []
        if self.expand:
            layers.append(nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False))
            layers.append(nn.BatchNorm2d(hidden_dim))
            layers.append(nn.SiLU())
        layers.append(nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2, groups=hidden_dim, bias=False))
        layers.append(nn.BatchNorm2d(hidden_dim))
        layers.append(nn.SiLU())
        layers.append(nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_residual:
            return x + self.block(x)
        else:
            return self.block(x)

class EfficientNetB3(nn.Module):
    def __init__(self):
        super(EfficientNetB3, self).__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 40, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(40),
            nn.SiLU()
        )
        self.blocks = nn.Sequential(
            MBConvBlock(40, 24, expand_ratio=1, stride=1, kernel_size=3),
            MBConvBlock(24, 24, expand_ratio=1, stride=1, kernel_size=3),
            MBConvBlock(24, 48, expand_ratio=6, stride=2, kernel_size=3),
            MBConvBlock(48, 48, expand_ratio=6, stride=1, kernel_size=3),
            MBConvBlock(48, 80, expand_ratio=6, stride=2, kernel_size=5),
            MBConvBlock(80, 80, expand_ratio=6, stride=1, kernel_size=5),
            MBConvBlock(80, 112, expand_ratio=6, stride=1, kernel_size=5),
            MBConvBlock(112, 112, expand_ratio=6, stride=1, kernel_size=5),
            MBConvBlock(112, 192, expand_ratio=6, stride=2, kernel_size=3),
            MBConvBlock(192, 192, expand_ratio=6, stride=1, kernel_size=3),
            MBConvBlock(192, 192, expand_ratio=6, stride=1, kernel_size=3),
            MBConvBlock(192, 320, expand_ratio=6, stride=1, kernel_size=5)
        )
        self.head = nn.Sequential(
            nn.Conv2d(320, 1280, kernel_size=1, bias=False),
            nn.BatchNorm2d(1280),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(1280, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        return x


if __name__ == "__main__":
    model_0 = baseCNN()
    model_1 = ResNet34()
    model_2 = EfficientNetB3()
    # 180x64 입력으로 테스트
    sample_input = torch.randn(1, 1, 180, 64)
    output_0 = model_0(sample_input)
    output_1 = model_1(sample_input)
    output_2 = model_2(sample_input)
    print(output_0, output_1, output_2) 