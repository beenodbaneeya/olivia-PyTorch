import torch.nn as nn

# Standard convulation block followed by batch normalization
class cbrblock(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(cbrblock, self).__init__()
        self.cbr = nn.Sequential(nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=(1,1), padding='same', bias=False),nn.BatchNorm2d(output_channels), nn.ReLU())
    def forward(self, x):
        return self.cbr(x)

# Basic residual block
class conv_block(nn.Module):
    def __init__(self, input_channels, output_channels, scale_input):
        super(conv_block, self).__init__()
        self.scale_input = scale_input
        if self.scale_input:
            self.scale = nn.Conv2d(input_channels,output_channels, kernel_size=1, stride=(1,1), padding='same')
        self.layer1 = cbrblock(input_channels, output_channels)
        self.dropout = nn.Dropout(p=0.01)
        self.layer2 = cbrblock(output_channels, output_channels)
    def forward(self,x):
        residual = x
        out = self.layer1(x)
        out = self.dropout(out)
        out = self.layer2(out)
        if self.scale_input:
            residual = self.scale(residual)
        return out + residual

# WideResnet model
class WideResNet(nn.Module):
    def __init__(self, num_classes):
        super(WideResNet, self).__init__()
        # RGB images (3 channels) input for CIFAR-100 dataset
        nChannels = [3, 16, 160, 320, 640]
        # Grayscale images (1 channel) for Fashion MNIST dataset
        # nChannels = [1, 16, 160, 320, 640]
        self.input_block = cbrblock(nChannels[0], nChannels[1])
        self.block1 = conv_block(nChannels[1], nChannels[2], scale_input=True)
        self.block2 = conv_block(nChannels[2], nChannels[2], scale_input=False)
        self.pool1 = nn.MaxPool2d(2)
        self.block3 = conv_block(nChannels[2], nChannels[3], scale_input=True)
        self.block4 = conv_block(nChannels[3], nChannels[3], scale_input=False)
        self.pool2 = nn.MaxPool2d(2)
        self.block5 = conv_block(nChannels[3], nChannels[4], scale_input=True)
        self.block6 = conv_block(nChannels[4], nChannels[4], scale_input=False)
        # Global Average pooling
        self.pool = nn.AvgPool2d(7)
        # Fully connected layer
        self.flat = nn.Flatten()
        self.fc = nn.Linear(nChannels[4], num_classes)

    def forward(self, x):
        out = self.input_block(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.pool1(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.pool2(out)
        out = self.block5(out)
        out = self.block6(out)
        out = self.pool(out)
        out = self.flat(out)
        out = self.fc(out)
        return out