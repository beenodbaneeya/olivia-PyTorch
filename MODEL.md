# Overview
This script defines the architecture of a WideResNet model, which is a variant of the ResNet (Residual Network) architecture. WideResNet is designed to improve upon the original ResNet by increasing the width (number of channels) of the convolutional layers, which has been shown to improve performance on image classification tasks.

The script includes:

- cbrblock: A convolutional block with batch normalization and ReLU activation.

- conv_block: A residual block with skip connections and optional input scaling.

- WideResNet: The main model architecture, designed for tasks like CIFAR-100 classification.

## Key Components

1. `cbrblock`

Purpose:

 Defines a standard convolutional block that consists of:
- A 2D convolution layer.
- Batch normalization for stabilizing training.
- ReLU activation for introducing non-linearity.

Parameters:
- `input_channels (int)`: Number of input channels.
- `output_channels (int)`: Number of output channels.

Implementation:
- The convolution layer uses a kernel size of 3, stride of 1, and padding set to 'same' to preserve the spatial dimensions.
- Batch normalization is applied to normalize the outputs of the convolution layer.
- ReLU activation introduces non-linearity.

````python
class cbrblock(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(cbrblock, self).__init__()
        self.cbr = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=(1, 1), padding='same', bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU()
        )
    def forward(self, x):
        return self.cbr(x)
````

2. `conv_block`

Purpose:
- Defines a residual block, which is a key component of ResNet architectures. It includes:
- Two cbrblock layers.
- A skip connection to add the input (residual) to the output.
- Optional input scaling using a 1x1 convolution layer if the input and output channels differ.

Parameters:
- `input_channels` (int): Number of input channels.
- `output_channels` (int): Number of output channels.
- `scale_input` (bool): Whether to scale the input using a `1x1` convolution to match the output dimensions.

Implementation:
- The first cbrblock processes the input.
- A dropout layer is applied for regularization (dropout probability = `0.01`).
- The second cbrblock processes the intermediate output.
- If scale_input is True, a `1x1` convolution scales the input to match the output dimensions.
- The input (residual) is added to the output to form the final output.

````python
class conv_block(nn.Module):
    def __init__(self, input_channels, output_channels, scale_input):
        super(conv_block, self).__init__()
        self.scale_input = scale_input
        if self.scale_input:
            self.scale = nn.Conv2d(input_channels, output_channels, kernel_size=1, stride=(1, 1), padding='same')
        self.layer1 = cbrblock(input_channels, output_channels)
        self.dropout = nn.Dropout(p=0.01)
        self.layer2 = cbrblock(output_channels, output_channels)
    def forward(self, x):
        residual = x
        out = self.layer1(x)
        out = self.dropout(out)
        out = self.layer2(out)
        if self.scale_input:
            residual = self.scale(residual)
        return out + residual
````

3. WideResNet

Purpose:

Defines the full WideResNet model, which is composed of multiple convolutional and residual blocks. It is designed for image classification tasks, such as CIFAR-100.

Parameters:

num_classes (int): Number of output classes for classification (e.g., 100 for CIFAR-100).

Architecture:
- Input Block: A `cbrblock` processes the input image (e.g., RGB images with 3 channels for CIFAR-100).
- Residual Blocks: Multiple `conv_block` layers are used to extract features at different levels.Skip connections are used to preserve information and improve gradient flow.

Pooling Layers:

Max pooling layers downsample the spatial dimensions.

Global Average Pooling:

Averages the spatial dimensions to produce a fixed-size feature vector.

Fully Connected Layer:

A linear layer maps the feature vector to the desired number of output classes.

Implementation:

- The number of channels at each stage is defined in the nChannels list.
- The model processes the input through a series of blocks, pooling layers, and finally a fully connected layer.

````python
class WideResNet(nn.Module):
    def __init__(self, num_classes):
        super(WideResNet, self).__init__()
        # Define the number of channels at each stage
        nChannels = [3, 16, 160, 320, 640]
        # Input block
        self.input_block = cbrblock(nChannels[0], nChannels[1])
        # Residual blocks
        self.block1 = conv_block(nChannels[1], nChannels[2], scale_input=True)
        self.block2 = conv_block(nChannels[2], nChannels[2], scale_input=False)
        self.pool1 = nn.MaxPool2d(2)
        self.block3 = conv_block(nChannels[2], nChannels[3], scale_input=True)
        self.block4 = conv_block(nChannels[3], nChannels[3], scale_input=False)
        self.pool2 = nn.MaxPool2d(2)
        self.block5 = conv_block(nChannels[3], nChannels[4], scale_input=True)
        self.block6 = conv_block(nChannels[4], nChannels[4], scale_input=False)
        # Global average pooling
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
````

