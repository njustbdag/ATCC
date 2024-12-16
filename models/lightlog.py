import torch.nn as nn
import torch.nn.functional as F


def calculate_padding(kernel_size, dilation_rate):
    return (kernel_size - 1) * dilation_rate // 2


class ResBlock(nn.Module):
    def __init__(self, in_channels, filters, kernel_size, dilation_rate):
        super(ResBlock, self).__init__()
        self.padding = calculate_padding(kernel_size, dilation_rate)
        self.conv1 = nn.Conv1d(in_channels, filters, kernel_size, padding=self.padding, dilation=dilation_rate)
        self.conv2 = nn.Conv1d(filters, 1, 3, padding=self.padding, dilation=dilation_rate)
        self.shortcut_conv = nn.Conv1d(in_channels, filters, 1) if in_channels != filters else None

    def forward(self, x):
        r = F.relu(self.conv1(x))  # First Conv1D layer with ReLU
        r = self.conv2(r)  # Second Conv1D layer
        if self.shortcut_conv:
            shortcut = self.shortcut_conv(x)  # Adjust input size if necessary
        else:
            shortcut = x  # No adjustment needed if the channel sizes match
        out = r + shortcut  # Residual connection
        out = F.relu(out)  # Activation after addition
        return out


class TCN(nn.Module):
    def __init__(self, input_channels=20, filters=3, kernel_size=3, dilation_rates=[1, 2, 4, 8], output_classes=2):
        super(TCN, self).__init__()

        self.resblock1 = ResBlock(input_channels, filters, kernel_size, dilation_rates[0])
        self.resblock2 = ResBlock(filters, filters, kernel_size, dilation_rates[1])
        self.resblock3 = ResBlock(filters, filters, kernel_size, dilation_rates[2])
        self.resblock4 = ResBlock(filters, filters, kernel_size, dilation_rates[3])

        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)  # Global average pooling
        self.fc = nn.Linear(filters, output_classes)  # Fully connected layer for classification

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.resblock3(x)
        x = self.resblock4(x)

        x = self.global_avg_pool(x)  # Apply global average pooling
        x = x.view(x.size(0), -1)  # Flatten
        output = self.fc(x)
        return output
