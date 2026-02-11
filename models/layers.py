import torch
import torch.nn as nn

def conv2D_layer_bn(in_channels, out_channels, kernel_size = 3, stride = 1, padding = 'same'):
    l = nn.Sequential(
        (nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = kernel_size, stride = stride, padding = padding, )),
        (nn.BatchNorm2d(out_channels)),
        (nn.ReLU())
    )
    return l

def deconv2D_layer_bn(in_channels, out_channels, kernel_size = 4, stride = 1, padding = 'same'):
    l = nn.Sequential(
        (nn.ConvTranspose2d(in_channels = in_channels, out_channels = out_channels, kernel_size = kernel_size, stride = stride, padding = padding)),
        (nn.BatchNorm2d(out_channels)),
        (nn.ReLU())
    )
    return l


def Dense_layer_ReLU_bn(in_features, out_features):
    l = nn.Sequential(
        (nn.Linear(in_features=in_features, out_features=out_features)),
        (nn.BatchNorm1d(out_features)),
        (nn.ReLU())
    )
    return l

def Dense_layer_Sigmoid_bn(in_features, out_features):
    l = nn.Sequential(
        (nn.Linear(in_features=in_features, out_features=out_features)),
        (nn.BatchNorm1d(out_features)),
        (nn.Sigmoid())
    )
    return l

class GlobalAvgPool2d(nn.Module):
    """Global Average Pooling layer."""

    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, x):
        return torch.mean(x, dim=(2, 3))
