import torch
import torch.nn as nn
from models.layers import conv2D_layer_bn, deconv2D_layer_bn, Dense_layer_Sigmoid_bn


class Generator(nn.Module):
    def __init__(self, conf):
        super(Generator, self).__init__()
        self.conf = conf
        self.f = conf.filters
        self.latent_space = conf.latent_space
        self.age_dim = conf.age_dim
        # common use
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.flat = nn.Flatten()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        # specific use
        self.conv1_1 = conv2D_layer_bn(in_channels=1, out_channels=self.f, kernel_size=3, stride=1,
                                       padding='same')
        self.conv1_2 = conv2D_layer_bn(in_channels=self.f, out_channels=self.f, kernel_size=3, stride=1,
                                       padding='same')

        self.conv2_1 = conv2D_layer_bn(in_channels=self.f, out_channels=self.f * 2, kernel_size=3, stride=1,
                                       padding='same')
        self.conv2_2 = conv2D_layer_bn(in_channels=self.f * 2, out_channels=self.f * 2, kernel_size=3, stride=1,
                                       padding='same')

        self.conv3_1 = conv2D_layer_bn(in_channels=self.f * 2, out_channels=self.f * 4, kernel_size=3, stride=1,
                                       padding='same')
        self.conv3_2 = conv2D_layer_bn(in_channels=self.f * 4, out_channels=self.f * 4, kernel_size=3, stride=1,
                                       padding='same')

        self.conv4_1 = conv2D_layer_bn(in_channels=self.f * 4, out_channels=self.f * 8, kernel_size=3, stride=1,
                                       padding='same')
        self.conv4_2 = conv2D_layer_bn(in_channels=self.f * 8, out_channels=self.f * 8, kernel_size=3, stride=1,
                                       padding='same')

        self.mid1_1 = conv2D_layer_bn(in_channels=self.f * 8, out_channels=self.f, kernel_size=3, stride=1,
                                      padding='same')

        self.dens1_1 = Dense_layer_Sigmoid_bn(10 * 13 * self.f, self.latent_space)



        self.dens2_1 = nn.Linear(in_features=self.latent_space + self.age_dim,
                                  out_features=10 * 13 * self.f)  # relu 해줄 것.


        self.upconv4 = deconv2D_layer_bn(in_channels=32 + self.f * 8, out_channels=self.f * 8, kernel_size=4, stride=2,
                                         padding=1)
        self.conv_m_1_1 = conv2D_layer_bn(in_channels=self.f * 16, out_channels=self.f * 8, kernel_size=3, stride=1,
                                          padding='same')
        self.conv_m_1_2 = conv2D_layer_bn(in_channels=self.f * 8, out_channels=self.f * 8, kernel_size=3, stride=1,
                                          padding='same')
        self.upconv3 = deconv2D_layer_bn(in_channels=self.f * 8, out_channels=self.f * 4, kernel_size=4, stride=2,
                                         padding=1)
        self.conv5_1 = conv2D_layer_bn(in_channels=self.f * 8, out_channels=self.f * 4, kernel_size=3, stride=1,
                                       padding='same')
        self.conv5_2 = conv2D_layer_bn(in_channels=self.f * 4, out_channels=self.f * 4, kernel_size=3, stride=1,
                                       padding='same')
        self.upconv2 = deconv2D_layer_bn(in_channels=self.f * 4, out_channels=self.f * 2, kernel_size=4, stride=2,
                                         padding=1)
        self.conv6_1 = conv2D_layer_bn(in_channels=self.f * 4, out_channels=self.f * 2, kernel_size=3, stride=1,
                                       padding='same')
        self.conv6_2 = conv2D_layer_bn(in_channels=self.f * 2, out_channels=self.f * 2, kernel_size=3, stride=1,
                                       padding='same')
        self.upconv1 = deconv2D_layer_bn(in_channels=self.f * 2, out_channels=self.f, kernel_size=4, stride=2,
                                         padding=1)
        self.conv7_1 = conv2D_layer_bn(in_channels=self.f * 2, out_channels=self.f, kernel_size=3, stride=1,
                                       padding='same')
        self.conv7_2 = conv2D_layer_bn(in_channels=self.f, out_channels=1, kernel_size=3, stride=1,
                                       padding='same')

    def forward(self, x, age_vector):
        # (batch size, 1, 208, 160)
        x = self.conv1_1(x)
        conv1_2 = self.conv1_2(x)
        x = self.pool(conv1_2)
        # (batch size, filters, 104, 80)
        x = self.conv2_1(x)
        conv2_2= self.conv2_2(x)
        x = self.pool(conv2_2)
        # (batch size, filters*2, 52, 40)
        x = self.conv3_1(x)
        conv3_2 = self.conv3_2(x)
        x = self.pool(conv3_2)
        # (batch size, filters*4, 26, 20)
        x = self.conv4_1(x)
        conv4_2 = self.conv4_2(x)
        pool4 = self.pool(conv4_2)
        # (batch size, filters*8, 13, 10)

# ======================================================================================================================

        x = self.mid1_1(pool4)
        # (batch size, filters, 13, 10)
        x = self.flat(x)
        # (batch size, 4160)
        x = self.dens1_1(x)
        # (batch size, 130)
        x = torch.cat([x, age_vector], dim=1)
        # (batch size, 230)
        x  = self.dens2_1(x)
        x = self.relu(x)
        # (batch size, 4160)
        x = torch.reshape(x, [x.shape[0], 32, 13, 10])
        x = torch.cat([pool4, x], dim=1)
        # (batch size, 32 + filters*8, 13, 10)
# ======================================================================================================================
        x = self.upconv4(x)
        x = torch.cat([x, conv4_2], dim=1)
        x = self.conv_m_1_1(x)
        x = self.conv_m_1_2(x)
        # (batch size, filters*8, 26, 20)
        x = self.upconv3(x)
        # (batch size, filters*4, 52, 40)
        x = torch.cat([x, conv3_2], dim = 1)
        # (batch size, filters*8, 52, 40)

        x = self.conv5_1(x)
        x = self.conv5_2(x)
        # (batch size, filters*4, 52, 40)
        x = self.upconv2(x)
        # (batch size, filters*2, 104, 80)
        x = torch.cat([x, conv2_2], dim = 1)
        # (batch size, filters*4, 104, 80)
        x = self.conv6_1(x)
        x = self.conv6_2(x)
        # (batch size, filters*2, 104, 80)
        x = self.upconv1(x)
        # (batch size, filters, 208, 160)
        x = torch.cat([x, conv1_2], dim = 1)
        # (batch size, filters*2, 208, 160)
        x = self.conv7_1(x)
        x = self.conv7_2(x)
        # (batch size, 1, 208, 160)

        return x