import torch
from torch import nn, optim
import torch.nn.functional as F


#inspo https://github.com/julianstastny/VAE-ResNet18-PyTorch/blob/master/model.py

class ResizeConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, scale_factor, mode='nearest'):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        x = self.conv(x)
        return x

class BasicBlockEnc(nn.Module):

    def __init__(self, in_planes, stride=1):
        super().__init__()

        planes = in_planes*stride

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        if stride == 1:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        #print('init bb shape', x.shape)

        out = torch.relu(self.bn1(self.conv1(x)))
        #print('bb shape 1', x.shape)

        out = self.bn2(self.conv2(out))
        #print('bb shape 2', x.shape)

        out += self.shortcut(x)
        out = torch.relu(out)
        return out

class BasicBlockDec(nn.Module):

    def __init__(self, in_planes, stride=1):
        super().__init__()

        planes = int(in_planes/stride)

        self.conv2 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(in_planes)
        # self.bn1 could have been placed here, but that messes up the order of the layers when printing the class

        if stride == 1:
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(planes)
            self.shortcut = nn.Sequential()
        else:
            self.conv1 = ResizeConv2d(in_planes, planes, kernel_size=3, scale_factor=stride)
            self.bn1 = nn.BatchNorm2d(planes)
            self.shortcut = nn.Sequential(
                ResizeConv2d(in_planes, planes, kernel_size=3, scale_factor=stride),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        #print('init bb shape', x.shape)
        out = torch.relu(self.bn2(self.conv2(x)))
        #print('bb shape 0', x.shape)
        out = self.bn1(self.conv1(out))
        #print('bb shape 1', x.shape)
        out += self.shortcut(x)
        out = torch.relu(out)
        return out

class ResNet18Enc(nn.Module):

    def __init__(self, num_Blocks=[2,2,2,2], z_dim=10, nc=1, test=False):
        super().__init__()

        self.in_planes = 64
        self.z_dim = z_dim
        self.conv1 = nn.Conv2d(nc, 64, kernel_size=5, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(BasicBlockEnc, 64, num_Blocks[0], stride=1)
        self.layer2 = self._make_layer(BasicBlockEnc, 128, num_Blocks[1], stride=2)
        self.layer3 = self._make_layer(BasicBlockEnc, 256, num_Blocks[2], stride=2)
        self.layer4 = self._make_layer(BasicBlockEnc, 512, num_Blocks[3], stride=2)
        self.linear = nn.Linear(512, 2*z_dim)

        self.test = test
    def _make_layer(self, BasicBlockEnc, planes, num_Blocks, stride):
        strides = [stride] + [1]*(num_Blocks-1)
        layers = []
        for stride in strides:
            layers += [BasicBlockEnc(self.in_planes, stride)]
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        #print(x.shape)
        x = self.layer1(x)
        #print(x.shape)
        x = self.layer2(x)
        #print(x.shape)
        x = self.layer3(x)
        #print(x.shape)
        x = self.layer4(x)
        #print(x.shape)
        x = F.adaptive_avg_pool2d(x, 1)
        #print(x.shape)
        x = x.view(x.size(0), -1)
        #print(x.shape)
        x = self.linear(x)
        #print(x.shape)
        mu = x[:, :self.z_dim]
        logvar = x[:, self.z_dim:]
        if self.test:
        #    print('still test')
            return x
        return self.reparameterize(mu, logvar), mu, logvar

    @staticmethod
    def reparameterize(mean, logvar):
        std = torch.exp(logvar / 2) # in log-space, squareroot is divide by two
        epsilon = torch.randn_like(std)
        return epsilon * std + mean

class ResNet18Dec(nn.Module):

    def __init__(self, num_Blocks=[2,2,2,2], z_dim=10, nc=1):
        super().__init__()
        self.nc=nc
        self.in_planes = 512

        self.linear = nn.Linear(z_dim, 512)

        self.layer4 = self._make_layer(BasicBlockDec, 256, num_Blocks[3], stride=2)
        self.layer3 = self._make_layer(BasicBlockDec, 128, num_Blocks[2], stride=2)
        self.layer2 = self._make_layer(BasicBlockDec, 64, num_Blocks[1], stride=2)
        self.layer1 = self._make_layer(BasicBlockDec, 64, num_Blocks[0], stride=1)
        self.conv1 = ResizeConv2d(64, nc, kernel_size=3, scale_factor=4)

    def _make_layer(self, BasicBlockDec, planes, num_Blocks, stride):
        strides = [stride] + [1]*(num_Blocks-1)
        layers = []
        for stride in reversed(strides):
            layers += [BasicBlockDec(self.in_planes, stride)]
        self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, z):
        #print('decoding')
        x = self.linear(z)
        x = x.view(z.size(0), 512, 1, 1)
        #print('shape1', x.shape)
        x = F.interpolate(x, scale_factor=4)
        #print('shape2', x.shape)
        x = self.layer4(x)
        #print('shape3', x.shape)
        x = self.layer3(x)
        #print('shape4', x.shape)
        x = self.layer2(x)
        #print('shape5', x.shape)
        x = self.layer1(x)
        #print('shape6', x.shape)
        x = torch.sigmoid(self.conv1(x))
        #print('shape7', x.shape)

        x = x.view(x.size(0), self.nc, 128, 128)
        return x

class VAE_ResNet(nn.Module):

    def __init__(self, z_dim, test=False):
        super().__init__()
        #print('test', test)
        self.test=test
        self.z_dim = z_dim
        self.encoder = ResNet18Enc(z_dim=z_dim, test=test)
        self.decoder = ResNet18Dec(z_dim=z_dim)

    def forward(self, x):
        #print('encoding')
        #mean, logvar = self.encoder(x)
        if self.test:
            u = self.encoder(x)
            #print(self.z_dim)
            mean = u[:, :self.z_dim]
            logvar = u[:, self.z_dim:]
            #print('a', u.shape)
            #print('b', mean.shape)
            #print('c', logvar.shape )
            std = torch.exp(logvar / 2) # in log-space, squareroot is divide by two
            #print('d', std.shape)
            epsilon = torch.randn_like(std)
            #print('e', epsilon.shape)
            z = epsilon * std + mean
            #z = mean
        else:
            z, mu, logvar = self.encoder(x)
        #print('parametrising')
        #z = self.reparameterize(mean, logvar)
        #print(z.shape)
        x = self.decoder(z)
        return z, x
    
   # @staticmethod
   # def reparameterize(mean, logvar):
   #     std = torch.exp(logvar / 2) # in log-space, squareroot is divide by two
   #     epsilon = torch.randn_like(std)
   #     return epsilon * std + mean