import torch.nn as nn
# import torch.nn.functional as F
from torchvision import models
from im2mesh.common import normalize_imagenet


class Resnet18(nn.Module):
    r''' ResNet-18 encoder network for image input.
    Args:
        c_dim (int): output dimension of the latent embedding
        normalize (bool): whether the input images should be normalized
        use_linear (bool): whether a final linear layer should be used
    '''

    def __init__(self, c_dim, normalize=True, use_linear=True):
        super().__init__()
        self.normalize = normalize
        self.use_linear = use_linear
        self.features = models.resnet18(pretrained=True)
        self.features.fc = nn.Sequential()
        if use_linear:
            self.fc = nn.Linear(512, c_dim)
        elif c_dim == 512:
            self.fc = nn.Sequential()
        else:
            raise ValueError('c_dim must be 512 if use_linear is False')

    def forward(self, x):
        if self.normalize:
            x = normalize_imagenet(x)
        net = self.features(x)
        out = self.fc(net)
        return out


class ConvEncoder3D(nn.Module):
    r''' Simple convolutional conditioning network.

    It consists of 6 convolutional layers, each downsampling the input by a
    factor of 2, and a final fully-connected layer projecting the output to
    c_dim dimensions.
    '''

    def __init__(self, c_dim=128, hidden_dim=32, **kwargs):
        r''' Initialisation.

        Args:
            c_dim (int): output dimension of the latent embedding
        '''
        super().__init__()
        self.conv0 = nn.Conv3d(3, hidden_dim, 3, stride=(1, 2, 2), padding=1)
        self.conv1 = nn.Conv3d(hidden_dim, hidden_dim*2,
                               3, stride=(2, 2, 2), padding=1)
        self.conv2 = nn.Conv3d(hidden_dim*2, hidden_dim*4,
                               3, stride=(1, 2, 2), padding=1)
        self.conv3 = nn.Conv3d(hidden_dim*4, hidden_dim*8,
                               3, stride=(2, 2, 2), padding=1)
        self.conv4 = nn.Conv3d(hidden_dim*8, hidden_dim*16,
                               3, stride=(2, 2, 2), padding=1)
        self.conv5 = nn.Conv3d(hidden_dim*16, hidden_dim*16,
                               3, stride=(2, 2, 2), padding=1)
        self.fc_out = nn.Linear(hidden_dim*16, c_dim)
        self.actvn = nn.ReLU()

    def forward(self, x):
        x = x.transpose(1, 2)
        batch_size = x.size(0)
        net = self.conv0(x)
        net = self.conv1(self.actvn(net))
        net = self.conv2(self.actvn(net))
        net = self.conv3(self.actvn(net))
        net = self.conv4(self.actvn(net))
        net = self.conv5(self.actvn(net))

        final_dim = net.shape[1]
        net = net.view(batch_size, final_dim, -1).mean(2)
        out = self.fc_out(self.actvn(net))

        return out
