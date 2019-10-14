import torch
import torch.nn as nn
import torch.nn.functional as F
from im2mesh.layers import ResnetBlockFC


def maxpool(x, dim=-1, keepdim=False):
    ''' Performs a max pooling operation.

    Args:
        x (tensor): input
        dim (int): dimension
        keepdim (bool): whether to keep dimenions
    '''
    out, _ = x.max(dim=dim, keepdim=keepdim)
    return out


class Encoder(nn.Module):
    ''' Encoder class for ONet 4D.

    Args:
        z_dim (int): dimension of latent code z
        c_dim (int): dimension of latent conditioned temporal code c
        dim (int): points dimension
        leaky (bool): whether to use leaky activation
    '''
    def __init__(self, z_dim=128, c_dim=128, dim=4, leaky=False):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim

        # Submodules
        self.fc_pos = nn.Linear(dim, 128)

        if c_dim != 0:
            self.fc_c = nn.Linear(c_dim, 128)

        if z_dim != 0:
            self.fc_z = nn.Linear(z_dim, 128)

        self.fc_0 = nn.Linear(1, 128)
        self.fc_1 = nn.Linear(128, 128)
        self.fc_2 = nn.Linear(256, 128)
        self.fc_3 = nn.Linear(256, 128)
        self.fc_mean = nn.Linear(128, z_dim)
        self.fc_logstd = nn.Linear(128, z_dim)

        if not leaky:
            self.actvn = F.relu
            self.pool = maxpool
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)
            self.pool = torch.mean

    def forward(self, p, x, c=None, z=None, **kwargs):
        ''' Performs a forward pass through the model.

        Args:
            p (tensor): points tensor
            x (tensor): occupancy values)
        '''
        batch_size, T, D = p.size()

        # output size: B x T X F
        net = self.fc_0(x.unsqueeze(-1))
        net = net + self.fc_pos(p)

        if self.c_dim != 0:
            net = net + self.fc_c(c).unsqueeze(1)

        if self.z_dim != 0:
            net = net + self.fc_z(z).unsqueeze(1)

        net = self.fc_1(self.actvn(net))
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.fc_2(self.actvn(net))
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.fc_3(self.actvn(net))
        # Recude to  B x F
        net = self.pool(net, dim=1)

        mean = self.fc_mean(net)
        logstd = self.fc_logstd(net)

        return mean, logstd


class PointNet(nn.Module):
    ''' Encoder PointNet class for ONet 4D.

    Args:
        z_dim (int): dimension of latent code z
        c_dim (int): dimension of latent conditioned temporal code c
        dim (int): points dimension
        hidden_dim (int): hidden dimension
        n_block (int): number of blocks to use
    '''
    def __init__(self, z_dim=128, c_dim=128, dim=51, hidden_dim=128,
                 n_blocks=3, **kwargs):
        super().__init__()
        self.c_dim = c_dim
        self.n_blocks = n_blocks

        self.fc_pos = nn.Linear(dim, 2*hidden_dim)

        self.blocks = nn.ModuleList(
            [ResnetBlockFC(2*hidden_dim, hidden_dim) for i in range(n_blocks)])

        if self.c_dim != 0:
            self.c_layers = nn.ModuleList(
                [nn.Linear(c_dim, 2*hidden_dim) for i in range(n_blocks)])

        self.actvn = nn.ReLU()
        self.pool = maxpool

        self.fc_mean = nn.Linear(hidden_dim, z_dim)
        self.fc_logstd = nn.Linear(hidden_dim, z_dim)

    def forward(self, inputs, c=None):
        batch_size, n_t, T, _ = inputs.shape

        inputs = inputs.transpose(1, 2).contiguous().view(batch_size, T, -1)
        # output size: B x T X F
        net = self.fc_pos(inputs)

        for i in range(self.n_blocks):
            if self.c_dim != 0:
                net_c = self.c_layers[i](c).unsqueeze(1)
                net = net + net_c

            net = self.blocks[i](net)        
            if i < self.n_blocks - 1:
                pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
                net = torch.cat([net, pooled], dim=2)

        # Recude to  B x F
        net = self.pool(net, dim=1)

        mean = self.fc_mean(net)
        logstd = self.fc_logstd(net)

        return mean, logstd
