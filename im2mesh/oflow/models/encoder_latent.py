import torch
import torch.nn as nn
import torch.nn.functional as F
from im2mesh.layers import ResnetBlockFC


def maxpool(x, dim=-1, keepdim=False):
    ''' Performs a maximum pooling operation.

    Args:
        x (tensor): input tensor
        dim (int): dimension of which the pooling operation is performed
        keepdim (bool): whether to keep the dimension
    '''
    out, _ = x.max(dim=dim, keepdim=keepdim)
    return out


class Encoder(nn.Module):
    ''' Latent encoder class.

    It encodes input points together with their occupancy values and an
    (optional) conditioned latent code c to mean and standard deviations of
    the posterior distribution.

    Args:
        dim (int): dimension of input points
        z_dim (int): dimension of latent code z
        c_dim (int): dimension of latent conditioned code c
        hidden_size (int): dimension of hidden size
        leaky (bool): whether to use leaky ReLUs as activation 

    '''

    def __init__(self, z_dim=128, c_dim=128, dim=3, hidden_dim=128,
                 leaky=False, **kwargs):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim

        # Submodules
        self.fc_pos = nn.Linear(dim, hidden_dim)

        if c_dim != 0:
            self.fc_c = nn.Linear(c_dim, hidden_dim)

        self.fc_0 = nn.Linear(1, hidden_dim)
        self.fc_1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_2 = nn.Linear(hidden_dim*2, hidden_dim)
        self.fc_3 = nn.Linear(hidden_dim*2, hidden_dim)
        self.fc_mean = nn.Linear(hidden_dim, z_dim)
        self.fc_logstd = nn.Linear(hidden_dim, z_dim)

        if not leaky:
            self.actvn = F.relu
            self.pool = maxpool
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)
            self.pool = torch.mean

    def forward(self, inputs, c=None, data=None, **kwargs):
        ''' Performs a forward pass through the network.

        Args:
            p (tensor): input points
            o (tensor): occupancy values
            c (tensor): latent code c
        '''
        device = inputs.device
        p = data['points'].to(device)
        o = data['points.occ'].to(device)

        if len(p.shape) > 3:  # Evaluation case
            p = p[:, 0]
            o = o[:, 0]

        batch_size, T, D = p.size()
        # output size: B x T X F
        net = self.fc_0(o.unsqueeze(-1))
        net = net + self.fc_pos(p)

        if self.c_dim != 0:
            net = net + self.fc_c(c).unsqueeze(1)

        net = self.fc_1(self.actvn(net))
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.fc_2(self.actvn(net))
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.fc_3(self.actvn(net))
        # Reduce to  B x F
        net = self.pool(net, dim=1)

        mean = self.fc_mean(net)
        logstd = self.fc_logstd(net)

        return mean, logstd


class PointNet(nn.Module):
    ''' Latent PointNet-based encoder class.

    It maps the inputs together with an (optional) conditioned code c
    to means and standard deviations.

    Args:
        dim (int): dimension of input points
        z_dim (int): dimension of latent code z
        c_dim (int): dimension of latent conditioned code c
        hidden_dim (int): dimension of hidden size
        n_blocks (int): number of ResNet-based blocks
    '''

    def __init__(self, z_dim=128, c_dim=128, dim=51, hidden_dim=128,
                 n_blocks=3, **kwargs):
        super().__init__()
        self.c_dim = c_dim
        self.dim = dim
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

    def forward(self, inputs, c=None, **kwargs):
        ''' Performs a forward pass through the network.

        Args:
            inputs (tensor): inputs
            c (tensor): latent conditioned code c
        '''
        batch_size, n_t, T, _ = inputs.shape

        # Reshape input is necessary
        if self.dim == 3:
            inputs = inputs[:, 0]
        else:
            inputs = inputs.transpose(
                1, 2).contiguous().view(batch_size, T, -1)
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

        net = self.pool(net, dim=1)

        mean = self.fc_mean(net)
        logstd = self.fc_logstd(net)

        return mean, logstd
