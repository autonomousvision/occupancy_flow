import torch
import torch.nn as nn
import torch.nn.functional as F
from im2mesh.layers import (
    ResnetBlockFC, CResnetBlockConv1d,
    CBatchNorm1d, CBatchNorm1d_legacy,
)


class Decoder(nn.Module):
    ''' Decoder class for ONet 4D.

    Args:
        z_dim (int): dimension of latent code z
        c_dim (int): dimension of latent conditioned temporal code c
        dim (int): points dimension
        hidden_size (int): hidden dimension
        leaky (bool): whether to use leaky activation
    '''

    def __init__(self, dim=4, z_dim=128, c_dim=128,
                 hidden_size=128, leaky=False, **kwargs):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.dim = dim

        # Submodules
        self.fc_p = nn.Linear(dim, hidden_size)

        if not z_dim == 0:
            self.fc_z = nn.Linear(z_dim, hidden_size)
        if not c_dim == 0:
            self.fc_c = nn.Linear(c_dim, hidden_size)

        self.block0 = ResnetBlockFC(hidden_size)
        self.block1 = ResnetBlockFC(hidden_size)
        self.block2 = ResnetBlockFC(hidden_size)
        self.block3 = ResnetBlockFC(hidden_size)
        self.block4 = ResnetBlockFC(hidden_size)

        self.fc_out = nn.Linear(hidden_size, 1)

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

    def add_time_axis(self, p, t):
        ''' Adds time axis to points.

        Args:
            p (tensor): points
            t (tensor): time values
        '''
        n_pts = p.shape[1]
        t = t.unsqueeze(1).repeat(1, n_pts, 1)
        p_out = torch.cat([p, t], dim=-1)
        return p_out

    def forward(self, p, z=None, c=None, **kwargs):
        ''' Performs a forward pass through the model.

        Args:
            p (tensor): points tensor
            z (tensor): latent code z
            c (tensor): latent conditioned temporal code c
        '''
        if p.shape[-1] != self.dim:
            p = self.add_time_axis(p, kwargs['t'])

        batch_size = p.shape[0]
        p = p.view(batch_size, -1, self.dim)
        net = self.fc_p(p)

        if self.z_dim != 0:
            net_z = self.fc_z(z).unsqueeze(1)
            net = net + net_z

        if self.c_dim != 0:
            net_c = self.fc_c(c).unsqueeze(1)
            net = net + net_c

        net = self.block0(net)
        net = self.block1(net)
        net = self.block2(net)
        net = self.block3(net)
        net = self.block4(net)

        out = self.fc_out(self.actvn(net))
        out = out.squeeze(-1)

        return out


class DecoderCBatchNorm(nn.Module):
    ''' Decoder class with CBN for ONet 4D.

    Args:
        z_dim (int): dimension of latent code z
        c_dim (int): dimension of latent conditioned temporal code c
        dim (int): points dimension
        hidden_size (int): hidden dimension
        leaky (bool): whether to use leaky activation
        legacy (bool): whether to use legacy version
    '''

    def __init__(self, dim=4, z_dim=128, c_dim=128,
                 hidden_size=256, leaky=False, legacy=False):
        super().__init__()
        self.z_dim = z_dim
        self.dim = dim
        if not z_dim == 0:
            self.fc_z = nn.Linear(z_dim, hidden_size)

        self.fc_p = nn.Conv1d(dim, hidden_size, 1)
        self.block0 = CResnetBlockConv1d(c_dim, hidden_size, legacy=legacy)
        self.block1 = CResnetBlockConv1d(c_dim, hidden_size, legacy=legacy)
        self.block2 = CResnetBlockConv1d(c_dim, hidden_size, legacy=legacy)
        self.block3 = CResnetBlockConv1d(c_dim, hidden_size, legacy=legacy)
        self.block4 = CResnetBlockConv1d(c_dim, hidden_size, legacy=legacy)

        if not legacy:
            self.bn = CBatchNorm1d(c_dim, hidden_size)
        else:
            self.bn = CBatchNorm1d_legacy(c_dim, hidden_size)

        self.fc_out = nn.Conv1d(hidden_size, 1, 1)

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

    # For ONet 4D
    def add_time_axis(self, p, t):
        ''' Adds time axis to points.

        Args:
            p (tensor): points
            t (tensor): time values
        '''
        n_pts = p.shape[1]
        t = t.unsqueeze(1).repeat(1, n_pts, 1)
        p_out = torch.cat([p, t], dim=-1)
        return p_out

    def forward(self, p, z, c, **kwargs):
        ''' Performs a forward pass through the model.

        Args:
            p (tensor): points tensor
            z (tensor): latent code z
            c (tensor): latent conditioned temporal code c
        '''
        if p.shape[-1] != self.dim:
            p = self.add_time_axis(p, kwargs['t'])

        p = p.transpose(1, 2)
        batch_size, D, T = p.size()
        net = self.fc_p(p)

        if self.z_dim != 0:
            net_z = self.fc_z(z).unsqueeze(2)
            net = net + net_z

        net = self.block0(net, c)
        net = self.block1(net, c)
        net = self.block2(net, c)
        net = self.block3(net, c)
        net = self.block4(net, c)

        out = self.fc_out(self.actvn(self.bn(net, c)))
        out = out.squeeze(1)

        return out
