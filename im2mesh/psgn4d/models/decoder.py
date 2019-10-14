import torch.nn as nn
import torch.nn.functional as F


class Decoder(nn.Module):
    r''' Simple decoder for the 4D Point Set Generation Network.

    The simple decoder consists of 4 fully-connected layers, resulting in an
    output of 3D coordinates for a fixed number of points and a fixed number of
    time steps.

    Args:
        dim (int): The output dimension of the points (e.g. 3)
        c_dim (int): dimension of the input vector
        n_points (int): number of output points
        n_steps (int): number of time steps
        hidden_dim (int): hidden dimension
    '''
    def __init__(self, dim=3, c_dim=128, n_points=1024, n_steps=17,
                 hidden_dim=512, **kwargs):
        super().__init__()
        # Attributes
        self.dim = dim
        self.c_dim = c_dim
        self.n_points = n_points
        self.n_steps = n_steps

        # Submodules
        self.actvn = F.relu
        self.fc_0 = nn.Linear(c_dim, hidden_dim)
        self.fc_1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, dim*n_points*n_steps)

    def forward(self, c):
        batch_size = c.size(0)

        net = self.fc_0(c)
        net = self.fc_1(self.actvn(net))
        net = self.fc_2(self.actvn(net))
        points = self.fc_out(self.actvn(net))
        points = points.view(
            batch_size, self.n_steps, self.n_points, self.dim)

        return points
