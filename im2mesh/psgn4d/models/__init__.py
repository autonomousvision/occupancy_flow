import torch.nn as nn
from im2mesh.psgn4d.models.decoder import Decoder

decoder_dict = {
    'simple': Decoder,
}


class psgn4d(nn.Module):
    ''' The 4D Point Set Generation Network.

    For the PSGN, the input image is first passed to a conditioning network,
    e.g. restnet-18. Next, this latent code is then used as the input for the
    decoder network.

    Args:
        decoder (nn.Module): The decoder network
        encoder (nn.Module): The encoder network
    '''

    def __init__(self, decoder, encoder, input_type='pointcloud'):

        super().__init__()
        self.decoder = decoder
        self.encoder = encoder
        self.input_type = input_type

    def forward(self, x):
        c = self.encoder(x)
        points = self.decoder(c)
        return points
