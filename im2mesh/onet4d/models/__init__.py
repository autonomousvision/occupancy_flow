import torch
import torch.nn as nn
from torch import distributions as dist
from im2mesh.onet4d.models import encoder_latent, decoder

encoder_latent_dict = {
    'simple': encoder_latent.Encoder,
    'pointnet': encoder_latent.PointNet,
}

decoder_dict = {
    'simple': decoder.Decoder,
    'cbatchnorm': decoder.DecoderCBatchNorm,
}


class OccupancyNetwork4D(nn.Module):
    ''' Occupancy Networks 4D class.

    Args:
        decoder (nn.Module): Decoder model
        encoder_latent (nn.Module): Latent encoder model
        encoder_temporal (nn.Module): Temporal encoder model
        p0_z (dist): Prior distribution over latent codes z
        device (device): Pytorch device
        input_type (str): Input type
    '''

    def __init__(
        self, decoder, encoder_latent=None,
            encoder_temporal=None, p0_z=None, device=None, input_type=None,
            **kwargs):
        super().__init__()
        if p0_z is None:
            p0_z = dist.Normal(torch.tensor([]), torch.tensor([]))

        self.device = device
        self.input_type = input_type

        self.decoder = decoder
        self.encoder_latent = encoder_latent
        self.encoder_temporal = encoder_temporal

        self.p0_z = p0_z

    def forward(self, p, inputs, sample=True, **kwargs):
        ''' Returns a forward pass through the model.

        Args:
            inputs (tensor): input tensor
            sample (bool): whether to sample from prior
        '''
        batch_size = p.size(0)
        c_t = self.encode_inputs(inputs)
        z = self.get_z_from_prior((batch_size,), sample=sample)
        p_r = self.decode(p, c=c_t, z=z)
        return p_r

    def decode(self, p, z=None, c=None, **kwargs):
        ''' Returns occupancy values for the points p at time step t.

        Args:
            p (tensor): points of dimension 4
            z (tensor): latent code z
            c (tensor): latent conditioned code c (For OFlow, this is
                c_spatial, whereas for ONet 4D, this is c_temporal)
        '''
        logits = self.decoder(p, z, c, **kwargs)
        p_r = dist.Bernoulli(logits=logits)
        return p_r

    def infer_z(self, inputs, c=None):
        ''' Infers a latent code z.

        The inputs and latent conditioned code are passed to the latent encoder
        to obtain the predicted mean and standard deviation.

        Args:
            inputs (tensor): input tensor
            c (tensor): latent conditioned code c
        '''
        if self.encoder_latent is not None:
            mean_z, logstd_z = self.encoder_latent(inputs, c)
        else:
            batch_size = inputs.size(0)
            mean_z = torch.empty(batch_size, 0).to(self.device)
            logstd_z = torch.empty(batch_size, 0).to(self.device)

        q_z = dist.Normal(mean_z, torch.exp(logstd_z))
        return q_z

    def get_z_from_prior(self, size=torch.Size([]), sample=True):
        ''' Returns z from the prior distribution.

        If sample is true, z is sampled, otherwise the mean is returned.

        Args:
            size (torch.Size): size of z
            sample (bool): whether to sample z
        '''
        if sample:
            z = self.p0_z.sample(size).to(self.device)
        else:
            z = self.p0_z.mean.to(self.device)
            z = z.expand(*size, *z.size())

        return z

    def to(self, device):
        ''' Puts the model to the device.

        Args:
            device (device): PyTorch device
        '''
        model = super().to(device)
        model._device = device
        return model

    def encode_inputs(self, inputs):
        ''' Returns encoded latent code for inputs.

        Args:
            inputs (tensor): inputs tensor
        '''
        batch_size = inputs.shape[0]
        device = self.device

        if self.encoder_temporal is not None:
            return self.encoder_temporal(inputs)
        else:
            return torch.empty(batch_size, 0).to(device)
