import torch
import torch.nn as nn
from torch import distributions as dist
from im2mesh.oflow.models import (
    encoder_latent, decoder, velocity_field)
from im2mesh.utils.torchdiffeq.torchdiffeq import odeint, odeint_adjoint

encoder_latent_dict = {
    'simple': encoder_latent.Encoder,
    'pointnet': encoder_latent.PointNet,
}

decoder_dict = {
    'simple': decoder.Decoder,
    'cbatchnorm': decoder.DecoderCBatchNorm,
}

velocity_field_dict = {
    'concat': velocity_field.VelocityField,
}


class OccupancyFlow(nn.Module):
    ''' Occupancy Flow model class.

    It consists of a decoder and, depending on the respective settings, an
    encoder, a temporal encoder, an latent encoder, a latent temporal encoder,
    and a vector field.

    Args:
        decoder (nn.Module): decoder network
        encoder (nn.Module): encoder network
        encoder_latent (nn.Module): latent encoder network
        encoder_latent_temporal (nn.Module): latent temporal encoder network
        encoder_temporal (nn.Module): temporal encoder network
        vector_field (nn.Module): vector field network
        ode_step_size (float): step size of ode solver
        use_adjoint (bool): whether to use the adjoint method for obtaining
            gradients
        rtol (float): relative tolerance for ode solver
        atol (float): absolute tolerance for ode solver
        ode_solver (str): ode solver method
        p0_z (dist): prior distribution
        device (device): PyTorch device
        input_type (str): type of input

    '''

    def __init__(
        self, decoder, encoder=None, encoder_latent=None,
            encoder_latent_temporal=None,
            encoder_temporal=None, vector_field=None,
            ode_step_size=None, use_adjoint=False,
            rtol=0.001, atol=0.00001, ode_solver='dopri5', p0_z=None,
            device=None, input_type=None, **kwargs):
        super().__init__()
        if p0_z is None:
            p0_z = dist.Normal(torch.tensor([]), torch.tensor([]))

        self.device = device
        self.input_type = input_type

        self.decoder = decoder
        self.encoder_latent = encoder_latent
        self.encoder_latent_temporal = encoder_latent_temporal
        self.encoder = encoder
        self.vector_field = vector_field
        self.encoder_temporal = encoder_temporal

        self.p0_z = p0_z
        self.rtol = rtol
        self.atol = atol
        self.ode_solver = ode_solver

        if use_adjoint:
            self.odeint = odeint_adjoint
        else:
            self.odeint = odeint

        self.ode_options = {}
        if ode_step_size:
            self.ode_options['step_size'] = ode_step_size

    def forward(self, p, time_val, inputs, sample=True):
        ''' Makes a forward pass through the network.

        Args:
            p (tensor): points tensor
            time_val (tensor): time values
            inputs (tensor): input tensor
            sample (bool): whether to sample
        '''
        batch_size = p.size(0)
        
        c_s, c_t = self.encode_inputs(inputs)
        z, z_t = self.get_z_from_prior((batch_size,), sample=sample)

        p_t_at_t0 = self.model.transform_to_t0(time_val, p, c_t=c_t, z=z_t)
        out = self.model.decode(p_t_at_t0, c=c_s, z=z)
        return out

    def decode(self, p, z=None, c=None, **kwargs):
        ''' Returns occupancy values for the points p at time step 0.

        Args:
            p (tensor): points
            z (tensor): latent code z
            c (tensor): latent conditioned code c (For OFlow, this is
                c_spatial)
        '''
        logits = self.decoder(p, z, c, **kwargs)
        p_r = dist.Bernoulli(logits=logits)
        return p_r

    def infer_z(self, inputs, c=None, data=None):
        ''' Infers a latent code z.

        The inputs and latent conditioned code are passed to the latent encoder
        to obtain the predicted mean and standard deviation.

        Args:
            inputs (tensor): input tensor
            c (tensor): latent conditioned code c
        '''
        if self.encoder_latent is not None:
            mean_z, logstd_z = self.encoder_latent(inputs, c, data=data)
        else:
            batch_size = inputs.size(0)
            mean_z = torch.empty(batch_size, 0).to(self.device)
            logstd_z = torch.empty(batch_size, 0).to(self.device)

        q_z = dist.Normal(mean_z, torch.exp(logstd_z))

        if self.encoder_latent_temporal is not None:
            mean_z, logstd_z = self.encoder_latent_temporal(inputs, c)

        q_z_t = dist.Normal(mean_z, torch.exp(logstd_z))

        return q_z, q_z_t

    def get_z_from_prior(self, size=torch.Size([]), sample=True):
        ''' Returns z from the prior distribution.

        If sample is true, z is sampled, otherwise the mean is returned.

        Args:
            size (torch.Size): size of z
            sample (bool): whether to sample z
        '''
        if sample:
            z_t = self.p0_z.sample(size).to(self.device)
            z = self.p0_z.sample(size).to(self.device)
        else:
            z = self.p0_z.mean.to(self.device)
            z = z.expand(*size, *z.size())
            z_t = z

        return z, z_t

    def to(self, device):
        ''' Puts the model to the device.

        Args:
            device (device): pytorch device
        '''
        model = super().to(device)
        model._device = device
        return model

    def eval_velocity_field(self, t, p, z=None, c_t=None):
        ''' Evaluates the velocity field at points p and time values t.

        Args:
            t (tensor): time values
            p (tensor): points tensor
            z (tensor): latent code z
            c_t (tensor): latent conditioned temporal code c_t
        '''
        z_dim = z.shape[-1]
        c_dim = c_t.shape[-1]

        p = self.concat_vf_input(p, c=c_t, z=z)
        t_steps_eval = torch.tensor(0).float().to(t.device)
        out = self.vector_field(t_steps_eval, p, T_batch=t).unsqueeze(0)
        p_out = self.disentangle_vf_output(
            out, c_dim=c_dim, z_dim=z_dim, return_start=True)
        p_out = p_out.squeeze(1)

        return out

    # ######################################################
    # #### Encoding related functions #### #

    def encode_temporal_inputs(self, inputs):
        ''' Returns the temporal encoding c_t.

        Args:
            inputs (tensor): input tensor)
        '''
        batch_size = inputs.shape[0]
        device = self.device
        '''
        if self.input_type == 'idx':
            c_t = self.encoder(inputs)
        '''
        if self.encoder_temporal is not None:
            c_t = self.encoder_temporal(inputs)
        else:
            c_t = torch.empty(batch_size, 0).to(device)

        return c_t

    def encode_spatial_inputs(self, inputs):
        ''' Returns the spatial encoding c_s

        Args:
            inputs (tensor): inputs tensor
        '''
        batch_size = inputs.shape[0]
        device = self.device

        # Reduce to only first time step
        if len(inputs.shape) > 1:
            inputs = inputs[:, 0, :]

        if self.encoder is not None:
            c = self.encoder(inputs)
        else:
            c = torch.empty(batch_size, 0).to(device)

        return c

    def encode_inputs(self, inputs):
        ''' Returns spatial and temporal latent code for inputs.

        Args:
            inputs (tensor): inputs tensor
        '''
        c_s = self.encode_spatial_inputs(inputs)
        c_t = self.encode_temporal_inputs(inputs)

        return c_s, c_t

    # ######################################################
    # #### Forward and Backward Flow functions #### #

    def transform_to_t_backward(self, t, p, z=None, c_t=None):
        ''' Transforms points p from time 1 (multiple) t backwards.

        For example, for t = [0.5, 1], it transforms the points from the
        coordinate system t = 1 to coordinate systems t = 0.5 and t = 0.

        Args:
            t (tensor): time values
            p (tensor): points tensor
            z (tensor): latent code z
            c_t (tensor): latent conditioned code c
        '''
        device = self.device
        batch_size = p.shape[0]

        p_out, _ = self.eval_ODE(t, p, c_t=c_t, z=z,
                                 t_batch=torch.ones(batch_size).to(device),
                                 invert=True, return_start=(0 in t))

        return p_out

    def transform_to_t(self, t, p, z=None, c_t=None):
        '''  Transforms points p from time 0 to (multiple) time values t.

        Args:
            t (tensor): time values
            p (tensor); points tensor
            z (tensor): latent code z
            c_t (tensor): latent conditioned temporal code c_t

        '''
        p_out, _ = self.eval_ODE(t, p, c_t=c_t, z=z, return_start=(0 in t))

        return p_out

    def transform_to_t0(self, t, p, z=None, c_t=None):
        ''' Transforms the points p at time t to time 0.

        Args:
            t (tensor): time values of the points
            p (tensor): points tensor
            z (tensor): latent code z
            c_t (tensor): latent conditioned temporal code c_t
        '''

        p_out, t_order = self.eval_ODE(t, p, c_t=c_t, z=z, t_batch=t,
                                       invert=True, return_start=True)

        # Select respective time value for each item from batch
        batch_size = len(t_order)
        p_out = p_out[torch.arange(batch_size), t_order]
        return p_out

    # ######################################################
    # #### ODE related functions and helper functions #### #

    def eval_ODE(self, t, p, c_t=None, z=None, t_batch=None, invert=False,
                 return_start=False):
        ''' Evaluates the ODE for points p and time values t.

        Args:
            t (tensor): time values
            p (tensor): points tensor
            c_t (tensor): latent conditioned temporal code
            z (tensor): latent code
            t_batch (tensor): helper time tensor for batch processing of points
                with different time values when going backwards
            invert (bool): whether to invert the velocity field (used for
                batch processing of points with different time values)
            return_start (bool): whether to return the start points
        '''
        c_dim = c_t.shape[-1]
        z_dim = z.shape[-1]

        t_steps_eval, t_order = self.return_time_steps(t)
        if len(t_steps_eval) == 1:
            return p.unsqueeze(1), t_order

        f_options = {'T_batch': t_batch, 'invert': invert}
        p = self.concat_vf_input(p, c=c_t, z=z)
        s = self.odeint(
            self.vector_field, p, t_steps_eval,
            method=self.ode_solver, rtol=self.rtol, atol=self.atol,
            options=self.ode_options, f_options=f_options)

        p_out = self.disentangle_vf_output(
            s, c_dim=c_dim, z_dim=z_dim, return_start=return_start)

        return p_out, t_order

    def return_time_steps(self, t):
        ''' Returns time steps for the ODE Solver.
        The time steps are ordered, duplicates are removed, and time 0
        is added for the start.

        Args:
            t (tensor): time values
        '''
        device = self.device
        t_steps_eval, t_order = torch.unique(
            torch.cat([torch.zeros(1).to(device), t]), sorted=True,
            return_inverse=True)
        return t_steps_eval, t_order[1:]

    def disentangle_vf_output(self, v_out, p_dim=3, c_dim=None,
                              z_dim=None, return_start=False):
        ''' Disentangles the output of the velocity field.

        The inputs and outputs for / of the velocity network are concatenated
        to be able to use the adjoint method.

        Args:
            v_out (tensor): output of the velocity field
            p_dim (int): points dimension
            c_dim (int): dimension of conditioned code c
            z_dim (int): dimension of latent code z
            return_start (bool): whether to return start points
        '''

        n_steps, batch_size, _ = v_out.shape

        if z_dim is not None and z_dim != 0:
            v_out = v_out[:, :, :-z_dim]

        if c_dim is not None and c_dim != 0:
            v_out = v_out[:, :, :-c_dim]

        v_out = v_out.contiguous().view(n_steps, batch_size, -1, p_dim)

        if not return_start:
            v_out = v_out[1:]

        v_out = v_out.transpose(0, 1)

        return v_out

    def concat_vf_input(self, p, c=None, z=None):
        ''' Concatenate points p and latent code c to use it as input for ODE Solver.

        p of size (B x T x dim) and c of size (B x c_dim) and z of size
        (B x z_dim) is concatenated to obtain a tensor of size
        (B x (T*dim) + c_dim + z_dim).

        This is done to be able to use to the adjont method for obtaining
        gradients.

        Args:
            p (tensor): points tensor
            c (tensor): latent conditioned code c
            c (tensor): latent code z
        '''
        batch_size = p.shape[0]
        p_out = p.contiguous().view(batch_size, -1)
        if c is not None and c.shape[-1] != 0:
            assert(c.shape[0] == batch_size)
            c = c.contiguous().view(batch_size, -1)
            p_out = torch.cat([p_out, c], dim=-1)

        if z is not None and z.shape[-1] != 0:
            assert(z.shape[0] == batch_size)
            z = z.contiguous().view(batch_size, -1)
            p_out = torch.cat([p_out, z], dim=-1)

        return p_out
