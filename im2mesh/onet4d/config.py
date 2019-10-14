import torch
import torch.distributions as dist
import os
from im2mesh.encoder import encoder_temporal_dict
from im2mesh.onet4d import models, training, generation
from im2mesh import data


def get_decoder(cfg, device, c_dim=0, z_dim=0):
    ''' Returns a decoder instance.

    Args:
        cfg (yaml): yaml config
        device (device): Pytorch device
        c_dim (int): dimension of latent conditioned code c
        z_dim (int): dimension of latent code z
    '''
    decoder = cfg['model']['decoder']
    decoder_kwargs = cfg['model']['decoder_kwargs']

    if decoder:
        decoder = models.decoder_dict[decoder](
            z_dim=z_dim, c_dim=c_dim,
            **decoder_kwargs).to(device)
    else:
        decoder = None

    return decoder


def get_encoder_latent(cfg, device, c_dim=0, z_dim=0):
    ''' Returns an encoder instance.

    Args:
        cfg (yaml): yaml config
        device (device): Pytorch device
        c_dim (int): dimension of latent conditioned code c
        z_dim (int): dimension of latent code z
    '''
    encoder_latent_kwargs = cfg['model']['encoder_latent_kwargs']
    encoder_latent = cfg['model']['encoder_latent']

    if encoder_latent:
        encoder_latent = models.encoder_latent_dict[encoder_latent](
            z_dim=z_dim, c_dim=c_dim,
            **encoder_latent_kwargs
        ).to(device)
    else:
        encoder_latent = None

    return encoder_latent


def get_encoder_temporal(cfg, device, c_dim=0, z_dim=0):
    ''' Returns a temporal encoder instance.

    Args:
        cfg (yaml): yaml config
        device (device): Pytorch device
        c_dim (int): dimension of latent conditioned code c
        z_dim (int): dimension of latent code z
    '''
    encoder_temporal = cfg['model']['encoder_temporal']
    encoder_temporal_kwargs = cfg['model']['encoder_temporal_kwargs']

    if encoder_temporal:
        encoder_temporal = encoder_temporal_dict[encoder_temporal](
            c_dim=c_dim, **encoder_temporal_kwargs).to(device)
    else:
        encoder_temporal = None

    return encoder_temporal


def get_model(cfg, device=None, dataset=None, **kwargs):
    ''' Returns a model instance.

    Args:
        cfg (yaml): yaml config
        device (device): Pytorch device
        dataset (dataset): Pytorch dataset
    '''
    # General arguments
    z_dim = cfg['model']['z_dim']
    c_dim = cfg['model']['c_dim']
    input_type = cfg['data']['input_type']

    decoder = get_decoder(cfg, device, c_dim, z_dim)
    encoder_latent = get_encoder_latent(cfg, device, c_dim, z_dim)
    encoder_temporal = get_encoder_temporal(cfg, device, c_dim, z_dim)

    p0_z = get_prior_z(cfg, device)
    model = models.OccupancyNetwork4D(
        decoder=decoder, encoder_latent=encoder_latent,
        encoder_temporal=encoder_temporal,
        p0_z=p0_z, device=device, input_type=input_type)

    return model


def get_trainer(model, optimizer, cfg, device, **kwargs):
    ''' Returns a trainer instance.

    Args:
        model (nn.Module): model instance
        optimzer (torch.optim): Pytorch optimizer
        cfg (yaml): yaml config
        device (device): Pytorch device
    '''
    threshold = cfg['test']['threshold']
    out_dir = cfg['training']['out_dir']
    vis_dir = os.path.join(out_dir, 'vis')
    input_type = cfg['data']['input_type']
    eval_sample = cfg['training']['eval_sample']

    trainer = training.Trainer(
        model, optimizer,
        device=device, input_type=input_type,
        vis_dir=vis_dir, threshold=threshold,
        eval_sample=eval_sample,
    )

    return trainer


def get_generator(model, cfg, device, **kwargs):
    ''' Returns a generator instance.

    Args:
        model (nn.Module): model instance
        cfg (yaml): yaml config
        device (device): Pytorch device
    '''
    generator = generation.Generator3D(
        model,
        device=device,
        threshold=cfg['test']['threshold'],
        resolution0=cfg['generation']['resolution_0'],
        upsampling_steps=cfg['generation']['upsampling_steps'],
        padding=cfg['generation']['padding'],
        sample=cfg['generation']['use_sampling'],
        refinement_step=cfg['generation']['refinement_step'],
        simplify_nfaces=cfg['generation']['simplify_nfaces'],
        n_time_steps=cfg['generation']['n_time_steps'],
        only_end_time_points=cfg['generation']['only_end_time_points'],
    )

    return generator


def get_prior_z(cfg, device, **kwargs):
    ''' Returns prior distribution.

    Args:
        cfg (yaml): yaml config
        device (device): Pytorch device
    '''
    z_dim = cfg['model']['z_dim']
    p0_z = dist.Normal(
        torch.zeros(z_dim, device=device),
        torch.ones(z_dim, device=device)
    )

    return p0_z


def get_transforms(cfg):
    ''' Returns transforms.

    Args:
        cfg (yaml): yaml config
    '''
    n_pt = cfg['data']['n_training_points']
    n_pt_eval = cfg['training']['n_eval_points']
    transf_pt = data.SubsamplePoints(n_pt)
    transf_pt_val = data.SubsamplePointsSeq(n_pt_eval, random=False)

    return transf_pt, transf_pt_val


def get_data_fields(mode, cfg):
    ''' Returns data fields.

    Args:
        mode (str): mode (train | val | test)
        cfg (yaml): yaml config
    '''
    fields = {}
    seq_len = cfg['data']['length_sequence']
    dataset = cfg['data']['dataset']
    p_folder = cfg['data']['points_iou_seq_folder']
    transf_pt, transf_pt_val = get_transforms(cfg)
    unpackbits = cfg['data']['points_unpackbits']

    if dataset == 'Humans':
        pts_iou_field = data.PointsSubseqField
    else:
        pts_iou_field = data.PointsSeqField

    if mode == 'train':
        fields['points_t'] = pts_iou_field(p_folder,
                                           transform=transf_pt,
                                           seq_len=seq_len,
                                           unpackbits=unpackbits)
    elif mode == 'val':
        fields['points_iou'] = pts_iou_field(p_folder, transform=transf_pt_val,
                                             all_steps=True, seq_len=seq_len,
                                             unpackbits=unpackbits)
    return fields
