import os
from im2mesh.encoder import encoder_temporal_dict
from im2mesh.psgn4d import models, training, generation
from im2mesh import data


def get_model(cfg, device=None, **kwargs):
    r''' Returns the model instance.

    Args:
        cfg (yaml object): the config file
        device (PyTorch device): the PyTorch device
    '''
    decoder = cfg['model']['decoder']
    encoder = cfg['model']['encoder']
    dim = cfg['data']['dim']
    c_dim = cfg['model']['c_dim']
    decoder_kwargs = cfg['model']['decoder_kwargs']
    encoder_kwargs = cfg['model']['encoder_kwargs']

    decoder = models.decoder_dict[decoder](
        dim=dim, c_dim=c_dim, **decoder_kwargs
    )

    encoder = encoder_temporal_dict[encoder](
        c_dim=c_dim, **encoder_kwargs
    )
    model = models.psgn4d(decoder, encoder,
                          input_type=cfg['data']['input_type'])
    model = model.to(device)
    return model


def get_trainer(model, optimizer, cfg, device, **kwargs):
    r''' Returns the trainer instance.

    Args:
        model (nn.Module): PSGN model
        optimizer (PyTorch optimizer): The optimizer that should be used
        cfg (yaml object): the config file
        device (PyTorch device): the PyTorch device
    '''
    input_type = cfg['data']['input_type']
    out_dir = cfg['training']['out_dir']
    loss_corr = cfg['model']['loss_corr']
    vis_dir = os.path.join(out_dir, 'vis')

    trainer = training.Trainer(
        model, optimizer, device=device, input_type=input_type,
        vis_dir=vis_dir, loss_corr=loss_corr,
    )
    return trainer


def get_generator(model, cfg, device, **kwargs):
    r''' Returns the generator instance.

    Args:
        cfg (yaml object): the config file
        device (PyTorch device): the PyTorch device
    '''
    generator = generation.Generator3D(model, device=device)
    return generator


def get_data_fields(mode, cfg):
    ''' Returns the data fields.

    Args:
        mode (str): mode (train|val|test)
        cfg (yaml file): yaml config file
    '''
    fields = {}

    # Define transform operation
    if mode == 'train':
        transform = data.SubsamplePointcloudSeq(
            cfg['data']['n_training_pcl_points'],
            connected_samples=True)
    else:
        transform = data.SubsamplePointcloudSeq(
                cfg['training']['n_eval_points'], random=False)

    # Define points field
    fields['points_mesh'] = data.PointCloudSubseqField(
            cfg['data']['pointcloud_seq_folder'],
            transform=transform,
    )

    return fields
