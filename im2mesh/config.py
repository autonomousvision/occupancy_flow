import yaml
from torchvision import transforms
from im2mesh import data
from im2mesh import oflow, psgn4d, onet4d

method_dict = {
    # 4D Methods
    'onet4d': onet4d,
    'oflow': oflow,
    'psgn4d': psgn4d,
}


# General config
def load_config(path, default_path=None):
    ''' Loads config file.

    Args:  
        path (str): path to config file
        default_path (bool): whether to use default path
    '''
    # Load configuration from file itself
    with open(path, 'r') as f:
        cfg_special = yaml.load(f)

    # Check if we should inherit from a config
    inherit_from = cfg_special.get('inherit_from')

    # If yes, load this config first as default
    # If no, use the default_path
    if inherit_from is not None:
        cfg = load_config(inherit_from, default_path)
    elif default_path is not None:
        with open(default_path, 'r') as f:
            cfg = yaml.load(f)
    else:
        cfg = dict()

    # Include main configuration
    update_recursive(cfg, cfg_special)

    return cfg


def update_recursive(dict1, dict2):
    ''' Update two config dictionaries recursively.

    Args:
        dict1 (dict): first dictionary to be updated
        dict2 (dict): second dictionary which entries should be used

    '''
    for k, v in dict2.items():
        if k not in dict1:
            dict1[k] = dict()
        if isinstance(v, dict):
            update_recursive(dict1[k], v)
        else:
            dict1[k] = v


# Models
def get_model(cfg, device=None, dataset=None):
    ''' Returns the model instance.

    Args:
        cfg (dict): config dictionary
        device (device): pytorch device
        dataset (dataset): dataset
    '''
    method = cfg['method']
    model = method_dict[method].config.get_model(
        cfg, device=device, dataset=dataset)
    return model


# Trainer
def get_trainer(model, optimizer, cfg, device):
    ''' Returns a trainer instance.

    Args:
        model (nn.Module): the model which is used
        optimizer (optimizer): pytorch optimizer
        cfg (dict): config dictionary
        device (device): pytorch device
    '''
    method = cfg['method']
    trainer = method_dict[method].config.get_trainer(
        model, optimizer, cfg, device)
    return trainer


# Generator for final mesh extraction
def get_generator(model, cfg, device):
    ''' Returns a generator instance.

    Args:
        model (nn.Module): the model which is used
        cfg (dict): config dictionary
        device (device): pytorch device
    '''
    method = cfg['method']
    generator = method_dict[method].config.get_generator(model, cfg, device)
    return generator


# Datasets
def get_dataset(mode, cfg, return_idx=False, return_category=False):
    ''' Returns the dataset.

    Args:
        model (nn.Module): the model which is used
        cfg (dict): config dictionary
        return_idx (bool): whether to include an ID field
    '''
    method = cfg['method']
    dataset_type = cfg['data']['dataset']
    dataset_folder = cfg['data']['path']
    categories = cfg['data']['classes']

    # Get split
    splits = {
        'train': cfg['data']['train_split'],
        'val': cfg['data']['val_split'],
        'test': cfg['data']['test_split'],
    }
    split = splits[mode]
    # Create dataset
    if dataset_type == 'Humans':
        # Dataset fields
        # Method specific fields (usually correspond to output)
        fields = method_dict[method].config.get_data_fields(mode, cfg)
        # Input fields
        inputs_field = get_inputs_field(mode, cfg)
        if inputs_field is not None:
            fields['inputs'] = inputs_field

        if return_idx:
            fields['idx'] = data.IndexField()

        if return_category:
            fields['category'] = data.CategoryField()

        dataset = data.HumansDataset(
            dataset_folder, fields, split=split, categories=categories,
            length_sequence=cfg['data']['length_sequence'],
            n_files_per_sequence=cfg['data']['n_files_per_sequence'],
            offset_sequence=cfg['data']['offset_sequence'],
            ex_folder_name=cfg['data']['pointcloud_seq_folder'])
    else:
        raise ValueError('Invalid dataset "%s"' % cfg['data']['dataset'])

    return dataset


def get_inputs_field(mode, cfg):
    ''' Returns the inputs fields.

    Args:
        mode (str): the mode which is used
        cfg (dict): config dictionary
    '''
    input_type = cfg['data']['input_type']

    if input_type is None:
        inputs_field = None
    elif input_type == 'img_seq':
        if mode == 'train' and cfg['data']['img_augment']:
            resize_op = transforms.RandomResizedCrop(
                cfg['data']['img_size'], (0.75, 1.), (1., 1.))
        else:
            resize_op = transforms.Resize((cfg['data']['img_size']))

        transform = transforms.Compose([
            resize_op, transforms.ToTensor(),
        ])

        if mode == 'train':
            random_view = True
        else:
            random_view = False

        inputs_field = data.ImageSubseqField(
            cfg['data']['img_seq_folder'],
            transform, random_view=random_view)
    elif input_type == 'pcl_seq':
        connected_samples = cfg['data']['input_pointcloud_corresponding']
        transform = transforms.Compose([
            data.SubsamplePointcloudSeq(
                cfg['data']['input_pointcloud_n'],
                connected_samples=connected_samples),
            data.PointcloudNoise(cfg['data']['input_pointcloud_noise'])
        ])
        inputs_field = data.PointCloudSubseqField(
            cfg['data']['pointcloud_seq_folder'],
            transform, seq_len=cfg['data']['length_sequence'])
    elif input_type == 'end_pointclouds':
        transform = data.SubsamplePointcloudSeq(
            cfg['data']['input_pointcloud_n'],
            connected_samples=cfg['data']['input_pointcloud_corresponding'])

        inputs_field = data.PointCloudSubseqField(
            cfg['data']['pointcloud_seq_folder'],
            only_end_points=True, seq_len=cfg['data']['length_sequence'],
            transform=transform)
    elif input_type == 'idx':
        inputs_field = data.IndexField()
    else:
        raise ValueError(
            'Invalid input type (%s)' % input_type)
    return inputs_field
