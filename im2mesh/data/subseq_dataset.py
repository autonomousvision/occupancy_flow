import os
import logging
from torch.utils import data
import yaml

logger = logging.getLogger(__name__)


class HumansDataset(data.Dataset):
    ''' 3D Shapes dataset class.
    '''

    def __init__(self, dataset_folder, fields, split=None,
                 categories=None, no_except=True, transform=None,
                 length_sequence=17, n_files_per_sequence=-1,
                 offset_sequence=0, ex_folder_name='pcl_seq', **kwargs):
        ''' Initialization of the the 3D shape dataset.

        Args:
            dataset_folder (str): dataset folder
            fields (dict): dictionary of fields
            split (str): which split is used
            categories (list): list of categories to use
            no_except (bool): no exception
            transform (callable): transformation applied to data points
        '''
        # Attributes
        self.dataset_folder = dataset_folder
        self.fields = fields
        self.no_except = no_except
        self.transform = transform
        self.length_sequence = length_sequence
        self.n_files_per_sequence = n_files_per_sequence
        self.offset_sequence = offset_sequence
        self.ex_folder_name = ex_folder_name

        # If categories is None, use all subfolders
        if categories is None:
            categories = os.listdir(dataset_folder)
            categories = [c for c in categories
                          if os.path.isdir(os.path.join(dataset_folder, c))]

        # Read metadata file
        metadata_file = os.path.join(dataset_folder, 'metadata.yaml')

        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                self.metadata = yaml.load(f)
        else:
            self.metadata = {
                c: {'id': c, 'name': 'n/a'} for c in categories
            }

        # Set index
        for c_idx, c in enumerate(categories):
            self.metadata[c]['idx'] = c_idx

        # Get all models
        self.models = []
        for c_idx, c in enumerate(categories):
            subpath = os.path.join(dataset_folder, c)
            if not os.path.isdir(subpath):
                logger.warning('Category %s does not exist in dataset.' % c)
            if split is not None and os.path.exists(
                os.path.join(subpath, split + '.lst')):
                split_file = os.path.join(subpath, split + '.lst')
                with open(split_file, 'r') as f:
                    models_c = f.read().split('\n')
            else:
                models_c = [f for f in os.listdir(subpath) if
                            os.path.isdir(os.path.join(subpath, f))]
            models_c = list(filter(lambda x: len(x) > 0, models_c))
            models_len = self.get_models_seq_len(subpath, models_c)
            models_c, start_idx = self.subdivide_into_sequences(
                models_c, models_len)
            self.models += [
                {'category': c, 'model': m, 'start_idx': start_idx[i]}
                for i, m in enumerate(models_c)
            ]

    def __len__(self):
        ''' Returns the length of the dataset.
        '''
        return len(self.models)

    def __getitem__(self, idx):
        ''' Returns an item of the dataset.

        Args:
            idx (int): ID of data point
        '''
        category = self.models[idx]['category']
        model = self.models[idx]['model']
        start_idx = self.models[idx]['start_idx']
        c_idx = self.metadata[category]['idx']

        model_path = os.path.join(self.dataset_folder, category, model)
        data = {}

        for field_name, field in self.fields.items():
            try:
                field_data = field.load(model_path, idx, c_idx, start_idx)
            except Exception:
                if self.no_except:
                    logger.warn(
                        'Error occured when loading field %s of model %s'
                        % (field_name, model)
                    )
                    return None
                else:
                    raise

            if isinstance(field_data, dict):
                for k, v in field_data.items():
                    if k is None:
                        data[field_name] = v
                    else:
                        data['%s.%s' % (field_name, k)] = v
            else:
                data[field_name] = field_data

        if self.transform is not None:
            data = self.transform(data)

        return data

    def get_model_dict(self, idx):
        return self.models[idx]

    def get_models_seq_len(self, subpath, models):
        ''' Returns the sequence length of a specific model.

        This is a little "hacky" as we assume the existence of the folder
        self.ex_folder_name. However, in our case this is always given.

        Args:
            subpath (str): subpath of model category
            models (list): list of model names
        '''
        ex_folder_name = self.ex_folder_name
        models_seq_len = [
            len(os.listdir(os.path.join(subpath, m, ex_folder_name)))
            for m in models]

        return models_seq_len

    def subdivide_into_sequences(self, models, models_len):
        ''' Subdivides model sequence into smaller sequences.

        Args:
            models (list): list of model names
            models_len (list): list of lengths of model sequences
        '''
        length_sequence = self.length_sequence
        n_files_per_sequence = self.n_files_per_sequence
        offset_sequence = self.offset_sequence

        # Remove files before offset
        models_len = [l - offset_sequence for l in models_len]

        # Reduce to maximum number of files that should be considered
        if n_files_per_sequence > 0:
            models_len = [min(n_files_per_sequence, l) for l in models_len]

        models_out = []
        start_idx = []
        for idx, model in enumerate(models):
            for n in range(0, models_len[idx] - length_sequence + 1):
                models_out.append(model)
                start_idx.append(n + offset_sequence)

        return models_out, start_idx

    def test_model_complete(self, category, model):
        ''' Tests if model is complete.

        Args:
            model (str): modelname
        '''
        model_path = os.path.join(self.dataset_folder, category, model)
        files = os.listdir(model_path)
        for field_name, field in self.fields.items():
            if not field.check_complete(files):
                logger.warn('Field "%s" is incomplete: %s'
                            % (field_name, model_path))
                return False

        return True
