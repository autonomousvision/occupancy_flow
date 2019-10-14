import torch
import os
import shutil
import argparse
from tqdm import tqdm
import time
from collections import defaultdict
import pandas as pd
from im2mesh import config
from im2mesh.checkpoints import CheckpointIO
from im2mesh.utils.io import export_pointcloud


parser = argparse.ArgumentParser(
    description='Extract meshes from a 4D model.'
)
parser.add_argument('config', type=str, help='Path to config file.')
parser.add_argument('--no-cuda', action='store_true', help='Do not use cuda.')

args = parser.parse_args()
cfg = config.load_config(args.config, 'configs/default.yaml')
is_cuda = (torch.cuda.is_available() and not args.no_cuda)
device = torch.device("cuda" if is_cuda else "cpu")

out_dir = cfg['training']['out_dir']
generation_dir = os.path.join(out_dir, cfg['generation']['generation_dir'])
out_time_file = os.path.join(generation_dir, 'time_generation_full.pkl')
out_time_file_class = os.path.join(generation_dir, 'time_generation.pkl')

batch_size = cfg['generation']['batch_size']
input_type = cfg['data']['input_type']
vis_n_outputs = cfg['generation']['vis_n_outputs']
if vis_n_outputs is None:
    vis_n_outputs = -1

# Dataset
dataset = config.get_dataset('test', cfg, return_idx=True)

# Model
model = config.get_model(cfg, device=device, dataset=dataset)

checkpoint_io = CheckpointIO(out_dir, model=model)
checkpoint_io.load(cfg['test']['model_file'])

# Generator
generator = config.get_generator(model, cfg, device=device)

# Determine what to generate
generate_mesh = cfg['generation']['generate_mesh']
generate_pointcloud = cfg['generation']['generate_pointcloud']

# Loader
torch.manual_seed(cfg['generation']['rand_seed'])
test_loader = torch.utils.data.DataLoader(
    dataset, batch_size=1, num_workers=1,
    shuffle=cfg['generation']['shuffle_generation'])

# Statistics
time_dicts = []

# Generate
model.eval()

# Count how many models already created
model_counter = defaultdict(int)
for it, data in enumerate(tqdm(test_loader)):
    # Output folders
    mesh_dir = os.path.join(generation_dir, 'meshes')
    pointcloud_dir = os.path.join(generation_dir, 'pointcloud')
    in_dir = os.path.join(generation_dir, 'input')
    generation_vis_dir = os.path.join(generation_dir, 'vis', )

    # Get index etc.
    idx = data['idx'].item()
    try:
        model_dict = dataset.get_model_dict(idx)
    except AttributeError:
        model_dict = {'model': str(idx), 'category': 'n/a'}

    modelname = model_dict['model']
    category_id = model_dict['category']
    start_idx = model_dict.get('start_idx', 0)
    try:
        category_name = dataset.metadata[category_id].get('name', 'n/a')
    except AttributeError:
        category_name = 'n/a'

    if category_id != 'n/a':
        mesh_dir = os.path.join(mesh_dir, str(category_id))
        pointcloud_dir = os.path.join(pointcloud_dir, str(category_id))
        in_dir = os.path.join(in_dir, str(category_id))

        folder_name = str(category_id)
        if category_name != 'n/a':
            folder_name = str(folder_name) + '_' + category_name.split(',')[0]

        generation_vis_dir = os.path.join(generation_vis_dir, folder_name)

    # Create directories if necessary
    if vis_n_outputs >= 0 and not os.path.exists(generation_vis_dir):
        os.makedirs(generation_vis_dir)

    if generate_mesh and not os.path.exists(mesh_dir):
        os.makedirs(mesh_dir)

    if generate_pointcloud and not os.path.exists(pointcloud_dir):
        os.makedirs(pointcloud_dir)

    if not os.path.exists(in_dir) and cfg['generation']['copy_input']:
        os.makedirs(in_dir)

    # Timing dict
    time_dict = {
        'idx': idx,
        'class id': category_id,
        'class name': category_name,
        'modelname': modelname,
    }
    time_dicts.append(time_dict)

    # Generate outputs
    out_file_dict = {}

    if generate_mesh:
        t0 = time.time()
        out = generator.generate(data)
        time_dict['mesh'] = time.time() - t0
        # Get statistics
        try:
            mesh, stats_dict = out
        except TypeError:
            mesh, stats_dict = out, {}
        time_dict.update(stats_dict)

        # Save files
        out_files = generator.export(mesh, mesh_dir, modelname, start_idx)
        for nf, f in enumerate(out_files):
            out_file_dict['mesh_%03d' % nf] = f

    if generate_pointcloud:
        t0 = time.time()
        pointcloud = generator.generate(data)
        time_dict['pcl'] = time.time() - t0
        out_files = generator.export(
            pointcloud, pointcloud_dir, modelname, start_idx)

    # Copy to visualization directory for first vis_n_output samples
    c_it = model_counter[category_id]
    if c_it < vis_n_outputs:
        # Save inputs
        if cfg['generation']['copy_input']:
            if input_type == 'pcl_seq':
                inputs = data['inputs'][0]
                L = inputs.shape[0]
                inputs_base_path = os.path.join(
                    in_dir, modelname, '%04d' % start_idx)
                if not os.path.exists(inputs_base_path):
                    os.makedirs(inputs_base_path)
                inputs_path = [os.path.join(
                    inputs_base_path, '%04d.ply' % i) for i in range(L)]
                for i in range(L):
                    export_pointcloud(
                        inputs[i].cpu().numpy(), inputs_path[i], False)
                    out_file_dict['in_%d' % i] = inputs_path[i]

        # Save output files
        for k, filepath in out_file_dict.items():
            ext = os.path.splitext(filepath)[1]
            out_file = os.path.join(generation_vis_dir, '%02d_%s%s'
                                    % (c_it, k, ext))
            shutil.copyfile(filepath, out_file)

    model_counter[category_id] += 1

# Create pandas dataframe and save
time_df = pd.DataFrame(time_dicts)
time_df.set_index(['idx'], inplace=True)
time_df.to_pickle(out_time_file)

# Create pickle files  with main statistics
time_df_class = time_df.groupby(by=['class name']).mean()
time_df_class.to_pickle(out_time_file_class)

# Print results
time_df_class.loc['mean'] = time_df_class.mean()
print('Timings [s]:')
print(time_df_class)
