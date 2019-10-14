import torch
import os
import argparse
from im2mesh import config
from im2mesh.checkpoints import CheckpointIO


parser = argparse.ArgumentParser(
    description='Generate a motion transfer for two sequences from the '
                'D-FAUST dataset.'
)
parser.add_argument('config', type=str, help='Path to config file.')

# Parse Arguments
args = parser.parse_args()
cfg = config.load_config(args.config, 'configs/default.yaml')
device = torch.device("cuda")
data_path = cfg['data']['path']
out_dir = cfg['training']['out_dir']
generation_dir = os.path.join(out_dir, cfg['generation']['generation_dir'])
motion_transfer_dir = os.path.join(out_dir, 'generation_motion_transfer')
latent_space_file_path = os.path.join(generation_dir, 'latent_space.pkl')

if cfg['method'] != 'oflow':
    print('This script is only available for Occupancy Flow.')
    exit(0)

# Check if latent space pickle file exists
if not os.path.exists(latent_space_file_path):
    raise FileNotFoundError(("Latent space encoding does not exists: Please "
                             "run encode_latent_motion_space.py before "
                             "generating a motion transfer."))

# Motion from motion_model is transferred to shape_model
motion_model = {'model': '50004_light_hopping_loose', 'start_idx': 60}
shape_model = {'model': '50009_light_hopping_loose', 'start_idx': 120}

print('Generating motion transfer for %s (%d) from %s (%d).' %
      (shape_model['model'], shape_model['start_idx'], motion_model['model'],
       motion_model['start_idx']))
# Path for shape of shape_model
shape_model_path = os.path.join(data_path, 'D-FAUST', shape_model['model'],
                                cfg['data']['mesh_seq_folder'])
if not os.path.exists(shape_model_path):
    print("Path to D-FAUST dataset does not exist.")
    exit(0)

# Model
model = config.get_model(cfg, device=device)

# Checkpoint
checkpoint_io = CheckpointIO(
    out_dir, initialize_from=cfg['model']['initialize_from'],
    initialization_file_name=cfg['model']['initialization_file_name'],
    model=model)
checkpoint_io.load(cfg['test']['model_file'])

# Generator
if cfg['generation']['mesh_color']:
    cfg['generation']['mesh_color'] = False
    print('Disabling mesh color for motion transfer.')
generator = config.get_generator(model, cfg, device=device)

# Generate
model.eval()
meshes = generator.generate_motion_transfer(motion_model, shape_model,
                                            shape_model_path,
                                            latent_space_file_path)

# Save generated sequence
if not os.path.isdir(motion_transfer_dir):
    os.makedirs(motion_transfer_dir)
modelname = '%s_%d_to_%s_%d' % (motion_model['model'],
                                motion_model['start_idx'],
                                shape_model['model'],
                                shape_model['start_idx'])
generator.export(meshes, motion_transfer_dir, modelname)
