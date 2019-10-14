import torch
import os
import argparse
from im2mesh import config
from im2mesh.checkpoints import CheckpointIO


parser = argparse.ArgumentParser(
    description='Generate a motion or shape interpolation for models from the '
                'D-FAUST dataset.'
)
parser.add_argument('config', type=str, help='Path to config file.')
parser.add_argument('--n-steps', type=int, default=10,
                    help='Number of steps for interpolation.')


# Parse Arguments
args = parser.parse_args()
cfg = config.load_config(args.config, 'configs/default.yaml')
device = torch.device("cuda")
data_path = cfg['data']['path']
out_dir = cfg['training']['out_dir']
generation_dir = os.path.join(out_dir, cfg['generation']['generation_dir'])
interpolation_dir = os.path.join(out_dir, 'generation_latent_interpolation')
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
model_0 = {'model': '50002_jumping_jacks', 'start_idx': 110}
model_1 = {'model': '50002_light_hopping_loose', 'start_idx': 53}

# Model
model = config.get_model(cfg, device=device)

# Checkpoint
checkpoint_io = CheckpointIO(
    out_dir, initialize_from=cfg['model']['initialize_from'],
    initialization_file_name=cfg['model']['initialization_file_name'],
    model=model)
checkpoint_io.load(cfg['test']['model_file'])

# Generator
generator = config.get_generator(model, cfg, device=device)

# Generate
model.eval()
meshes, _ = generator.generate_latent_space_interpolation(
    model_0, model_1, latent_space_file_path, n_samples=args.n_steps)

# Save generated sequence
if not os.path.isdir(interpolation_dir):
    os.makedirs(interpolation_dir)
modelname = '%s_%d_to_%s_%d' % (model_0['model'],
                                model_0['start_idx'],
                                model_1['model'],
                                model_1['start_idx'])
generator.export(meshes, interpolation_dir, modelname)
