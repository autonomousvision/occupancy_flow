import torch
import os
import argparse
from im2mesh import config
from im2mesh.checkpoints import CheckpointIO
import trimesh
import numpy as np
import glob
from im2mesh.common import load_and_scale_mesh


parser = argparse.ArgumentParser(
    description='Generate interpolation between meshes.'
)
parser.add_argument('config', type=str, help='Path to config file.')
parser.add_argument('--folder', type=str, help='Path to folder of meshes.',
                    required=True)
parser.add_argument('--n-steps', type=int, default=30,
                    help='Number of steps between two adjacent meshes.')
parser.add_argument('--extension', default='obj',
                    help='Data format of meshes (Standard obj)')
parser.add_argument('--seq-name', type=str, default='model0',
                    help='Name of sequence')
parser.add_argument('--num-points', type=int, default=10000,
                    help='Number of points which are sampled from the meshes.')

# Parse Arguments
args = parser.parse_args()
cfg = config.load_config(args.config, 'configs/default.yaml')
device = torch.device("cuda")
out_dir = cfg['training']['out_dir']
folder_dir = args.folder
generation_dir = os.path.join(folder_dir, 'generated_interpolation')

if cfg['method'] != 'oflow':
    print('This script is only available for Occupancy Flow.')
    exit(0)

mesh_files = glob.glob(os.path.join(folder_dir, '*.%s' % args.extension))
if len(mesh_files) == 0:
    raise FileNotFoundError("Folder path or does not contain any meshes.")
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
# Generator
if cfg['generation']['mesh_color']:
    cfg['generation']['mesh_color'] = False
    print('Disabling mesh color for interpolation generation.')
generator = config.get_generator(model, cfg, device=device)

# Generate
model.eval()

if not os.path.exists(generation_dir):
    os.makedirs(generation_dir)

# Process folder path
mesh_files.sort()
modelname = args.seq_name

loc, scale, mesh = load_and_scale_mesh(mesh_files[0])
f = trimesh.load(mesh_files[0], process=False).faces

# Sample random points
n_p = args.num_points
_, face_idx = mesh.sample(n_p, return_index=True)
alpha = np.random.dirichlet((1,)*3, n_p)
points = []
vertices = []
for mesh_path in mesh_files:
    _, _, mesh = load_and_scale_mesh(mesh_path, loc, scale)
    p = (alpha[:, :, None] * mesh.vertices[mesh.faces[face_idx]]).sum(axis=1)
    points.append(torch.tensor(p).float())
    vertices.append(torch.tensor(mesh.vertices).float())

# Prepare data dictionary
data = {
    'mesh.vertices': torch.stack(vertices).to(device),
    'inputs': torch.stack(points).to(device),
    'mesh.faces': torch.from_numpy(f).unsqueeze(0).to(device)
}

# Generate and save meshes
meshes = generator.interpolate_sequence(data, n_time_steps=args.n_steps)
generator.export(meshes, generation_dir, modelname)
