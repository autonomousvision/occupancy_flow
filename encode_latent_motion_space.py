from tqdm import tqdm
from im2mesh.checkpoints import CheckpointIO
from im2mesh import config, data
import time
import argparse
import os
import torch
from sklearn.manifold import TSNE
# from MulticoreTSNE import MulticoreTSNE as TSNE
import pandas as pd
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import cm as cm
matplotlib.use('Agg')

# Arguments
parser = argparse.ArgumentParser(
    description='Saves the latent space of motions of the D-FAUST dataset to a'
                'file. Optionally, a t-SNE embedding is created.'
)
parser.add_argument('config', type=str, help='Path to config file.')
parser.add_argument('--no-cuda', action='store_true', help='Do not use cuda.')
parser.add_argument('--tsne', action='store_true',
                    help='Whether a tsne embedding is generated.')
parser.add_argument('--tsne-max-n', type=int, default=-1,
                    help='Sets a max number of data points used for t-sne.')
parser.add_argument('--dpi', type=int,
                    help='Sets dpi of output figure.', default=200)

args = parser.parse_args()
cfg = config.load_config(args.config, 'configs/default.yaml')
is_cuda = (torch.cuda.is_available() and not args.no_cuda)
device = torch.device("cuda" if is_cuda else "cpu")
N = args.tsne_max_n

if cfg['method'] != 'oflow':
    print('This script is only available for Occupancy Flow.')
    exit(0)

# Set t0
t0 = time.time()

# Shorthands
out_dir = cfg['training']['out_dir']
generation_dir = cfg['generation']['generation_dir']
batch_size = cfg['training']['batch_size']

# Out file
generation_out_dir = os.path.join(out_dir, generation_dir)
if not os.path.exists(generation_out_dir):
    os.makedirs(generation_out_dir)
out_file_latent = os.path.join(generation_out_dir, 'latent_space.pkl')

if not os.path.isfile(out_file_latent):
    # Dataset
    train_dataset = config.get_dataset('train', cfg, return_idx=True)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, num_workers=4, shuffle=True,
        collate_fn=data.collate_remove_none,
        worker_init_fn=data.worker_init_fn)
    # Model
    model = config.get_model(cfg, device=device, dataset=train_dataset)

    kwargs = {'model': model, }
    checkpoint_io = CheckpointIO(
        out_dir, initialize_from=cfg['model']['initialize_from'],
        initialization_file_name=cfg['model']['initialization_file_name'],
        **kwargs)
    try:
        load_dict = checkpoint_io.load('model.pt')
    except FileExistsError:
        load_dict = dict()

    latent_dicts = []
    for batch in tqdm(train_loader):
        idx = batch['idx']
        # Encode inputs
        inputs = batch.get('inputs', torch.empty(1, 1, 0)).to(device)
        with torch.no_grad():
            c_s, c_t = model.encode_inputs(inputs)
            q_z, q_z_t = model.infer_z(inputs, c=c_t, data=batch)

        for i, id_i in enumerate(idx):
            model_dict = train_dataset.get_model_dict(id_i)
            model_dict['loc_z'] = q_z.loc[i].cpu().numpy()
            model_dict['scale_z'] = q_z.scale[i].cpu().numpy()
            model_dict['loc_z_t'] = q_z_t.loc[i].cpu().numpy()
            model_dict['scale_z_t'] = q_z_t.scale[i].cpu().numpy()
            model_dict['idx'] = id_i.item()
            latent_dicts.append(model_dict)

    # Save output file
    latent_df = pd.DataFrame(latent_dicts)
    latent_df.set_index(['idx'], inplace=True)
    latent_df.to_pickle(out_file_latent)
else:
    print('Latent space encoding already exists: %s.' % out_file_latent)
df = pd.read_pickle(out_file_latent)
df = df[['model', 'loc_z_t']]
df = df.values  # depending on the version, this is to_numpy()

if args.tsne:
    K = np.array([k[6:] for k in df[:, 0]])
    X = np.array([z for z in df[:, 1]])

    if N > 0 and N < len(X):
        idx = np.random.choice(len(X), size=(N,), replace=False)
        K = K[idx]
        X = X[idx]
    colors = cm.rainbow(np.linspace(0, 0.85, len(set(K))))

    K_list = ['punching', 'shake_arms', 'chicken_wings',  'jumping jacks',
              'shake_shoulders', 'jiggle_on_toes',  'light_hopping_stiff',
              'light_hopping_loose', 'shake_hips', 'hips', 'one_leg_jump',
              'one_leg_loose', 'knees', 'running_on_spot']

    tsne = TSNE(perplexity=50)
    X_proj = tsne.fit_transform(X)

    f = plt.figure(figsize=(8, 6))
    f.subplots_adjust(right=0.75)
    for i, c in enumerate(colors):
        mask = K == K_list[i]
        pts = X_proj[mask]
        label = K_list[i].replace('_', ' ')
        plt.scatter(pts[:, 0], pts[:, 1], color=c, label=label, s=1)
    plt.title('t-SNE Visualization of the Latent Space')
    plt.xticks([])
    plt.yticks([])

    out_fig_file = os.path.join(
        out_dir, generation_dir, 'tsne_visualization.png')
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left", prop={'size': 6})
    plt.savefig(out_fig_file, dpi=args.dpi)
    plt.close()
