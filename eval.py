import argparse
# import numpy as np
import os
from tqdm import tqdm
import pandas as pd
import torch
from im2mesh import config, data
from im2mesh.eval import MeshEvaluator
from im2mesh.utils.io import load_pointcloud, load_mesh
import glob

parser = argparse.ArgumentParser(
    description='Evaluate mesh algorithms.'
)
parser.add_argument('config', type=str, help='Path to config file.')
parser.add_argument('--no-cuda', action='store_true', help='Do not use cuda.')
parser.add_argument('--eval_input', action='store_true',
                    help='Evaluate inputs instead.')

args = parser.parse_args()
cfg = config.load_config(args.config, 'configs/default.yaml')
is_cuda = (torch.cuda.is_available() and not args.no_cuda)
device = torch.device("cuda" if is_cuda else "cpu")
dataset_folder = cfg['data']['path']

# Shorthands
out_dir = cfg['training']['out_dir']
generation_dir = os.path.join(out_dir, cfg['generation']['generation_dir'])
project_to_final_mesh = cfg['test']['project_to_final_mesh']
if not args.eval_input:
    out_file = os.path.join(generation_dir, 'eval_meshes_full.pkl')
    out_file_tmp = os.path.join(generation_dir, 'eval_meshes_full_tmp.pkl')
    out_file_class = os.path.join(generation_dir, 'eval_meshes.csv')
    out_file_class_tmp = os.path.join(generation_dir, 'eval_meshes_tmp.csv')
else:
    out_file = os.path.join(generation_dir, 'eval_input_full.pkl')
    out_file_class = os.path.join(generation_dir, 'eval_input.csv')

# Dataset
fields = {
    'pointcloud_chamfer': data.PointCloudSubseqField(
        cfg['data']['pointcloud_seq_folder'],
        seq_len=cfg['data']['length_sequence']),
    'idx': data.IndexField(),
}
if cfg['test']['eval_mesh_iou']:
    fields['points'] = data.PointsSubseqField(
        cfg['data']['points_iou_seq_folder'], all_steps=True,
        seq_len=cfg['data']['length_sequence'],
        unpackbits=cfg['data']['points_unpackbits'])


print('Test split: ', cfg['data']['test_split'])


dataset = data.HumansDataset(
    dataset_folder, fields, split=cfg['data']['test_split'],
    categories=cfg['data']['classes'],
    length_sequence=cfg['data']['length_sequence'],
    n_files_per_sequence=cfg['data']['n_files_per_sequence'],
    offset_sequence=cfg['data']['offset_sequence'])

# Evaluator
evaluator = MeshEvaluator(n_points=100000)

# Loader
test_loader = torch.utils.data.DataLoader(
    dataset, batch_size=1, num_workers=1, shuffle=True)

# Evaluate all classes
eval_dicts = []
print('Starting evaluation process ...')
for it, data_batch in enumerate(tqdm(test_loader)):
    if data_batch is None:
        print('Invalid data.')
        continue

    # Output folders
    if not args.eval_input:
        mesh_dir = os.path.join(generation_dir, 'meshes')
        pointcloud_dir = os.path.join(generation_dir, 'pointcloud')
    else:
        mesh_dir = os.path.join(generation_dir, 'input')
        pointcloud_dir = os.path.join(generation_dir, 'input')

    # Get index etc.
    idx = data_batch['idx'].item()

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
        mesh_dir = os.path.join(mesh_dir, category_id)
        pointcloud_dir = os.path.join(pointcloud_dir, category_id)

    # Evaluate
    pointcloud_tgt = data_batch['pointcloud_chamfer'].squeeze(0).cpu().numpy()
    if cfg['test']['eval_mesh_iou']:
        points_tgt = data_batch['points'].squeeze(0).cpu().numpy()
        occ_tgt = data_batch['points.occ'].squeeze(0).cpu().numpy()

    # Evaluating mesh and pointcloud
    # Start row and put basic information inside
    eval_dict = {
        'idx': idx,
        'class id': category_id,
        'class name': category_name,
        'modelname': modelname,
        'start_idx': start_idx,
    }
    eval_dicts.append(eval_dict)
    # Evaluate mesh

    if cfg['test']['eval_mesh']:
        mesh_folder = os.path.join(mesh_dir, modelname, '%05d' % start_idx)
        if os.path.exists(mesh_folder):
            off_files = glob.glob(os.path.join(mesh_folder, '*.off'))
            off_files.sort()

            if cfg['test']['eval_only_end_time_steps']:
                off_files = [off_files[0], off_files[-1]]
                pointcloud_tgt = pointcloud_tgt[[0, -1]]

            if cfg['test']['eval_mesh_correspondences']:
                eval_dict_cor = evaluator.eval_correspondences_mesh(
                    off_files, pointcloud_tgt,
                    project_to_final_mesh=project_to_final_mesh)
                for k, v in eval_dict_cor.items():
                    eval_dict[k] = v
            if cfg['test']['eval_mesh_iou']:
                for i, mesh_file in enumerate(off_files):
                    mesh = load_mesh(mesh_file)
                    eval_dict_mesh = evaluator.eval_mesh(
                        mesh, pointcloud_tgt[i], None, points_tgt[i],
                        occ_tgt[i])
                    for k, v in eval_dict_mesh.items():
                        eval_dict['%s %d (mesh)' % (k, i)] = v
        else:
            print('Warning: mesh does not exist: %s (%d)' %
                  (modelname, start_idx))

    # Evaluate point cloud
    if cfg['test']['eval_pointcloud']:
        pointcloud_folder = os.path.join(
            pointcloud_dir, '%s' % modelname, '%05d' % start_idx)
        if os.path.exists(pointcloud_folder):
            ply_files = glob.glob(os.path.join(pointcloud_folder, '*.ply'))
            ply_files.sort()
            # Eval Pointcloud Correspondences
            if cfg['test']['eval_pointcloud_correspondences']:
                eval_dict_cor = evaluator.eval_correspondences_pointcloud(
                    ply_files, pointcloud_tgt)
                for k, v in eval_dict_cor.items():
                    eval_dict[k] = v
            # Eval Point Cloud Chamfer distances
            for i, ply_file in enumerate(ply_files[:17]):
                pointcloud = load_pointcloud(ply_file)
                eval_dict_pcl = evaluator.eval_pointcloud(
                    pointcloud, pointcloud_tgt[i])
                for k, v in eval_dict_pcl.items():
                    eval_dict['%s %d (pcl)' % (k, i)] = v
        else:
            print('Warning: pointcloud does not exist: %s (%d)' %
                  (modelname, start_idx))

    if it > 0 and (it % 50) == 0:
        # Create pandas dataframe and save
        eval_df = pd.DataFrame(eval_dicts)
        eval_df.set_index(['idx'], inplace=True)
        eval_df.to_pickle(out_file_tmp)

        # Create CSV file  with main statistics
        eval_df_class = eval_df.groupby(by=['class name']).mean()
        eval_df_class.to_csv(out_file_class_tmp)

        print('Saved tmp file after %d iterations.' % it)

# Create pandas dataframe and save
eval_df = pd.DataFrame(eval_dicts)
eval_df.set_index(['idx'], inplace=True)
eval_df.to_pickle(out_file)

# Create CSV file  with main statistics
eval_df_class = eval_df.groupby(by=['class name']).mean()
eval_df_class.to_csv(out_file_class)

# Print results
eval_df_class.loc['mean'] = eval_df_class.mean()
print(eval_df_class)
