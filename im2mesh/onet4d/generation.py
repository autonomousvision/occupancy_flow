import torch
import os
from im2mesh.utils.io import save_mesh
import time
from im2mesh.utils.onet_generator import Generator3D as Generator3DONet


class Generator3D(object):
    '''  Generator class for Occupancy Networks 4D.

    It provides functions to generate the final mesh as well refining options.

    Args:
        model (nn.Module): trained Occupancy Network model
        device (device): pytorch device
        points_batch_size (int): batch size for points evaluation
        threshold (float): threshold value
        refinement_step (int): number of refinement steps
        resolution0 (int): start resolution for MISE
        upsampling steps (int): number of upsampling steps
        with_normals (bool): whether normals should be estimated
        padding (float): how much padding should be used for MISE
        sample (bool): whether z should be sampled
        simplify_nfaces (int): number of faces the mesh should be simplified to
        n_time_steps (int): number of time steps to generate
        only_ent_time_points (bool): whether to only generate end points
    '''

    def __init__(self, model, device=None, points_batch_size=100000,
                 threshold=0.5, refinement_step=0,
                 resolution0=16, upsampling_steps=3,
                 with_normals=False, padding=0.1,
                 sample=False, simplify_nfaces=None, n_time_steps=17,
                 only_end_time_points=False, **kwargs):
        self.n_time_steps = n_time_steps
        self.only_end_time_points = only_end_time_points
        self.onet_generator = Generator3DONet(
            model, device=device,
            points_batch_size=points_batch_size,
            threshold=threshold, refinement_step=refinement_step,
            resolution0=resolution0, upsampling_steps=upsampling_steps,
            with_normals=with_normals, padding=padding,
            sample=sample,
            simplify_nfaces=simplify_nfaces)

    def generate_mesh_t0(self, z=None, c_t=None, data=None, stats_dict={}):
        ''' Generates mesh at first time step.

        Args:
            z (tensor): latent code z
            c_t (tensor): latent conditioned temporal code c_t
            data (dict): data dictionary
            stats_dict (dict): statistics dictionary
        '''
        t = torch.tensor([0.]).view(1, 1).to(self.onet_generator.device)
        kwargs = {'t': t}
        mesh = self.onet_generator.generate_from_latent(
            z, c_t, stats_dict=stats_dict, **kwargs)
        return mesh

    def get_time_steps(self):
        ''' Return time steps values.
        '''
        n_steps = self.n_time_steps
        device = self.onet_generator.device

        if self.only_end_time_points:
            t = torch.tensor([0., 1.]).to(device)
        else:
            t = (torch.arange(1, n_steps).float() / (n_steps - 1)).to(device)

        return t

    def generate_meshes_t(self, z=None, c_t=None, data=None, stats_dict={}):
        ''' Generates meshes at time steps > 0.

        Args:
            z (tensor): latent code z
            c_t (tensor): latent conditioned temporal code c_t
            data (dict): data dictionary
            stats_dict (dict): statistics dictionary
        '''
        t = self.get_time_steps()
        meshes = []
        for i, t_v in enumerate(t):
            kwargs = {'t': t_v.view(1, 1)}
            stats_dict_i = {}
            mesh = self.onet_generator.generate_from_latent(
                z, c_t, stats_dict=stats_dict_i, **kwargs)
            meshes.append(mesh)
            for k, v in stats_dict_i.items():
                stats_dict[k] += v

        return meshes

    def export_mesh(self, mesh, model_folder, modelname, start_idx=0, n_id=1):
        ''' Exports a mesh.

        Args:
            mesh(trimesh): mesh to export
            model_folder (str): model folder
            model_name (str): name of the model
            start_idx (int): start id of sequence
            n_id (int): number of mesh in the sequence (e.g. 1 -> start)
        '''
        out_path = os.path.join(
            model_folder, '%s_%04d_%04d.off' % (modelname, start_idx, n_id))
        save_mesh(mesh, out_path)
        return out_path

    def export_meshes_t(self, meshes, model_folder, modelname, start_idx=0,
                        start_id_seq=2):
        ''' Exports meshes.

        Args:
            meshes (list): list of meshes to export
            model_folder (str): model folder
            model_name (str): name of the model
            start_idx (int): start id of sequence
            start_id_seq (int): start number of first mesh in the sequence
        '''
        out_files = []
        for i, m in enumerate(meshes):
            out_file = self.export_mesh(
                m, model_folder, modelname, start_idx, n_id=start_id_seq + i)
            out_files.append(out_file)

        return out_files

    def export(self, meshes, mesh_dir, modelname, start_idx=0, start_id_seq=1):
        ''' Exports a list of meshes.

        Args:
            meshes (list): list of meshes to export
            model_folder (str): model folder
            model_name (str): name of the model
            start_idx (int): start id of sequence
            start_id_seq (int): start number of first mesh in the sequence
        '''
        model_folder = os.path.join(mesh_dir, modelname, '%05d' % start_idx)
        if not os.path.isdir(model_folder):
            os.makedirs(model_folder)

        return self.export_meshes_t(
            meshes, model_folder, modelname, start_idx=0, start_id_seq=1)

    def generate(self, data, return_stats=True, **kwargs):
        ''' Generates meshes for input data.

        Args:
            data (dict): data dictionary
            return_stats (bool): whether to return statistics
        '''
        self.onet_generator.model.eval()
        stats_dict = {}
        device = self.onet_generator.device
        inputs = data.get('inputs', torch.empty(1, 0)).to(device)

        meshes = []
        with torch.no_grad():
            t0 = time.time()
            c_t = self.onet_generator.model.encode_inputs(inputs)
            # Only for testing
            z = self.onet_generator.model.get_z_from_prior(
                (1,), sample=self.onet_generator.sample).to(device)
            stats_dict['time (encode inputs)'] = time.time() - t0

            # Generate and save first mesh
            mesh_t0 = self.generate_mesh_t0(
                z, c_t, data, stats_dict=stats_dict)
            meshes.append(mesh_t0)
            # Generate and save later time steps
            meshes_t = self.generate_meshes_t(
                z=z, c_t=c_t, data=data, stats_dict=stats_dict)
            meshes.extend(meshes_t)

        return meshes, stats_dict
