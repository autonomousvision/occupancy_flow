import os
import glob
import random
from PIL import Image
import numpy as np
import trimesh
from im2mesh.data.core import Field
import torch


class IndexField(Field):
    ''' Basic index field.'''
    # def load(self, model_path, idx, category):

    def load(self, model_path, idx, category, start_idx=0, **kwargs):
        ''' Loads the index field.

        Args:
            model_path (str): path to model
            idx (int): ID of data point
            category (int): index of category
            start_idx (int): id of sequence start
        '''
        return idx

    def check_complete(self, files):
        ''' Check if field is complete.

        Args:
            files: files
        '''
        return True


class CategoryField(Field):
    ''' Basic category field.'''

    def load(self, model_path, idx, category):
        ''' Loads the category field.

        Args:
            model_path (str): path to model
            idx (int): ID of data point
            category (int): index of category
        '''
        return category

    def check_complete(self, files):
        ''' Check if field is complete.

        Args:
            files: files
        '''
        return True


class PointsSubseqField(Field):
    ''' Points subsequence field class.

    Args:
        folder_name (str): points folder name
        transform (transform): transform
        seq_len (int): length of sequence
        all_steps (bool): whether to return all time steps
        fixed_time_step (int): if and which fixed time step to use
        unpackbits (bool): whether to unpack bits
    '''

    def __init__(self, folder_name, transform=None, seq_len=17,
                 all_steps=False, fixed_time_step=None, unpackbits=False,
                 **kwargs):
        self.folder_name = folder_name
        self.transform = transform
        self.seq_len = seq_len
        self.all_steps = all_steps
        self.sample_padding = 0.1
        self.fixed_time_step = fixed_time_step
        self.unpackbits = unpackbits

    def get_loc_scale(self, mesh):
        ''' Returns location and scale of mesh.

        Args:
            mesh (trimesh): mesh
        '''
        bbox = mesh.bounding_box.bounds

        # Compute location and scale with padding of 0.1
        loc = (bbox[0] + bbox[1]) / 2
        scale = (bbox[1] - bbox[0]).max() / (1 - self.sample_padding)

        return loc, scale

    def normalize_mesh(self, mesh, loc, scale):
        ''' Normalize mesh.

        Args:
            mesh (trimesh): mesh
            loc (tuple): location for normalization
            scale (float): scale for normalization
        '''
        # Transform input mesh
        mesh.apply_translation(-loc)
        mesh.apply_scale(1 / scale)

        return mesh

    def load_files(self, model_path, start_idx):
        ''' Loads the model files.

        Args:
            model_path (str): path to model
            start_idx (int): id of sequence start
        '''
        folder = os.path.join(model_path, self.folder_name)
        files = glob.glob(os.path.join(folder, '*.npz'))
        files.sort()
        files = files[start_idx:start_idx+self.seq_len]

        return files

    def load_all_steps(self, files, points_dict, loc0, scale0):
        ''' Loads data for all steps.

        Args:
            files (list): list of files
            points_dict (dict): points dictionary for first step of sequence
            loc0 (tuple): location of first time step mesh
            scale0 (float): scale of first time step mesh
        '''
        p_list = []
        o_list = []
        t_list = []
        for i, f in enumerate(files):
            points_dict = np.load(f)
            
            # Load points
            points = points_dict['points']
            if (points.dtype == np.float16):
                # break symmetry (nec. for some version?)
                points = points.astype(np.float32)
                points += 1e-4 * np.random.randn(*points.shape)
            points = points.astype(np.float32)
            occupancies = points_dict['occupancies']
            if self.unpackbits:
                occupancies = np.unpackbits(occupancies)[:points.shape[0]]
            occupancies = occupancies.astype(np.float32)
            loc = points_dict['loc'].astype(np.float32)
            scale = points_dict['scale'].astype(np.float32)
            # Transform to loc0, scale0
            points = (loc + scale * points - loc0) / scale0
            time = np.array(i / (self.seq_len - 1), dtype=np.float32)

            p_list.append(points)
            o_list.append(occupancies)
            t_list.append(time)

        data = {
            None: np.stack(p_list),
            'occ': np.stack(o_list),
            'time': np.stack(t_list),
        }

        return data

    def load_single_step(self, files, points_dict, loc0, scale0):
        ''' Loads data for a single step.

        Args:
            files (list): list of files
            points_dict (dict): points dictionary for first step of sequence
            loc0 (tuple): location of first time step mesh
            scale0 (float): scale of first time step mesh
        '''
        if self.fixed_time_step is None:
            # Random time step
            time_step = np.random.choice(self.seq_len)
        else:
            time_step = int(self.fixed_time_step)

        if time_step != 0:
            points_dict = np.load(files[time_step])
        # Load points
        points = points_dict['points'].astype(np.float32)
        occupancies = points_dict['occupancies']
        if self.unpackbits:
            occupancies = np.unpackbits(occupancies)[:points.shape[0]]
        occupancies = occupancies.astype(np.float32)
        loc = points_dict['loc'].astype(np.float32)
        scale = points_dict['scale'].astype(np.float32)
        # Transform to loc0, scale0
        points = (loc + scale * points - loc0) / scale0

        if self.seq_len > 1:
            time = np.array(
                time_step / (self.seq_len - 1), dtype=np.float32)
        else:
            time = np.array([1], dtype=np.float32)

        data = {
            None: points,
            'occ': occupancies,
            'time': time,
        }
        return data

    def load(self, model_path, idx, c_idx=None, start_idx=0, **kwargs):
        ''' Loads the points subsequence field.

        Args:
            model_path (str): path to model
            idx (int): ID of data point
            category (int): index of category
            start_idx (int): id of sequence start
        '''
        files = self.load_files(model_path, start_idx)
        # Load loc and scale from t_0
        points_dict = np.load(files[0])
        loc0 = points_dict['loc'].astype(np.float32)
        scale0 = points_dict['scale'].astype(np.float32)

        if self.all_steps:
            data = self.load_all_steps(files, points_dict, loc0, scale0)
        else:
            data = self.load_single_step(files, points_dict, loc0, scale0)

        if self.transform is not None:
            data = self.transform(data)
        return data


class ImageSubseqField(Field):
    ''' Image subsequence field class.

    Args:
        folder_name (str): points folder name
        transform (transform): transform
        seq_len (int): length of sequence
        extension (str): image extension
        random_view (bool): whether to return a random view
        only_end_points (bool): whether to only return end points
    '''

    def __init__(self, folder_name, transform=None, seq_len=17,
                 extension='jpg', random_view=True, only_end_points=False,
                 **kwargs):
        self.folder_name = folder_name
        self.transform = transform
        self.seq_len = seq_len
        self.extension = extension
        self.only_end_points = only_end_points
        self.random_view = random_view
        self.num_views = 4

    def get_img_seq_idx(self):
        ''' Returns image sequence idx.
        '''
        if self.random_view:
            idx_img_seq = random.randint(0, self.num_views - 1)
        else:
            idx_img_seq = 0
        return idx_img_seq

    def get_file_paths(self, model_path, start_idx):
        ''' Returns file paths.

        Args:
            model_path (str): model path
            start_idx (int): id of sequence start
        '''
        idx_img_seq = self.get_img_seq_idx()
        folder = os.path.join(model_path, self.folder_name)
        folder = os.path.join(folder, '%03d' % idx_img_seq)
        files = glob.glob(os.path.join(folder, '*.%s' % self.extension))
        files.sort()
        files = files[start_idx:start_idx+self.seq_len]
        if self.only_end_points:
            files = [files[0], files[-1]]

        return files

    def load(self, model_path, idx, c_idx=None, start_idx=0, **kwargs):
        ''' Loads the image sequence field.

        Args:
            model_path (str): path to model
            idx (int): ID of data point
            c_idx (int): index of category
            start_idx (int): id of sequence start
        '''
        files = self.get_file_paths(model_path, start_idx)
        imgs = []
        for f in files:
            image = Image.open(f).convert('RGB')
            if self.transform is not None:
                image = self.transform(image)
            imgs.append(image)

        data = {
            None: torch.stack(imgs),
        }

        return data


class PointCloudSubseqField(Field):
    ''' Point cloud subsequence field class.

    Args:
        folder_name (str): points folder name
        transform (transform): transform
        seq_len (int): length of sequence
        only_end_points (bool): whether to only return end points
        scale_pointcloud (bool): whether to scale the point cloud
            w.r.t. the first point cloud of the sequence
    '''

    def __init__(self, folder_name, transform=None, seq_len=17,
                 only_end_points=False,
                 scale_pointcloud=True):
        self.folder_name = folder_name
        self.transform = transform
        self.seq_len = seq_len
        self.only_end_points = only_end_points
        self.scale_pointcloud = scale_pointcloud

    def return_loc_scale(self, mesh):
        ''' Returns location and scale of mesh.

        Args:
            mesh (trimesh): mesh
        '''
        bbox = mesh.bounding_box.bounds
        # Compute location and scale
        loc = (bbox[0] + bbox[1]) / 2
        scale = (bbox[1] - bbox[0]).max() / (1 - 0)
        return loc, scale

    def apply_normalization(self, mesh, loc, scale):
        ''' Normalizes the mesh.

        Args:
            mesh (trimesh): mesh
            loc (tuple): location for normalization
            scale (float): scale for normalization
        '''
        mesh.apply_translation(-loc)
        mesh.apply_scale(1/scale)
        return mesh

    def load_files(self, model_path, start_idx):
        ''' Loads the model files.

        Args:
            model_path (str): path to model
            start_idx (int): id of sequence start
        '''
        folder = os.path.join(model_path, self.folder_name)
        files = glob.glob(os.path.join(folder, '*.npz'))
        files.sort()
        files = files[start_idx:start_idx+self.seq_len]

        if self.only_end_points:
            files = [files[0], files[-1]]

        return files

    def load_single_file(self, file_path):
        ''' Loads a single file.

        Args:
            file_path (str): file path
        '''
        pointcloud_dict = np.load(file_path)
        points = pointcloud_dict['points'].astype(np.float32)
        loc = pointcloud_dict['loc'].astype(np.float32)
        scale = pointcloud_dict['scale'].astype(np.float32)

        return points, loc, scale

    def get_time_values(self):
        ''' Returns the time values.
        '''
        if self.seq_len > 1:
            time = \
                np.array([i/(self.seq_len - 1) for i in range(self.seq_len)],
                         dtype=np.float32)
        else:
            time = np.array([1]).astype(np.float32)
        return time

    def load(self, model_path, idx, c_idx=None, start_idx=0, **kwargs):
        ''' Loads the point cloud sequence field.

        Args:
            model_path (str): path to model
            idx (int): ID of data point
            c_idx (int): index of category
            start_idx (int): id of sequence start
        '''
        pc_seq = []
        # Get file paths
        files = self.load_files(model_path, start_idx)
        # Load first pcl file
        _, loc0, scale0 = self.load_single_file(files[0])
        for f in files:
            points, loc, scale = self.load_single_file(f)
            # Transform mesh to loc0 / scale0
            if self.scale_pointcloud:
                points = (loc + scale * points - loc0) / scale0

            pc_seq.append(points)

        data = {
            None: np.stack(pc_seq),
            'time': self.get_time_values(),
        }

        if self.transform is not None:
            data = self.transform(data)
        return data


class MeshSubseqField(Field):
    ''' Mesh subsequence field class.

    Args:
        folder_name (str): points folder name
        seq_len (int): length of sequence
        only_end_points (bool): whether to only return end points
        only_start_point (bool): whether to only return the start point
        scale (bool): whether to scale the meshes w.r.t. the first mesh of the
            sequence
        file_ext (str): mesh file extension
    '''

    def __init__(self, folder_name, seq_len=17, only_end_points=False,
                 only_start_point=False, scale=True, file_ext='obj'):
        self.folder_name = folder_name
        self.seq_len = seq_len
        self.only_end_points = only_end_points
        self.only_start_point = only_start_point
        self.scale = scale
        self.file_ext = file_ext

    def return_loc_scale(self, mesh):
        ''' Returns location and scale of mesh.

        Args:
            mesh (trimesh): mesh
        '''
        bbox = mesh.bounding_box.bounds
        # Compute location and scale
        loc = (bbox[0] + bbox[1]) / 2
        scale = (bbox[1] - bbox[0]).max() / (1 - 0)
        return loc, scale

    def apply_normalization(self, mesh, loc, scale):
        ''' Normalizes the mesh.

        Args:
            mesh (trimesh): mesh
            loc (tuple): location for normalization
            scale (float): scale for normalization
        '''
        mesh.apply_translation(-loc)
        mesh.apply_scale(1/scale)
        return mesh

    def load(self, model_path, idx, c_idx=None, start_idx=0, **kwargs):
        ''' Loads the mesh sequence field.

        Args:
            model_path (str): path to model
            idx (int): ID of data point
            c_idx (int): index of category
            start_idx (int): id of sequence start
        '''
        folder = os.path.join(model_path, self.folder_name)
        mesh_files = glob.glob(os.path.join(folder, '*.%s' % self.file_ext))
        mesh_files.sort()
        mesh_files = mesh_files[start_idx:start_idx+self.seq_len]
        if self.only_end_points:
            mesh_files = [mesh_files[0], mesh_files[-1]]
        elif self.only_start_point:
            mesh_files = mesh_files[[0]]

        if self.scale:
            mesh_0 = trimesh.load(mesh_files[0], process=False)
            loc, scale = self.return_loc_scale(mesh_0)

            vertices = []
            for mesh_p in mesh_files:
                mesh = trimesh.load(mesh_p, process=False)
                mesh = self.apply_normalization(mesh, loc, scale)
                vertices.append(np.array(mesh.vertices, dtype=np.float32))

        faces = np.array(trimesh.load(
            mesh_files[0], process=False).faces, dtype=np.float32)
        data = {
            'vertices': np.stack(vertices),
            'faces': faces
        }

        return data
