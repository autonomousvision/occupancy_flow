import numpy as np


# Transforms
class PointcloudNoise(object):
    ''' Point cloud noise transformation class.

    It adds noise to point cloud data.

    Args:
        stddev (int): standard deviation
    '''

    def __init__(self, stddev):
        self.stddev = stddev

    def __call__(self, data):
        ''' Calls the transformation.

        Args:
            data (dictionary): data dictionary
        '''
        data_out = data.copy()
        points = data[None]
        noise = self.stddev * np.random.randn(*points.shape)
        noise = noise.astype(np.float32)
        data_out[None] = points + noise
        return data_out


class SubsamplePointcloud(object):
    ''' Point cloud subsampling transformation class.

    It subsamples the point cloud data.

    Args:
        N (int): number of points to be subsampled
    '''

    def __init__(self, N):
        self.N = N

    def __call__(self, data):
        ''' Calls the transformation.

        Args:
            data (dict): data dictionary
        '''
        data_out = data.copy()
        points = data[None]

        indices = np.random.randint(points.shape[0], size=self.N)
        data_out[None] = points[indices, :]

        if 'normals' in data.keys():
            normals = data['normals']
            data_out['normals'] = normals[indices, :]

        return data_out


class SubsamplePoints(object):
    ''' Points subsampling transformation class.

    It subsamples the points data.

    Args:
        N (int): number of points to be subsampled
    '''

    def __init__(self, N):
        self.N = N

    def __call__(self, data):
        ''' Calls the transformation.

        Args:
            data (dictionary): data dictionary
        '''
        points = data[None]
        occ = data['occ']

        data_out = data.copy()
        if isinstance(self.N, int):
            idx = np.random.randint(points.shape[0], size=self.N)
            data_out.update({
                None: points[idx, :],
                'occ':  occ[idx],
            })
        else:
            Nt_out, Nt_in = self.N
            occ_binary = (occ >= 0.5)
            points0 = points[~occ_binary]
            points1 = points[occ_binary]

            idx0 = np.random.randint(points0.shape[0], size=Nt_out)
            idx1 = np.random.randint(points1.shape[0], size=Nt_in)

            points0 = points0[idx0, :]
            points1 = points1[idx1, :]
            points = np.concatenate([points0, points1], axis=0)

            occ0 = np.zeros(Nt_out, dtype=np.float32)
            occ1 = np.ones(Nt_in, dtype=np.float32)
            occ = np.concatenate([occ0, occ1], axis=0)

            volume = occ_binary.sum() / len(occ_binary)
            volume = volume.astype(np.float32)

            data_out.update({
                None: points,
                'occ': occ,
                'volume': volume,
            })
        return data_out


class SubsamplePointcloudSeq(object):
    ''' Point cloud sequence subsampling transformation class.

    It subsamples the point cloud sequence data.

    Args:
        N (int): number of points to be subsampled
        connected_samples (bool): whether to obtain connected samples
        random (bool): whether to sub-sample randomly
    '''

    def __init__(self, N, connected_samples=False, random=True):
        self.N = N
        self.connected_samples = connected_samples
        self.random = random

    def __call__(self, data):
        ''' Calls the transformation.

        Args:
            data (dictionary): data dictionary
        '''
        data_out = data.copy()
        points = data[None]  # n_steps x T x 3
        n_steps, T, dim = points.shape
        N_max = min(self.N, T)
        if self.connected_samples or not self.random:
            indices = (np.random.randint(T, size=self.N) if self.random else
                       np.arange(N_max))
            data_out[None] = points[:, indices, :]
        else:
            indices = np.random.randint(T, size=(n_steps, self.N))
            data_out[None] = \
                points[np.arange(n_steps).reshape(-1, 1), indices, :]
        return data_out


class SubsamplePointsSeq(object):
    ''' Points sequence subsampling transformation class.

    It subsamples the points sequence data.

    Args:
        N (int): number of points to be subsampled
        connected_samples (bool): whether to obtain connected samples
        random (bool): whether to sub-sample randomly
    '''

    def __init__(self, N, connected_samples=False, random=True):
        self.N = N
        self.connected_samples = connected_samples
        self.random = random

    def __call__(self, data):
        ''' Calls the transformation.

        Args:
            data (dictionary): data dictionary
        '''
        points = data[None]
        occ = data['occ']
        data_out = data.copy()
        n_steps, T, dim = points.shape

        N_max = min(self.N, T)

        if self.connected_samples or not self.random:
            indices = (np.random.randint(T, size=self.N) if self.random
                       else np.arange(N_max))
            data_out.update({
                None: points[:, indices],
                'occ':  occ[:, indices],
            })
        else:
            indices = np.random.randint(T, size=(n_steps, self.N))
            help_arr = np.arange(n_steps).reshape(-1, 1)
            data_out.update({
                None: points[help_arr, indices, :],
                'occ': occ[help_arr, indices, :]
            })
        return data_out
