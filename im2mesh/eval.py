# from im2mesh import icp
import logging
import numpy as np
import trimesh
# from scipy.spatial import cKDTree
from im2mesh.utils.libkdtree import KDTree
from im2mesh.utils.libmesh import check_mesh_contains
from im2mesh.common import compute_iou, get_nearest_neighbors_indices_batch
from im2mesh.utils.io import load_pointcloud, load_mesh


# Maximum values for bounding box [-0.5, 0.5]^3
EMPTY_PCL_DICT = {
    'completeness': np.sqrt(3),
    'accuracy': np.sqrt(3),
    'completeness2': 3,
    'accuracy2': 3,
    'chamfer': 6,
}

EMPTY_PCL_DICT_NORMALS = {
    'normals completeness': -1.,
    'normals accuracy': -1.,
    'normals': -1.,
}

logger = logging.getLogger(__name__)


class MeshEvaluator(object):
    ''' Mesh evaluation class.

    It handles the mesh evaluation process.

    Args:
        n_points (int): number of points to be used for evaluation
    '''

    def __init__(self, n_points=100000):
        self.n_points = n_points

    def eval_mesh(self, mesh, pointcloud_tgt, normals_tgt,
                  points_iou, occ_tgt):
        ''' Evaluates a mesh.

        Args:
            mesh (trimesh): mesh which should be evaluated
            pointcloud_tgt (numpy array): target point cloud
            normals_tgt (numpy array): target normals
            points_iou (numpy_array): points tensor for IoU evaluation
            occ_tgt (numpy_array): GT occupancy values for IoU points
        '''
        if len(mesh.vertices) != 0 and len(mesh.faces) != 0:
            pointcloud, idx = mesh.sample(self.n_points, return_index=True)
            pointcloud = pointcloud.astype(np.float32)
            normals = mesh.face_normals[idx]
        else:
            pointcloud = np.empty((0, 3))
            normals = np.empty((0, 3))

        out_dict = self.eval_pointcloud(
            pointcloud, pointcloud_tgt, normals, normals_tgt)

        if len(mesh.vertices) != 0 and len(mesh.faces) != 0:
            occ = check_mesh_contains(mesh, points_iou)
            out_dict['iou'] = compute_iou(occ, occ_tgt)
        else:
            out_dict['iou'] = 0.

        return out_dict

    def eval_pointcloud(self, pointcloud, pointcloud_tgt,
                        normals=None, normals_tgt=None):
        ''' Evaluates a point cloud.

        Args:
            pointcloud (numpy array): predicted point cloud
            pointcloud_tgt (numpy array): target point cloud
            normals (numpy array): predicted normals
            normals_tgt (numpy array): target normals
        '''
        # Return maximum losses if pointcloud is empty
        if pointcloud.shape[0] == 0:
            logger.warn('Empty pointcloud / mesh detected!')
            out_dict = EMPTY_PCL_DICT.copy()
            if normals is not None and normals_tgt is not None:
                out_dict.update(EMPTY_PCL_DICT_NORMALS)
            return out_dict

        pointcloud = np.asarray(pointcloud)
        pointcloud_tgt = np.asarray(pointcloud_tgt)

        # Completeness: how far are the points of the target point cloud
        # from thre predicted point cloud
        completeness, completeness_normals = distance_p2p(
            pointcloud_tgt, normals_tgt, pointcloud, normals
        )
        completeness2 = completeness**2

        completeness = completeness.mean()
        completeness2 = completeness2.mean()
        completeness_normals = completeness_normals.mean()

        # Accuracy: how far are th points of the predicted pointcloud
        # from the target pointcloud
        accuracy, accuracy_normals = distance_p2p(
            pointcloud, normals, pointcloud_tgt, normals_tgt
        )
        accuracy2 = accuracy**2

        accuracy = accuracy.mean()
        accuracy2 = accuracy2.mean()
        accuracy_normals = accuracy_normals.mean()

        # Chamfer distance
        chamfer = completeness2 + accuracy2
        normals_correctness = (
            0.5 * completeness_normals + 0.5 * accuracy_normals
        )

        out_dict = {
            'completeness': completeness,
            'accuracy': accuracy,
            'normals completeness': completeness_normals,
            'normals accuracy': accuracy_normals,
            'normals': normals_correctness,
            'completeness2': completeness2,
            'accuracy2': accuracy2,
            'chamfer': chamfer,
        }

        return out_dict

    def eval_correspondences_NN(self, vertices, pcl_inp, n_pts=10000):
        ''' Evaluates correspondences with nearest neighbor algorithm.
        The vertices of time time 0 are taken, and then the predictions for
        later time steps are the NNs for these vertices in the later point
        clouds.

        Args:
            vertices (numpy array): vertices of size L x N_v x 3
            pcl_inp (numpy array): random point clouds of size L x N_pcl x 3
            n_pts (int): how many points should be used from the point clouds
        '''
        n_t, n_pt = pcl_inp.shape[:2]
        v = np.expand_dims(vertices[0], axis=0)
        eval_dict = {}
        for i in range(n_t):
            # subsample pointcloud for given n_pts
            pcl = pcl_inp[i, np.random.randint(n_pt, size=(n_pts)), :]
            # obtain NN in current pcl
            ind, _ = get_nearest_neighbors_indices_batch(
                v, np.expand_dims(pcl, axis=0))[0]
            # select NNs as predictions for current vertices
            pred_v = pcl[ind]
            # calculate l2 norm between predicted and GT vertices at step i
            l2_loss = np.mean(np.linalg.norm(
                pred_v - vertices[i], axis=-1)).item()
            eval_dict['l2 %d (mesh)' % i] = l2_loss
        return eval_dict

    def eval_correspondences_mesh(self, mesh_files, pcl_tgt,
                                  project_to_final_mesh=False):
        ''' Calculates correspondence score for meshes.

        Args:
            mesh_files (list): list of mesh files
            pcl_tgt (list): list of target point clouds
            project_to_final_mesh (bool): whether to project predictions to
                GT mesh by finding its NN in the target point cloud
        '''
        mesh_t0 = load_mesh(mesh_files[0])
        mesh_pts_t0 = mesh_t0.vertices
        mesh_pts_t0 = np.expand_dims(mesh_pts_t0.astype(np.float32), axis=0)
        ind, _ = get_nearest_neighbors_indices_batch(
            mesh_pts_t0, np.expand_dims(pcl_tgt[0], axis=0))
        ind = ind[0].astype(int)
        # Nex time steps
        eval_dict = {}
        for i in range(len(pcl_tgt)):
            v_t = load_mesh(mesh_files[i]).vertices
            pc_nn_t = pcl_tgt[i][ind]

            if project_to_final_mesh and i == (len(pcl_tgt)-1):
                ind2, _ = get_nearest_neighbors_indices_batch(
                    np.expand_dims(v_t, axis=0).astype(np.float32),
                    np.expand_dims(pcl_tgt[i], axis=0))
                v_t = pcl_tgt[i][ind2[0]]
            l2_loss = np.mean(np.linalg.norm(v_t - pc_nn_t, axis=-1)).item()

            eval_dict['l2 %d (mesh)' % i] = l2_loss

        return eval_dict

    def eval_correspondences_pointcloud(self, pcl_pred_files, pcl_tgt):
        ''' Calculates correspondence score for point clouds.

        Args:
            pcl_pred_files (list): list of point cloud prediction files
            pcl_tgt (list): list of target point clouds
        '''
        # Detect NN for GT points
        pc_t0 = np.expand_dims(load_pointcloud(pcl_pred_files[0]), axis=0)
        ind, _ = get_nearest_neighbors_indices_batch(
            pc_t0, np.expand_dims(pcl_tgt[0], axis=0))
        ind = ind[0]

        # Next time steps
        eval_dict = {}
        for i in range(len(pcl_tgt)):
            pc_t = load_pointcloud(pcl_pred_files[i])
            pc_nn_t = pcl_tgt[i][ind]

            l2_loss = np.mean(np.linalg.norm(pc_t - pc_nn_t, axis=-1)).item()
            eval_dict['l2 %d (pcl)' % i] = l2_loss

        return eval_dict


def distance_p2p(points_src, normals_src, points_tgt, normals_tgt):
    ''' Computes minimal distances of each point in points_src to points_tgt.

    Args:
        points_src (numpy array): source points
        normals_src (numpy array): source normals
        points_tgt (numpy array): target points
        normals_tgt (numpy array): target normals
    '''
    kdtree = KDTree(points_tgt)
    dist, idx = kdtree.query(points_src)

    if normals_src is not None and normals_tgt is not None:
        normals_src = \
            normals_src / np.linalg.norm(normals_src, axis=-1, keepdims=True)
        normals_tgt = \
            normals_tgt / np.linalg.norm(normals_tgt, axis=-1, keepdims=True)

        normals_dot_product = (normals_tgt[idx] * normals_src).sum(axis=-1)
        # Handle normals that point into wrong direction gracefully
        # (mostly due to mehtod not caring about this in generation)
        normals_dot_product = np.abs(normals_dot_product)
    else:
        normals_dot_product = np.array(
            [np.nan] * points_src.shape[0], dtype=np.float32)
    return dist, normals_dot_product


def distance_p2m(points, mesh):
    ''' Compute minimal distances of each point in points to mesh.

    Args:
        points (numpy array): points array
        mesh (trimesh): mesh

    '''
    _, dist, _ = trimesh.proximity.closest_point(mesh, points)
    return dist
