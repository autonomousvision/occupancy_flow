import torch
from im2mesh.utils.io import export_pointcloud
import tempfile
import subprocess
import os
import trimesh


class Generator3D(object):
    r''' Generator Class for 4D Point Set Generation Network.

    While for point cloud generation the output of the network if used, for
    mesh generation, we perform surface reconstruction in the form of ball
    pivoting. In practice, this is done by using a respective meshlab script.

    Args:
        model (nn.Module): Point Set Generation Network model
        device (PyTorch Device): the PyTorch device
    '''

    def __init__(self, model, device=None):
        self.model = model.to(device)
        self.device = device


    def generate(self, data):
        r''' Generates a sequence point clouds by simply using the output of the network.

        Args:
            data (tensor): input data
            pointcloud_dir (str): pointcloud directory
            modelname (str): model name
        '''
        self.model.eval()
        device = self.device

        inputs = data.get('inputs', torch.empty(1, 0)).to(device)

        with torch.no_grad():
            points = self.model(inputs).squeeze(0)
        points = points.cpu().numpy()

        #self.save_pointclouds(points, pointcloud_dir, modelname, start_idx)

        return points

    def export(self, points, pointcloud_dir, modelname, start_idx=0):
        ''' Saves a sequence of point clouds.

        Args:
            points (numpy array): point cloud points
            pointcloud_dir (str): pointcloud directory
            modelname (str): model name
        '''
        model_folder = os.path.join(
            pointcloud_dir, modelname, '%05d' % start_idx)

        if not os.path.isdir(model_folder):
            os.makedirs(model_folder)

        out_files = []
        for i, pc in enumerate(points):

            pc_out_file = os.path.join(
                model_folder, '%s_%04d_%04d.ply' % (modelname, start_idx, i+1))
            export_pointcloud(pc, pc_out_file)
            out_files.append(pc_out_file)

        return out_files

    # TODO Needs to be implemented
    def generate_mesh(self, data):
        r''' Generates meshes by performing ball pivoting on the output of the network.

        Args:
            data (tensor): input data
        '''
        self.model.eval()
        device = self.device

        inputs = data.get('inputs', torch.empty(1, 0)).to(device)

        with torch.no_grad():
            points = self.model(inputs).squeeze(0)

        points = points.cpu().numpy()
        mesh = meshlab_poisson(points)

        return mesh


FILTER_SCRIPT_RECONSTRUCTION = '''
<!DOCTYPE FilterScript>
<FilterScript>
 <filter name="Surface Reconstruction: Ball Pivoting">
  <Param value="0" type="RichAbsPerc" max="1.4129" name="BallRadius" description="Pivoting Ball radius (0 autoguess)" min="0" tooltip="The radius of the ball pivoting (rolling) over the set of points. Gaps that are larger than the ball radius will not be filled; similarly the small pits that are smaller than the ball radius will be filled."/>
  <Param value="20" type="RichFloat" name="Clustering" description="Clustering radius (% of ball radius)" tooltip="To avoid the creation of too small triangles, if a vertex is found too close to a previous one, it is clustered/merged with it."/>
  <Param value="90" type="RichFloat" name="CreaseThr" description="Angle Threshold (degrees)" tooltip="If we encounter a crease angle that is too large we should stop the ball rolling"/>
  <Param value="false" type="RichBool" name="DeleteFaces" description="Delete intial set of faces" tooltip="if true all the initial faces of the mesh are deleted and the whole surface is rebuilt from scratch, other wise the current faces are used as a starting point. Useful if you run multiple times the algorithm with an incrasing ball radius."/>
 </filter>
</FilterScript>
'''


def meshlab_poisson(pointcloud):
    r''' Runs the meshlab ball pivoting algorithm.

    Args:
        pointcloud (numpy tensor): input point cloud
    '''
    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = os.path.join(tmpdir, 'script.mlx')
        input_path = os.path.join(tmpdir, 'input.ply')
        output_path = os.path.join(tmpdir, 'out.off')

        # Write script
        with open(script_path, 'w') as f:
            f.write(FILTER_SCRIPT_RECONSTRUCTION)

        # Write pointcloud
        export_pointcloud(pointcloud, input_path, as_text=False)

        # Export
        env = os.environ
        subprocess.Popen('meshlabserver -i ' + input_path + ' -o '
                         + output_path + ' -s ' + script_path,
                         env=env, cwd=os.getcwd(), shell=True,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                         ).wait()
        mesh = trimesh.load(output_path, process=False)

    return mesh
