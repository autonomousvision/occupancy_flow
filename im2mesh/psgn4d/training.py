import os
import torch
from im2mesh.common import chamfer_distance
from im2mesh.training import BaseTrainer


class Trainer(BaseTrainer):
    r''' Trainer object for the Point Set Generation Network.

    The PSGN network is trained on Chamfer distance. The Trainer object
    obtains methods to perform a train and eval step as well as to visualize
    the current training state by plotting the respective point clouds.

    Args:
        model (nn.Module): PSGN model
        optimizer (PyTorch optimizer): The optimizer that should be used
        device (PyTorch device): the PyTorch device
        input_type (string): The input type (e.g. 'img')
        vis_dir (string): the visualisation directory
    '''

    def __init__(self, model, optimizer, device=None, input_type='img',
                 vis_dir=None, loss_corr=False):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.input_type = input_type
        self.vis_dir = vis_dir
        self.loss_corr = loss_corr
        if vis_dir is not None and not os.path.exists(vis_dir):
            os.makedirs(vis_dir)

    def train_step(self, data):
        r''' Performs a train step.

        The chamfer loss is calculated and an appropriate backward pass is
        performed.

        Args:
            data (tensor): training data
        '''
        self.model.train()
        points = data.get('points_mesh').to(self.device)
        inputs = data.get('inputs').to(self.device)

        loss = self.compute_loss(points, inputs)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def eval_step(self, data):
        r''' Performs an evaluation step.

        The chamfer loss is calculated and returned in a dictionary.

        Args:
            data (tensor): input data
        '''
        self.model.eval()

        points = data.get('points_mesh').to(self.device)
        inputs = data.get('inputs').to(self.device)
        batch_size, n_steps, n_pts, dim = points.shape

        with torch.no_grad():
            loss = self.compute_loss(points, inputs).item()

        eval_dict = {
            'loss': loss,
            'chamfer': loss,
        }

        return eval_dict

    def visualize(self, data):
        r''' Visualizes the current output data of the model.

        Args:
            data (tensor): input data
        '''
        print('not implemented')
        return 0

    def compute_loss(self, points, inputs):
        r''' Computes the loss.

        The Point Set Generation Network is trained on the Chamfer distance.

        Args:
            points (tensor): GT point cloud data
            inputs (tensor): input data for the model
        '''
        batch_size, n_steps, n_pts, dim = points.shape

        points_out = self.model(inputs)
        if self.loss_corr:
            n_pts_pred = points_out.shape[2]
            points_out = points_out.transpose(1, 2).contiguous().view(
                batch_size, n_pts_pred, -1)
            points = points.transpose(1, 2).contiguous().view(
                batch_size, n_pts, -1)
        else:
            points_out = points_out.contiguous().view(
                batch_size*n_steps, -1, dim)
            points = points.contiguous().view(-1, n_pts, dim)
        loss = chamfer_distance(points, points_out).mean()
        return loss
