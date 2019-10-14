import os
import torch
from torch.nn import functional as F
from im2mesh.common import compute_iou
from torch import distributions as dist
from im2mesh.training import BaseTrainer


class Trainer(BaseTrainer):
    r''' Trainer object for ONet 4D.

    Onet 4D is trained with BCE. The Trainer object
    obtains methods to perform a train and eval step as well as to visualize
    the current training state.

    Args:
        model (nn.Module): Onet 4D model
        optimizer (PyTorch optimizer): The optimizer that should be used
        device (PyTorch device): the PyTorch device
        input_type (string): The input type (e.g. 'img')
        vis_dir (string): the visualisation directory
        threshold (float): threshold value for decision boundary
    '''

    def __init__(self, model, optimizer, device=None, input_type='img',
                 vis_dir=None, threshold=0.5):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.input_type = input_type
        self.vis_dir = vis_dir
        self.threshold = threshold

        if vis_dir is not None and not os.path.exists(vis_dir):
            os.makedirs(vis_dir)

    def train_step(self, data):
        ''' Performs a train step.

        Args:
            data (tensor): training data
        '''
        self.model.train()
        self.optimizer.zero_grad()
        loss = self.compute_loss(data)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def eval_step(self, data):
        ''' Performs a validation step.

        Args:
            data (tensor): validation data
        '''
        self.model.eval()
        device = self.device
        inputs = data.get('inputs', torch.empty(1, 0)).to(device)
        eval_dict = {}
        loss = 0
        with torch.no_grad():
            # Encode inputs
            c_t = self.model.encode_inputs(inputs)
            q_z = self.model.infer_z(inputs, c=c_t)
            z = q_z.rsample()

            # KL Divergence
            loss_kl = self.get_kl(q_z)
            eval_dict['kl'] = loss_kl.item()
            loss += loss_kl

            # IoU
            eval_dict_iou = self.eval_step_iou(data, c_t=c_t, z=z)
            for (k, v) in eval_dict_iou.items():
                eval_dict[k] = v
                loss += eval_dict['iou']

        eval_dict['loss'] = loss.mean().item()
        return eval_dict

    def eval_step_iou(self, data, c_t=None, z=None):
        ''' Calculates the IoU for the evaluation step.

        Args:
            data (tensor): training data
            c_t (tensor): temporal conditioned latent code
            z (tensor): latent code
        '''
        device = self.device
        threshold = self.threshold
        eval_dict = {}

        pts_iou = data.get('points_iou').to(device)
        occ_iou = data.get('points_iou.occ')
        pts_iou_t = data.get('points_iou.time').to(device)

        batch_size, n_steps, n_pts, dim = pts_iou.shape

        c_t = c_t.unsqueeze(1).repeat(1, n_steps, 1)
        z = z.unsqueeze(1).repeat(1, n_steps, 1)
        t_axis = pts_iou_t.unsqueeze(2).unsqueeze(3).repeat(1, 1, n_pts, 1)
        p = torch.cat([pts_iou, t_axis], dim=-1)

        # Reshape network inputs
        p = p.view(batch_size * n_steps, n_pts, -1)
        z = z.view(batch_size * n_steps, z.shape[-1])
        c_t = c_t.view(batch_size * n_steps, c_t.shape[-1])
        occ_iou = occ_iou.view(batch_size * n_steps, n_pts)

        occ_pred = self.model.decode(p, z, c_t)

        occ_pred = (occ_pred.probs > threshold).cpu().numpy()
        occ_gt = (occ_iou >= 0.5).numpy()
        iou = compute_iou(occ_pred, occ_gt)

        iou = iou.reshape(batch_size, -1).mean(0)

        eval_dict['iou'] = iou.sum() / len(iou)
        for i in range(len(iou)):
            eval_dict['iou_t%d' % i] = iou[i]

        return eval_dict

    def visualize(self, data):
        ''' Performs a visualization step.

        Args:
            data (tensor): visualization data
        '''
        print('No visualization implemented.')
        return 0

    def get_loss_recon(self, data, c_t=None, z=None):
        ''' Calculates the reconstruction loss.

        Args:
            data (tensor): training data
            c_t (tensor): temporal conditioned latent code
            z (tensor): latent code
        '''
        device = self.device
        p_t = data.get('points_t').to(device)
        occ_t = data.get('points_t.occ').to(device)
        time_val = data.get('points_t.time').to(device)
        batch_size, n_pts, _ = p_t.shape

        p = torch.cat([
            p_t, time_val.view(-1, 1, 1).repeat(1, n_pts, 1)], dim=-1)
        logits_pred = self.model.decode(p, c=c_t, z=z).logits

        loss_occ_t = F.binary_cross_entropy_with_logits(
            logits_pred, occ_t.view(batch_size, -1), reduction='none')
        loss_occ_t = loss_occ_t.mean()

        return loss_occ_t

    def get_kl(self, q_z):
        ''' Returns the KL divergence.

        Args:
            q_z (distribution): predicted distribution over latent codes
        '''
        loss_kl = dist.kl_divergence(q_z, self.model.p0_z).mean()
        if torch.isnan(loss_kl):
            loss_kl = torch.tensor([0.]).to(self.device)
        return loss_kl

    def compute_loss(self, data):
        ''' Calculates the loss.

        Args:
            data (tensor): training data
        '''
        device = self.device
        # Encode inputs
        inputs = data.get('inputs', torch.empty(1, 0)).to(device)
        c_t = self.model.encode_inputs(inputs)
        q_z = self.model.infer_z(inputs, c=c_t)
        z = q_z.rsample()

        # Losses
        # KL-divergence
        loss_kl = self.get_kl(q_z)

        # Reconstruction Loss
        loss_recon = self.get_loss_recon(data, c_t, z)

        loss = loss_recon + loss_kl
        return loss
