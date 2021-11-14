import torch
import torch.nn.functional as F
from trainers.utils.diff_ops import gradient
from trainers.utils.igp_utils import sample_points_for_loss


def loss_eikonal(
        net,  gtr=None, deform=None,
        npoints=1000, use_surf_points=False, invert_sampling=True,
        x=None, dim=3, reduction='mean',
        weights=1, use_weights=True
):
    if x is None:
        x, weights = sample_points_for_loss(
            npoints, dim=dim, use_surf_points=use_surf_points,
            gtr=gtr, net=net, deform=deform, invert_sampling=invert_sampling,
            return_weight=True
        )
        bs, npoints = x.size(0), x.size(1)
    else:
        if len(x.size()) == 2:
            bs, npoints = 1, x.size(0)
        else:
            bs, npoints = x.size(0), x.size(1)
    x = x.view(bs, npoints, dim)

    x.requires_grad = True
    y = net(x.view(1, -1, dim))
    grad_norm = gradient(y, x).view(-1, dim).norm(dim=-1)
    loss_all = torch.nn.functional.mse_loss(
        grad_norm, torch.ones_like(grad_norm), reduction='none')
    if use_weights:
        loss_all = loss_all * weights

    if reduction == 'none':
        loss = loss_all
    elif reduction == 'mean':
        loss = loss_all.mean()
    elif reduction == 'sum':
        loss = loss_all.sum()
    else:
        raise NotImplementedError
    return loss

