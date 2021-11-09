import torch
import torch.nn.functional as F
from trainers.utils.diff_ops import jacobian
from trainers.utils.igp_utils import sample_points_for_loss


def deform_ortho_jacobian(
        gtr, net, deform_net,
        x=None, dim=3, npoints=1000, ortho_reg_type='so',
        use_surf_points=True, invert_sampling=True,
        loss_type='l2', reduction='mean', weights=1, use_weight=True):

    if x is None:
        x, weights = sample_points_for_loss(
            npoints, dim=dim, use_surf_points=use_surf_points,
            gtr=gtr, net=net, deform=deform_net,
            invert_sampling=invert_sampling, return_weight=True
        )
        bs, npoints = x.size(0), x.size(1)
    else:
        if len(x.size()) == 2:
            bs, npoints = 1, x.size(0)
        else:
            bs, npoints = x.size(0), x.size(1)
    x = x.view(bs, npoints, dim)

    bs = x.size(0)
    x = x.detach().clone()
    x.requires_grad = True
    y = deform_net(x, None)
    jac_delta, _ = jacobian(y, x)
    jac_delta = jac_delta.view(bs * npoints, dim, dim)
    jac_identity = torch.eye(dim).view(1, dim, dim).to(jac_delta)
    # jac_delta = (jac_identity + jac_delta).view(bs * npoints, dim, dim)

    # Types: SO, DSO, MC and ... are referred to this paper:
    # Can we gain more from orthogonality regulairzations in training deep CNNs?
    # Bansal et. al., NeurIPS 2018
    # https://papers.nips.cc/paper/2018/file/bf424cb7b0dea050a42b9739eb261a3a-Paper.pdf
    if ortho_reg_type.lower() in ['so', 'dso']:
        diff = torch.bmm(jac_delta, jac_delta) - jac_identity
        diff = diff.view(bs, npoints, -1).norm(dim=-1, keepdim=False)
    elif ortho_reg_type.lower() in ['mc']:
        diff = torch.bmm(jac_delta, jac_delta) - jac_identity
        diff = diff.view(bs, npoints, -1).max(dim=-1, keepdim=False)[0]
    else:
        raise NotImplemented

    if use_weight:
        diff = diff * weights

    if loss_type.lower() == 'l2':
        loss = F.mse_loss(
            diff, torch.zeros_like(diff), reduction=reduction)
    elif loss_type.lower() == 'l1':
        loss = F.l1_loss(
            diff, torch.zeros_like(diff), reduction=reduction)
    else:
        raise NotImplemented
    return loss


def deform_det_jacobian(
        gtr, net, deform_net,
        x=None, dim=3, npoints=1000,
        use_surf_points=True, invert_sampling=True,
        loss_type='l2', reduction='mean', weights=1, use_weight=True,
        detach_weight=False):
    if x is None:
        x, weights = sample_points_for_loss(
            npoints, dim=dim, use_surf_points=use_surf_points,
            gtr=gtr, net=net, deform=deform_net,
            invert_sampling=invert_sampling, return_weight=True,
            detach_weight=detach_weight
        )
        bs, npoints = x.size(0), x.size(1)
    else:
        if len(x.size()) == 2:
            bs, npoints = 1, x.size(0)
        else:
            bs, npoints = x.size(0), x.size(1)
    x = x.view(bs, npoints, dim)

    bs = x.size(0)
    x = x.detach().clone()
    x.requires_grad = True
    y = deform_net(x, None)
    jac_delta, status = jacobian(y, x)
    assert status == 0
    jac_det = torch.abs(torch.linalg.det(jac_delta.view(bs, npoints, dim, dim)).view(bs, npoints))
    diff = jac_det - 1

    if use_weight:
        diff = diff * weights

    if loss_type.lower() == 'l2':
        loss = F.mse_loss(
            diff, torch.zeros_like(diff), reduction=reduction)
    elif loss_type.lower() == 'l1':
        loss = F.l1_loss(
            diff, torch.zeros_like(diff), reduction=reduction)
    else:
        raise NotImplemented
    return loss
