import torch
import torch.nn.functional as F
from trainers.utils.diff_ops import laplace
from trainers.utils.igp_utils import get_surf_pcl, sample_points_for_loss


def loss_boundary(gtr, net, npoints=1000, dim=3, x=None, use_surf_points=False):
    """
    This function tries to enforce that the field [gtr] and [net] are similar.
    Basically computing |gtr(x) - net(x)| for some [x].
    [x] will be sampled from surface of [gtr] if [use_surf_points] is True
    Otherwise, [x] is sampled from [-1, 1]^3

    :param gtr:
    :param net:
    :param npoints:
    :param dim:
    :param x:
    :param use_surf_points:
    :return:
    """
    if x is None:
        if use_surf_points:
            assert net is not None
            x = get_surf_pcl(
                gtr, npoints=npoints, dim=dim,
                steps=5, noise_sigma=1e-3, filtered=False, sigma_decay=1.
            ).detach().cuda().float()
        else:
            x = torch.rand(1, npoints, dim).cuda().float() * 2 - 1
        bs = 1
        x = x.view(bs, npoints, dim)
    else:
        if len(x.size()) == 2:
            bs, npoints = 1, x.size(0)
        else:
            bs, npoints = x.size(0), x.size(1)
    x = x.view(bs, npoints, dim)

    if use_surf_points:
        net_y = net(x)
        loss_all = F.mse_loss(net_y, torch.zeros_like(net_y), reduction='none')
    else:
        net_y = net(x)
        gtr_y = gtr(x)
        loss_all = F.mse_loss(net_y, gtr_y, reduction='none')
    loss_all = loss_all.view(bs, npoints)
    loss = loss_all.mean()
    return loss, x


def loss_lap(
        gtr, net, deform=None,
        x=None, npoints=1000, dim=3, use_surf_points=True, invert_sampling=True,
        beta=1., masking_thr=10, return_mask=False, use_weights=False, weights=1
):
    """
    Matching the Laplacian between [gtr] and [net] on sampled points.

    :param gtr:
    :param net:
    :param deform:
    :param x:
    :param npoints:
    :param dim:
    :param use_surf_points:
    :param invert_sampling:
    :param beta:
    :param masking_thr:
    :param return_mask:
    :param use_weights:
    :param weights:
    :return:
    """
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

    if deform is None:
        gtr_x = x
    else:
        gtr_x = deform(x, None)
    gtr_x = gtr_x.view(bs, npoints, dim).contiguous()
    if gtr_x.is_leaf:
        gtr_x.requires_grad = True
    else:
        gtr_x.retain_grad()
    gtr_y = gtr(gtr_x)
    lap_gtr = laplace(gtr_y, gtr_x, normalize=True).view(bs, npoints)

    if x.is_leaf:
        x.requires_grad = True
    else:
        x.retain_grad()
    net_y = net(x)
    lap_net = laplace(net_y, x, normalize=True).view(*lap_gtr.shape)

    diff = lap_gtr * beta - lap_net
    if masking_thr is not None:
        mask = ((torch.abs(lap_gtr) < masking_thr) &
                (torch.abs(lap_net) < masking_thr))
    else:
        mask = torch.ones_like(lap_gtr) > 0
    loss = F.mse_loss(diff, torch.zeros_like(diff), reduction='none')
    if use_weights:
        loss = loss * weights
    loss = loss[mask].mean()
    if return_mask:
        return loss, mask
    else:
        return loss