import torch
import torch.nn.functional as F
from trainers.utils.diff_ops import laplace
from trainers.utils.igp_utils import get_surf_pcl, sample_points_for_loss, \
    mean_curvature


def loss_boundary(gtr, net, npoints=1000, dim=3, x=None, use_surf_points=None):
    """

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


def loss_mean_curvature_match(
        gtr, net, x=None, npoints=1000, dim=3, masking_thr=10, beta=1.,
        y_net=None, loss_type='l2', reduction='mean', eps=0., return_mask=False,
        diff_type='abs', diff_eps=1e-7, use_surf_points=False,
        invert_sampling=False, deform=None, weights=1, use_weight=False
):
    """
    The loss to match mean curvatures between shapes. Let [gtr] be the IMF for
    the ground truth shape and let [net] be the one to learn. The transformation
    from [gtr] to [net] is [D]. Then the loss is :

    ```1 / n * sum_{x_i = 1...n} |KM(net, x) - KM(gtr, D(x)) * beta | ```

    :param gtr:
    :param net:
    :param x:
    :param npoints:
    :param dim:
    :param masking_thr:
    :param beta:
    :param y_net:
    :param loss_type:
    :param reduction:
    :param eps:
    :param return_mask:
    :param diff_type:
    :param diff_eps:
    :param use_surf_points:
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

    if y_net is None:
        x.requires_grad = True
        y_net, delta, _ = net(x)
    km_net = mean_curvature(net, x, y=y_net, eps=eps)

    x_gtr = x + delta[..., :dim]
    y_gtr = gtr(x_gtr)
    km_gtr = mean_curvature(gtr, x_gtr, y=y_gtr, eps=eps) * beta

    if masking_thr is not None:
        mask = ((torch.abs(km_gtr) < masking_thr) &
                (torch.abs(km_net) < masking_thr))
    else:
        mask = torch.ones_like(km_gtr) > 0

    def _diff_(a, b):
        if diff_type == 'abs':
            return a - b
        elif diff_type == 'rel':
            return 2 * (a - b) / (torch.abs(a) + torch.abs(b) + diff_eps)
        else:
            raise NotImplemented

    if reduction is not 'none':
        if use_weight:
            diff = (_diff_(km_gtr, km_net) * weights)[mask]
        else:
            diff = _diff_(km_gtr, km_net)[mask]
        if loss_type == 'l2':
            loss = F.mse_loss(
                diff, torch.zeros_like(diff), reduction=reduction)
        elif loss_type == 'l1':
            loss = F.l1_loss(
                diff, torch.zeros_like(diff), reduction=reduction)
        else:
            raise NotImplemented
    else:
        diff = _diff_(km_gtr, km_net)
        if use_weight:
            diff = diff * weights
        if loss_type == 'l2':
            loss = F.mse_loss(
                diff, torch.zeros_like(diff), reduction='none')
        elif loss_type == 'l1':
            loss = F.l1_loss(
                diff, torch.zeros_like(diff), reduction='none')
        else:
            raise NotImplemented
        loss[torch.logical_not(mask)] = torch.zeros_like(
            loss[torch.logical_not(mask)])
    if return_mask:
        return loss, mask
    return loss



