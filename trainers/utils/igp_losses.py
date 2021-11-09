import torch
import torch.nn.functional as F
from trainers.utils.diff_ops import laplace, gradient, jacobian
from trainers.utils.igp_utils import get_surf_pcl, sample_points_for_loss, \
    mean_curvature


def loss_boundary(gtr, net, npoints=1000, invert_sampling=True,
                  dim=3, x=None, use_surf_points=None):
    assert use_surf_points is not None
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


def lap_loss(
        gtr, net, deform=None,
        x=None, npoints=1000, dim=3, use_surf_points=True, invert_sampling=True,
        beta=1., masking_thr=10,
        return_mask=False,
        use_weights=False, weights=1
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


def deform_prior_loss(
        gtr, net, deform=None,
        x=None, dim=3, npoints=1000, use_surf_points=True, invert_sampling=True,
        loss_type='l2', reduction='mean',
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

    x = x.clone()
    y = deform(x, None)
    diff = (y - x).norm(dim=-1, keepdim=False)
    if use_weights:
        loss = diff * weights
    if loss_type == 'l2':
        loss = F.mse_loss(diff, torch.zeros_like(diff), reduction=reduction)
    elif loss_type == 'l1':
        loss = F.l1_loss(diff, torch.zeros_like(diff), reduction=reduction)
    else:
        raise NotImplementedError
    return loss


def deform_volpresv_loss(deform_net, x=None, dim=3, npoints=1000,
                         loss_type='l2', reduction='mean', use_log=False):
    if x is None:
        assert npoints is not None
        x = torch.rand(1, npoints, dim).cuda().float() * 2 - 1
    else:
        npoints = x.size(1)

    bs = x.size(0)
    x = x.clone()
    x.requires_grad = True
    delta = deform_net(x)
    jac_delta, _ = jacobian(delta, x)
    jac_delta = jac_delta.view(bs, npoints, dim, dim)
    jac_identity = torch.eye(dim).view(1, 1, dim, dim).to(jac_delta)
    jac_delta = jac_identity + jac_delta
    det_jac_delta = torch.linalg.det(jac_delta.view(-1, dim, dim))

    det_jac_delta_abs = torch.abs(det_jac_delta)
    if use_log:
        det_jac_delta_abs = torch.abs(torch.log(det_jac_delta_abs))

    if loss_type.lower() == 'l2':
        loss = F.mse_loss(
            det_jac_delta_abs, torch.ones_like(det_jac_delta),
            reduction=reduction)
    elif loss_type.lower() == 'l1':
        loss = F.l1_loss(
            det_jac_delta_abs, torch.ones_like(det_jac_delta),
            reduction=reduction)
    else:
        raise NotImplemented
    return loss


def mean_curvature_match_loss(
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



"""
Laplacian based loss:

Different from mean_curvature_match_loss, it's a match of two vectors
"""
def laplacian_beltrami_match_loss(
        gtr, net, x=None, npoints=1000, dim=3, masking_thr=10, beta=1.,
        y_net=None, loss_type='l2', reduction='mean', eps=0., return_mask=False,
        use_surf_points=False, invert_sampling=False, deform=None, weights=1,
        use_weights=False
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

    if y_net is None:
        x.requires_grad = True
        x.retain_grad()
        y_net, delta_x, _ = net(x)
    km_net, grad_net = mean_curvature(
        net, x, y=y_net, eps=eps, return_grad=True)
    lap_bal_net = (km_net * grad_net).view(bs * npoints, dim, 1)

    x_gtr = x + delta_x
    delta_jac_x, delta_jac_x_status = jacobian(x_gtr, x)
    assert delta_jac_x_status == 0
    delta_jac_x = delta_jac_x.view(bs * npoints, dim, dim)

    lap_bal_net_transf = torch.bmm(delta_jac_x, lap_bal_net) # (bs x npoints, dim, 1)
    lap_bal_net_transf = lap_bal_net_transf.view(bs * npoints, dim)

    y_gtr = gtr(x_gtr)
    km_gtr, grad_gtr = mean_curvature(
        gtr, x_gtr, y=y_gtr, eps=eps, return_grad=True)
    lap_bal_gtr = (km_gtr * grad_gtr * beta).view(bs * npoints, dim)

    if masking_thr is not None:
        mask = ((torch.abs(km_gtr) < masking_thr) &
                (torch.abs(km_net) < masking_thr)).view(bs * npoints)
    else:
        mask = torch.ones_like(km_gtr) > 0

    def _diff_(a, b):
        return (a - b).norm(dim=-1, keepdim=False)

    if reduction is not 'none':
        if use_weights:
            diff = (_diff_(lap_bal_gtr, lap_bal_net_transf) * weights)[mask]
        else:
            diff = _diff_(lap_bal_gtr, lap_bal_net_transf)[mask]
        if loss_type == 'l2':
            loss = F.mse_loss(
                diff, torch.zeros_like(diff), reduction=reduction)
        elif loss_type == 'l1':
            loss = F.l1_loss(
                diff, torch.zeros_like(diff), reduction=reduction)
        else:
            raise NotImplemented
    else:
        diff = _diff_(lap_bal_gtr, lap_bal_net_transf)
        if use_weights:
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


def gaussian_curvature_match_loss(
        gtr, net, deform=None,
        x=None, npoints=1000, dim=3, use_surf_points=True, invert_sampling=True,
        masking_thr=10, y_gtr=None, y_net=None, loss_type='l2', reduction='mean'
):
    raise NotImplementedError
    # if x is None:
    #     x, weights = sample_points_for_loss(
    #         npoints, dim=dim, use_surf_points=use_surf_points,
    #         gtr=gtr, net=net, deform=deform, invert_sampling=invert_sampling,
    #         return_weight=True
    #     )
    #     bs, npoints = x.size(0), x.size(1)
    # else:
    #     if len(x.size()) == 2:
    #         bs, npoints = 1, x.size(0)
    #     else:
    #         bs, npoints = x.size(0), x.size(1)
    # x = x.view(bs, npoints, dim)

    # if y_net is None:
    #     x.requires_grad = True
    #     y_net, delta = net(x)
    # kg_net = gaussian_curvature(net, x, y=y_net, dim=dim)

    # if y_gtr is None:
    #     x_gtr = (x + delta[..., :dim]).clone().detach()
    #     x_gtr.requires_grad = True
    #     y_gtr = gtr(x_gtr)
    # kg_gtr = gaussian_curvature(gtr, x, y=y_gtr, dim=dim).detach()

    # if masking_thr is not None:
    #     mask = ((torch.abs(kg_gtr) < masking_thr) &
    #             (torch.abs(kg_net) < masking_thr))
    # else:
    #     mask = torch.ones_like(kg_gtr) > 0
    # # diff = (kg_gtr - kg_net)[mask]
    # # if loss_type == 'l2':
    # #     loss = F.mse_loss(diff, torch.zeros_like(diff), reduction=reduction)
    # # elif loss_type == 'l1':
    # #     loss = F.l1_loss(diff, torch.zeros_like(diff), reduction=reduction)
    # # else:
    # #     raise NotImplemented
    # # return loss

    # if reduction is not 'none':
    #     diff = (kg_gtr - kg_net)[mask]
    #     if loss_type == 'l2':
    #         loss = F.mse_loss(
    #             diff, torch.zeros_like(diff), reduction=reduction)
    #     elif loss_type == 'l1':
    #         loss = F.l1_loss(
    #             diff, torch.zeros_like(diff), reduction=reduction)
    #     else:
    #         raise NotImplemented
    # else:
    #     diff = kg_gtr - kg_net
    #     if loss_type == 'l2':
    #         loss = F.mse_loss(
    #             diff, torch.zeros_like(diff), reduction='none')
    #     elif loss_type == 'l1':
    #         loss = F.l1_loss(
    #             diff, torch.zeros_like(diff), reduction='none')
    #     else:
    #         raise NotImplemented
    #     loss[torch.logical_not(mask)] = torch.zeros_like(
    #         loss[torch.logical_not(mask)])
    # return loss


def deform_relu_mlp_neighbor_cell_regularization(
        net, x=None, npoints=10000, dim=3, matmul=True,
        loss_type='l2', reduction='mean'
):
    raise NotImplementedError
    # if x is None:
    #     x = (torch.rand(npoints, dim) * 2 - 1).cuda()
    # else:
    #     npoints = x.size(0)

    # I = torch.eye(dim).view(1, dim, dim).to(x).expand(npoints, dim, dim)
    # W_init = torch.eye(dim).view(
    #     1, dim, dim).expand(npoints, dim, dim).to(x)
    # B_init = torch.zeros(npoints, dim, 1).to(x)
    # y, R = net(x, None, return_state=True)
    # # Get the constraints, and output J at R
    # J, _, W_cons, B_cons = net.forward_linear_operator(
    #     W_init, B_init, states=R, return_constr=True)
    # with torch.no_grad():
    #     W_c = torch.cat(
    #         [wc for Wc in W_cons for wc in Wc], dim=-1
    #     ).view(npoints, -1, dim)
    #     B_c = torch.cat(
    #         [bc for Bc in B_cons for bc in Bc], dim=-1
    #     ).view(npoints, W_c.size(1), 1)
    #     dist = torch.abs(torch.bmm(W_c, x.view(npoints, dim, 1)) + B_c)
    #     min_dist_idx = dist.view(npoints, -1).min(dim=-1, keepdim=False)[1]
    #     min_dist_idx = min_dist_idx.view(npoints).detach()
    #     R_flipped, _ = net.get_neighbor_states(R, min_dist_idx)

    # # Now that we got R_flipped, compute J at R_flipped
    # J_flipped = net.forward_linear_operator(
    #     W_init, B_init, states=R_flipped, return_constr=False)[0]

    # if matmul:
    #     # Compute loss, |J(T)^TJ_f(T) - I|_F^2,
    #     # and since J(T) and J_f(T) are symmetrical,
    #     # this amounts to |J(T)J_f(T) - I|_F^ 2
    #     # NOTE: we use J(T) = J + I and J_f(T) = J_f + I here since we refers
    #     #       to the jacobian of the whole transformation not just the deformation
    #     diff = torch.bmm(J_flipped + I, J + I) - I
    # else:
    #     diff = J_flipped - J

    # if reduction == 'none':
    #     if loss_type == 'l2':
    #         loss = (diff ** 2 ).view(npoints, -1).sum(dim=-1, keepdim=False)
    #     elif loss_type == 'l1':
    #         loss = torch.abs(diff).view(npoints, -1).sum(dim=-1, keepdim=False)
    #     else:
    #         raise NotImplemented
    #     return loss

    # if loss_type == 'l2':
    #     loss = F.mse_loss(
    #         diff, torch.zeros_like(diff), reduction=reduction)
    # elif loss_type == 'l1':
    #     loss = F.l1_loss(
    #         diff, torch.zeros_like(diff), reduction=reduction)
    # else:
    #     raise NotImplemented
    # return loss

def deform_smoothness_loss(
        deform_net, x=None, dim=3, npoints=1000,
        loss_type='l2', reduction='mean',
        with_identity=False
):
    # TODO: deprecatd
    raise NotImplementedError
    # if x is None:
    #     assert npoints is not None
    #     x = torch.rand(1, npoints, dim).cuda().float() * 2 - 1

    # x = x.clone()
    # x.requires_grad = True
    # delta = deform_net(x)
    # if with_identity:
    #     delta = delta + x
    # div_delta = divergence(delta, x)
    # if loss_type.lower() == 'l2':
    #     loss = F.mse_loss(
    #         div_delta, torch.zeros_like(div_delta), reduction=reduction)
    # elif loss_type.lower() == 'l1':
    #     loss = F.l1_loss(
    #         div_delta, torch.zeros_like(div_delta), reduction=reduction)
    # else:
    #     raise NotImplemented
    # return loss


def conv_loss(
        gtr, net, sigma, xy=None, npoints=1000, nconv_pts=100,
        dim=3, sharpen=False, sharpen_ratio=0.
):
    raise NotImplementedError
    # with torch.no_grad():
    #     if xy is None:
    #         assert npoints is not None
    #         xy = torch.rand(npoints, dim).cuda().float() * 2 - 1
    #     y1 = gaussian_conv(
    #         gtr, xy.view(1, -1, dim), sigma, n_points=nconv_pts).view(-1)
    #     if sharpen:
    #         out = gtr(xy.view(1, -1, dim)).view(-1)
    #         y1 = out + sharpen_ratio * (out - y1)
    # y2 = net(xy.view(1, -1, dim)).view(y1.size())
    # return F.mse_loss(y1, y2)


