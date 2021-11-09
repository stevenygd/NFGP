import torch
import torch.nn.functional as F
from trainers.utils.diff_ops import gradient, hessian, jacobian
from trainers.utils.igp_utils import sample_points_for_loss, \
    _addr_, tangential_projection_matrix

"""
Hessian based losses:
    -   the idea being that the Hessian of the SDF contains all information
        including the first and the second foundamental form
    -   supervising the Hessian should give us similar things as matching the
        first and the second foudamental form
"""

def transform_matrix(x, y, x_gtr):
    bs, npoints, dim = x.size(0), x.size(1), x.size(2)
    _, P_G = tangential_projection_matrix(y, x)
    JJ, JJ_status = jacobian(x_gtr, x)
    assert JJ_status == 0
    JJ = JJ.view(bs * npoints, dim, dim)
    P_G = P_G.view(bs * npoints, dim, dim)
    TT = torch.bmm(JJ, P_G)
    return TT, JJ, P_G


def field_hessian(y, x, hessian_type='direct', beta=1.,
                  tang_proj=False, extend_tang=False):
    if hessian_type == 'direct':
        H, status = hessian(y, x)
        assert status == 0
    elif hessian_type == 'norm':
        g = gradient(y, x)
        n = g / g.norm(dim=-1, keepdim=True)
        H, status = jacobian(n, x)
        assert status == 0
    else:
        raise NotImplemented

    if tang_proj:
        assert hessian_type != 'norm', "Hessian type == norm doesn't go with tang projection"
        bs, npoints, dim = x.size(0), x.size(1), x.size(2)
        xy_n, n_proj = tangential_projection_matrix(y, x)
        n_proj = n_proj.view(bs * npoints, dim, dim)
        H = torch.bmm(n_proj, torch.bmm(H.view(bs * npoints, dim, dim), n_proj))
        H = H.view(bs, npoints, dim, dim) * beta

        if extend_tang:
            H = _addr_(H, xy_n, xy_n)
    else:
        assert beta == 1, "Not implemented beta != 1 cases yet"
        H = H * beta

    return H


def hessian_match_loss(
        gtr, net, x=None, npoints=1000, dim=3, beta=1.,
        y_net=None, loss_type='l2', reduction='mean',
        use_surf_points=False, hessian_type='direct',
        deform=None, invert_sampling=False,
        tang_proj=False, extend_tang=False,
        weights=1., use_weights=False, use_log=False,
        quantile=None, use_bending=False,
        detach_weight=True, use_rejection=False,
        use_square=False,
):
    if x is None:
        x, weights = sample_points_for_loss(
            npoints, dim=dim, use_surf_points=use_surf_points,
            gtr=gtr, net=net, deform=deform, invert_sampling=invert_sampling,
            return_weight=True, detach_weight=detach_weight,
            use_rejection=use_rejection,
            use_square=use_square,
        )
        bs, npoints = x.size(0), x.size(1)
    else:
        if len(x.size()) == 2:
            bs, npoints = 1, x.size(0)
        else:
            bs, npoints = x.size(0), x.size(1)
    x = x.view(bs, npoints, dim)

    if use_bending:
        return bending_loss(
            gtr=gtr, net=net, x=x, beta=beta,
            y_net=y_net, loss_type=loss_type, reduction=reduction,
            use_surf_points=use_surf_points, hessian_type=hessian_type,
            deform=deform, invert_sampling=invert_sampling, weights=weights,
            use_weights=use_weights, use_log=use_log, quantile=quantile,
        )

    if y_net is None:
        if x.is_leaf:
            x.requires_grad = True
        y_net, delta_x, _ = net(x)

    h_net = field_hessian(
        y_net, x, hessian_type=hessian_type, beta=1.,
        tang_proj=tang_proj,
        extend_tang=extend_tang,
    )
    h_net = h_net.view(bs * npoints, dim, dim)

    x_gtr = x + delta_x
    delta_jac_x, delta_jac_x_status = jacobian(x_gtr, x)
    assert delta_jac_x_status == 0
    delta_jac_x = delta_jac_x.view(bs * npoints, dim, dim)
    delta_jac_x_T = delta_jac_x.transpose(1, 2).contiguous()

    x_gtr.retain_grad()
    y_gtr = gtr(x_gtr)
    h_gtr = field_hessian(
        y_gtr, x_gtr, hessian_type=hessian_type, beta=beta,
        tang_proj=tang_proj,
        extend_tang=extend_tang
    )
    # y-space
    h_gtr = h_gtr.view(bs * npoints, dim, dim)
    diff = torch.bmm(torch.bmm(delta_jac_x_T, h_gtr), delta_jac_x) - h_net

    # F_norm = (diff ** 2).view(bs * npoints, -1).sum(dim=-1, keepdim=False) ** 0.5
    F_norm = diff.view(bs * npoints, -1).norm(dim=-1, keepdim=False)
    F_norm = F_norm.view(bs, npoints)
    if use_log:
        F_norm = torch.log(F_norm)

    if quantile is not None:
        with torch.no_grad():
            q_F_norm = torch.quantile(
                F_norm.view(bs, npoints), torch.tensor([quantile]).to(x),
                dim=-1, keepdim=True)
        F_norm = torch.minimum(F_norm.view(bs, npoints), q_F_norm)

    if use_weights:
        F_norm = F_norm * weights

    if loss_type == 'l2':
        loss = F.mse_loss(
            F_norm, torch.zeros_like(F_norm), reduction=reduction)
    elif loss_type == 'l1':
        loss = F.l1_loss(
            F_norm, torch.zeros_like(F_norm), reduction=reduction)
    else:
        raise NotImplemented

    return loss



def bending_loss(
        gtr, net, x, beta=1.,
        y_net=None, loss_type='l2', reduction='mean',
        use_surf_points=False,
        hessian_type='direct',
        deform=None, invert_sampling=False,
        weights=1., use_weights=False,
        use_log=False, quantile=None,
):
    assert x is not None
    dim = x.size(-1)
    if len(x.size()) == 2:
        bs, npoints = 1, x.size(0)
    else:
        bs, npoints = x.size(0), x.size(1)
    x = x.view(bs, npoints, dim)

    if y_net is None:
        if x.is_leaf:
            x.requires_grad = True
        y_net, delta_x, _ = net(x)
    h_net = field_hessian(
        y_net, x, hessian_type=hessian_type, beta=1.,
        tang_proj=False, extend_tang=False,
    )
    h_net = h_net.view(bs * npoints, dim, dim)

    x_gtr = x + delta_x
    TT, JJ, PG = transform_matrix(x, y_net, x_gtr)

    x_gtr.retain_grad()
    y_gtr = gtr(x_gtr)
    h_gtr = field_hessian(
        y_gtr, x_gtr, hessian_type=hessian_type, beta=beta,
        tang_proj=False, extend_tang=False,
    )
    # y-space
    h_gtr = h_gtr.view(bs * npoints, dim, dim)
    diff = torch.bmm(TT.transpose(1, 2).contiguous(), torch.bmm(h_gtr, TT)) - \
           torch.bmm(PG.transpose(1, 2).contiguous(), torch.bmm(h_net, PG))

    # F_norm = (diff ** 2).view(bs * npoints, -1).sum(dim=-1, keepdim=False) ** 0.5
    F_norm = diff.view(bs * npoints, -1).norm(dim=-1, keepdim=False)
    F_norm = F_norm.view(bs, npoints)
    if use_log:
        F_norm = torch.log(F_norm)

    if quantile is not None:
        with torch.no_grad():
            q_F_norm = torch.quantile(
                F_norm.view(bs, npoints), torch.tensor([quantile]).to(x),
                dim=-1, keepdim=True)
        F_norm = torch.minimum(F_norm.view(bs, npoints), q_F_norm)

    if use_weights:
        F_norm = F_norm * weights

    if loss_type == 'l2':
        loss = F.mse_loss(
            F_norm, torch.zeros_like(F_norm), reduction=reduction)
    elif loss_type == 'l1':
        loss = F.l1_loss(
            F_norm, torch.zeros_like(F_norm), reduction=reduction)
    else:
        raise NotImplemented

    return loss


## Stretching loss
def get_cauchy_strain_tensor(x, y, deformed_x, deformed_y,
                             tang_proj=True, extend_tang=True):
    bs, npoints, dim = x.size(0), x.size(1), x.size(2)

    delta_jac_x, delta_jac_x_status = jacobian(deformed_x, x)
    assert delta_jac_x_status == 0
    delta_jac_x = delta_jac_x.view(bs * npoints, dim, dim)

    # Compute normal projection
    if tang_proj:
        # Projection matrix in the original/input
        dfm_xy_n, normals_proj_dfm = tangential_projection_matrix(
            deformed_y, deformed_x)
        normals_proj_dfm = normals_proj_dfm.view(bs * npoints, dim, dim)
    else:
        normals_proj_dfm = torch.eye(dim).to(x).view(
            1, dim, dim).expand(bs * npoints, dim, dim)

    proj_jac_x = torch.bmm(normals_proj_dfm, delta_jac_x)
    proj_jac_x = proj_jac_x.view(bs * npoints, dim, dim)

    # Projection matrix in the input space (i.e. deformed one)
    xy_n, normals_proj = tangential_projection_matrix(y, x)
    normals_proj = normals_proj.view(bs * npoints, dim, dim)
    if extend_tang:
        proj_jac_x = torch.bmm(proj_jac_x, normals_proj)
        proj_jac_x = _addr_(
            proj_jac_x.view(bs, npoints, dim, dim), dfm_xy_n, xy_n)
        proj_jac_x = proj_jac_x.view(bs * npoints, dim, dim)

        trg = torch.eye(dim).view(1, dim, dim).expand(bs * npoints, dim, dim).to(x)
    else:
        trg = torch.bmm(
            normals_proj.transpose(1, 2).contiguous(), normals_proj).to(x)

    cauchy_green_strain_tensor = torch.bmm(
        proj_jac_x.transpose(1, 2).contiguous(),
        proj_jac_x
    )

    return cauchy_green_strain_tensor.view(bs, npoints, dim, dim), \
           trg.view(bs, npoints, dim, dim)


def simple_stretch_loss(
        original, net, deform=None,
        x=None, npoints=1000, dim=3, use_surf_points=False, invert_sampling=False,
        alpha=1., beta=1., tang_proj=True, extend_tang=False,
        loss_type='area_length', reduction='mean', weights=1, use_weights=False,
        detach_weight=True, use_rejection=False, use_square=False,
):
    if x is None:
        x, weights = sample_points_for_loss(
            npoints, dim=dim, use_surf_points=use_surf_points,
            gtr=original, net=net, deform=deform,
            invert_sampling=invert_sampling, return_weight=True,
            detach_weight=detach_weight, use_rejection=use_rejection,
            use_square=use_square,
        )
        bs, npoints = x.size(0), x.size(1)
        weights = weights.view(bs, npoints)
    else:
        if len(x.size()) == 2:
            bs, npoints = 1, x.size(0)
        else:
            bs, npoints = x.size(0), x.size(1)
    x = x.view(bs, npoints, dim)

    if x.is_leaf:
        x.requires_grad = True
    x.retain_grad()
    y, delta_x, _ = net(x)
    x_deformed = x + delta_x
    x_deformed.retain_grad()
    y_deformed = original(x_deformed)

    if loss_type  in ['area_length']:
        cauchy_green_strain_tensor, base_mat = get_cauchy_strain_tensor(
            x, y, x_deformed, y_deformed,
            tang_proj=tang_proj, extend_tang=extend_tang)

        assert tang_proj and extend_tang
        area_distortion = torch.linalg.det(cauchy_green_strain_tensor)
        # area_distortion = area_distortion + 1. / area_distortion
        # Trace
        length_distortion = torch.diagonal(
            cauchy_green_strain_tensor, dim1=-2, dim2=-1
        ).sum(dim=-1, keepdim=False)

        if use_weights:
            area_distortion_loss = alpha * (area_distortion * weights).mean()
            length_distortion_loss = beta * (length_distortion * weights).mean()
        else:
            area_distortion_loss = alpha * area_distortion.mean()
            length_distortion_loss = beta * length_distortion.mean()
        loss = area_distortion_loss + length_distortion_loss

    elif loss_type == "id":
        cauchy_green_strain_tensor, base_mat = get_cauchy_strain_tensor(
            x, y, x_deformed, y_deformed,
            tang_proj=tang_proj, extend_tang=extend_tang)
        diff = base_mat - cauchy_green_strain_tensor
        diff_f_norm = diff.view(bs, npoints, dim * dim).norm(dim=-1)
        if use_weights:
            diff_f_norm = diff_f_norm * weights
        loss = F.mse_loss(diff_f_norm, torch.zeros_like(diff_f_norm),
                          reduction=reduction)
    elif loss_type == "simple":
        TT, _, P_G = transform_matrix(x, y, x_deformed)
        diff = P_G - torch.bmm(TT.transpose(1, 2).contiguous(), TT)
        diff_f_norm = diff.view(bs, npoints, dim * dim).norm(dim=-1)
        if use_weights:
            diff_f_norm = diff_f_norm * weights
        loss = F.mse_loss(diff_f_norm, torch.zeros_like(diff_f_norm),
                          reduction=reduction)
    else:
        raise NotImplementedError

    return loss
