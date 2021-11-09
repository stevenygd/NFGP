import torch
import trimesh
import numpy as np
import open3d as o3d
import torch.nn as nn
import torch.nn.functional as F
from trainers.utils.vis_utils import imf2mesh
from trainers.utils.igp_losses import loss_eikonal, loss_boundary, lap_loss, \
    mean_curvature_match_loss, laplacian_beltrami_match_loss, deform_prior_loss
from trainers.utils.igp_volume_loss import deform_ortho_jacobian, deform_det_jacobian
from trainers.utils.igp_shell_energy import simple_stretch_loss, hessian_match_loss


def trimesh_to_o3dmesh(mesh):
    o3d_mesh = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(np.array(mesh.vertices)),
        triangles=o3d.utility.Vector3iVector(np.array(mesh.faces))
    )
    o3d_mesh.compute_vertex_normals()
    return o3d_mesh


def o3dmesh_to_trimesh(mesh):
    mesh = trimesh.Trimesh(
        vertices=np.asarray(mesh.vertices).reshape(-1, 3).astype(np.float),
        faces=np.asarray(mesh.triangles).reshape(-1, 3).astype(np.int)
    )
    return mesh


def deform_mesh_o3d(imf, handles, targets, normalize=True, res=256,
                    imf_mesh=None, steps=50, smoothed_alpha=0.01, verbose=True):
    """
    Use Open3D to do deformation
    Args:
        [imf]
        [handles] (n, 3) Source points.
        [targets] (n, 3) Target points.
        [normalize] Whether normalize the mesh to unit sphere. Default (True).
        [res] Resolution for MC. Default (256).
    Returns:
    """
    if imf_mesh is None:
        mesh = imf2mesh(imf, res=res, threshold=0.00)

        if normalize:
            verts = (mesh.vertices * 2 - res) / float(res)
            mesh = trimesh.Trimesh(vertices=verts, faces=mesh.faces)
    else:
        mesh = imf_mesh

    vertices = np.asarray(mesh.vertices).reshape(-1, 3)
    vert_ids = []
    vert_pos = []
    for i in range(handles.reshape(-1, 3).shape[0]):
        dist = np.linalg.norm(
            vertices - handles[i, :].reshape(1, 3), axis=-1
        ).flatten()
        handle_idx = np.argmin(dist)
        vert_ids.append(handle_idx)
        vert_pos.append(
            vertices[handle_idx].reshape(3) + targets[i].reshape(3) -
            handles[i].reshape(3))

    constraint_ids = o3d.utility.IntVector(vert_ids)
    constraint_pos = o3d.utility.Vector3dVector(vert_pos)
    o3d_vert0 = o3d.utility.Vector3dVector(mesh.vertices)
    o3d_face0 = o3d.utility.Vector3iVector(mesh.faces)
    o3d_mesh0 = o3d.geometry.TriangleMesh(
        vertices=o3d_vert0, triangles=o3d_face0)
    o3d_mesh0.compute_vertex_normals()

    with o3d.utility.VerbosityContextManager(
            o3d.utility.VerbosityLevel.Debug) as cm:
        if smoothed_alpha > 0:
            mesh_deformed = o3d_mesh0.deform_as_rigid_as_possible(
                constraint_ids, constraint_pos, max_iter=steps,
                smoothed_alpha=smoothed_alpha,
                energy=o3d.geometry.DeformAsRigidAsPossibleEnergy.Smoothed)
        else:
            mesh_deformed = o3d_mesh0.deform_as_rigid_as_possible(
                constraint_ids, constraint_pos, max_iter=steps,
                smoothed_alpha=0,
                energy=o3d.geometry.DeformAsRigidAsPossibleEnergy.Spokes)

    return o3dmesh_to_trimesh(mesh_deformed)


def chamfer_distance(x, y):
    x = x.unsqueeze(1)
    y = y.unsqueeze(0)
    diff = (x - y).norm(dim=-1, keepdim=False)
    diff_x = diff.min(dim=-2, keepdim=False)[0]
    diff_y = diff.min(dim=-1, keepdim=False)[0]
    return (0.5 * (diff_x + diff_y)).mean()


def evaluate_mesh(imf, gtr_mesh, imf_mesh=None, npoints=2048, res=256,
                  threshold=0., dim=3, verbose=False, normalize=True):
    if imf_mesh is None:
        imf_mesh = imf2mesh(imf, res=res, threshold=threshold, verbose=verbose)
        if normalize:
            verts = (imf_mesh.vertices * 2 - res) / float(res)
            imf_mesh = trimesh.Trimesh(vertices=verts, faces=imf_mesh.faces)

    gtr_pcl = gtr_mesh.sample(npoints)
    imf_pcl = imf_mesh.sample(npoints)

    with torch.no_grad():
        x = torch.from_numpy(gtr_pcl).float().cuda().view(-1, dim)
        y = torch.from_numpy(imf_pcl).float().cuda().view(-1, dim)
        cd = chamfer_distance(x, y).detach().cpu().item()

    return { 'cd': cd, }


def deform_step(
        net, opt, original, handles_ts, targets_ts, dim=3,
        sample_cfg=None, x=None, weights=1,
        # Loss handle
        loss_h_weight=1., lag_mult=None, use_l1_loss=False, loss_h_thr=None,
        # Loss G
        loss_g_weight=1e-2, n_g_pts=5000,
        # Loss KM
        loss_km_weight=1e-4, n_km_pts=5000, km_mask_thr=5., km_diff_type='abs',
        km_use_surf_points=True, use_lapbal=False, km_invert_sample=True,
        # Loss orthogonality
        loss_orth_weight=0., n_orth_pts=5000, orth_reg_type='so',
        orth_use_surf_points=True, orth_invert_sample=True,
        # Loss determinant
        loss_det_weight=0., n_det_pts=5000,
        det_use_surf_points=True, det_invert_sample=True,
        det_detach_weight=False,

        # Loss hessian
        loss_hess_weight=0., n_hess_pts=5000, hess_use_surf_points=True,
        hess_invert_sample=True, hess_type='direct',
        hess_tang_proj=False, hess_tang_extend=False, hess_use_weight=False,
        hess_quantile=None, hess_use_bending=False, hess_detach_weight=True,
        hess_use_rejection=False, hess_use_square=False,

        # Loss stretch
        loss_stretch_weight=0., n_s_pts=5000, stretch_use_surf_points=True,
        stretch_invert_sample=True, stretch_alpha=0.5, stretch_beta=0.5,
        stretch_extend_tang=False, stretch_proj_tang=True,
        stretch_loss_type='area_length',
        stretch_use_weight=False, stretch_detach_weight=True,
        stretch_use_rejection=False,
        stretch_use_square=False,

        # Clip gradient
        grad_clip=None,
):

    opt.zero_grad()
    if loss_h_weight is not None:
        # x
        handles_ts = handles_ts.clone().detach().float().cuda()
        # y
        targets_ts = targets_ts.clone().detach().float().cuda()
        constr = (
                net(targets_ts, None, return_delta=True)[0] + targets_ts - handles_ts
        ).view(-1, dim).norm(dim=-1, keepdim=False)
        if loss_h_thr is not None:
            loss_h_thr = float(loss_h_thr)
            constr = F.relu(constr - loss_h_thr)
        if lag_mult is not None:
            n_constr = constr.size(0)
            if isinstance(lag_mult, float):
                lag_mult = torch.ones(n_constr) * lag_mult
            lag_mult = lag_mult.view(n_constr).to(handles_ts)

            # Lagrangian multiplier term
            lag_mult_orig = (lag_mult * constr).mean()
            # Augmentation term
            lag_mult_aug = (loss_h_weight * 0.5 * constr ** 2).mean()

            loss_h = lag_mult_orig + lag_mult_aug
        else:
            if use_l1_loss:
                loss_h = F.l1_loss(
                    constr, torch.zeros_like(constr)) * loss_h_weight
            else:
                loss_h = F.mse_loss(
                    constr, torch.zeros_like(constr)) * loss_h_weight
    else:
        assert lag_mult is None
        loss_h = torch.zeros(1).cuda().float()
        constr = 0

    if sample_cfg is not None and x is None:
        npoints = getattr(sample_cfg, "num_points", 5000)
        invert_sampling = getattr(sample_cfg, "invert_sample", True)
        use_surf_points = getattr(sample_cfg, "use_surf_points", True)
        x, weights = sample_points_for_loss(
            npoints, dim=dim, use_surf_points=use_surf_points,
            gtr=(lambda x: original(x, None)),
            net=(lambda x: net(x, None)),
            deform=net.deform, invert_sampling=invert_sampling,
            return_weight=True,
            detach_weight=getattr(sample_cfg, "detach_weight", True),
            use_rejection=getattr(sample_cfg, "use_rejection", False),
            use_square=getattr(sample_cfg, "use_square", False)
        )
    else:
        # NOTE: defined in the arguement!
        pass

    if loss_g_weight > 0.:
        loss_g = loss_eikonal(
            lambda x: net(x, None), npoints=n_g_pts, dim=dim) * loss_g_weight
    else:
        loss_g = torch.zeros(1).cuda().float()

    if loss_orth_weight > 0.:
        loss_orth = deform_ortho_jacobian(
            gtr=lambda x: original(x, None),
            net=lambda x: net(x, None),
            deform_net=net.deform,
            npoints=n_orth_pts, dim=dim, ortho_reg_type=orth_reg_type,
            use_surf_points=orth_use_surf_points,
            invert_sampling=orth_invert_sample,
            x=x, weights=weights
        )
        loss_orth = loss_orth * loss_orth_weight
    else:
        loss_orth = torch.zeros(1).cuda().float()

    if loss_det_weight > 0.:
        loss_det = deform_det_jacobian(
            gtr=lambda x: original(x, None),
            net=lambda x: net(x, None),
            deform_net=net.deform,
            npoints=n_det_pts, dim=dim,
            use_surf_points=det_use_surf_points,
            invert_sampling=det_invert_sample,
            x=x, weights=weights, detach_weight=det_detach_weight
        )
        loss_det = loss_det * loss_det_weight
    else:
        loss_det = torch.zeros(1).cuda().float()

    if loss_km_weight > 0.:
        if use_lapbal:
            loss_km, loss_km_mask = laplacian_beltrami_match_loss(
                lambda x: original(x, None),
                lambda x: net(x, None, return_both=True),
                masking_thr=km_mask_thr, dim=dim, npoints=n_km_pts,
                return_mask=True, use_surf_points=km_use_surf_points,
                invert_sampling=km_invert_sample,
                deform=net.deform,
                x=x, weights=weights
            )
        else:
            loss_km, loss_km_mask = mean_curvature_match_loss(
                lambda x: original(x, None),
                lambda x: net(x, None, return_both=True),
                masking_thr=km_mask_thr, dim=dim, npoints=n_km_pts,
                return_mask=True, diff_type=km_diff_type,
                use_surf_points=km_use_surf_points,
                invert_sampling=km_invert_sample,
                deform=net.deform,
                x=x, weights=weights
            )
        loss_km *= loss_km_weight
        loss_km_mask_perc = loss_km_mask.float().mean() * 100.
    else:
        loss_km = torch.zeros(1).cuda().float()
        loss_km_mask_perc = torch.zeros(1).cuda().float()

    if loss_hess_weight > 0.:
        loss_hess = hessian_match_loss(
            lambda x: original(x, None),
            lambda x: net(x, None, return_both=True),
            dim=dim, npoints=n_hess_pts,
            use_surf_points=hess_use_surf_points,
            invert_sampling=hess_invert_sample,
            deform=net.deform,
            hessian_type=hess_type,
            extend_tang=hess_tang_extend,
            tang_proj=hess_tang_proj,
            use_weights=hess_use_weight,
            quantile=hess_quantile,
            x=x, weights=weights,
            use_bending=hess_use_bending,
            detach_weight=hess_detach_weight,
            use_rejection=hess_use_rejection,
            use_square=hess_use_square,
        )
        loss_hess *= loss_hess_weight
    else:
        loss_hess = torch.zeros(1).cuda().float()

    if loss_stretch_weight > 0.:
        loss_stretch = simple_stretch_loss(
            lambda x: original(x, None),
            lambda x: net(x, None, return_both=True),
            deform=net.deform,
            npoints=n_s_pts, dim=dim,
            use_surf_points=stretch_use_surf_points,
            invert_sampling=stretch_invert_sample,
            alpha=stretch_alpha,
            beta=stretch_beta,
            extend_tang=stretch_extend_tang,
            tang_proj=stretch_proj_tang,
            loss_type=stretch_loss_type,
            use_weights=stretch_use_weight,
            x=x, weights=weights,
            detach_weight=stretch_detach_weight,
            use_rejection=stretch_use_rejection,
            use_square=stretch_use_square,
        )
        loss_stretch *= loss_stretch_weight
    else:
        loss_stretch = torch.zeros(1).cuda().float()

    loss = loss_h + loss_g + loss_km + loss_hess + loss_stretch + \
           loss_det + loss_orth
    loss.backward()
    if grad_clip is not None:
        torch.nn.utils.clip_grad_norm_(net.deform.parameters(), grad_clip)

    opt.step()
    with torch.no_grad():
        if lag_mult is not None:
            assert constr is not None
            lag_mult += loss_h_weight * constr

    return {
        'loss': loss.detach().cpu().item(),
        'loss_h': loss_h.detach().cpu().item(),
        'lag_mult': lag_mult.detach().cpu() if lag_mult is not None else None,
        # Repairing
        'loss_g': loss_g.detach().cpu().item(),
        # Curvature (useless)
        'loss_km': loss_km.detach().cpu().item(),
        'loss_km_mask_perc': loss_km_mask_perc.detach().cpu().item(),
        # Volume
        'loss_orth': loss_orth.detach().cpu().item(),
        'loss_det': loss_det.detach().cpu().item(),
        # Shell energy
        'loss_hess': loss_hess.detach().cpu().item(),
        'loss_stretch': loss_stretch.detach().cpu().item()
    }


def filtering_step(
        beta, net, opt, original, deform=None, dim=3,
        # Sampling configuration
        sample_cfg=None,
        loss_boundary_cfg=None,
        loss_grad_cfg=None,
        loss_deform_prior_cfg=None,
        loss_lap_cfg=None,
        loss_km_cfg=None,
        loss_hess_cfg=None,
        # Clip gradient
        grad_clip=None,
        x = None, weights=1.
):
    opt.zero_grad()
    if sample_cfg is not None and x is None:
        npoints = getattr(sample_cfg, "num_points", 5000)
        invert_sampling = getattr(sample_cfg, "invert_sample", True)
        use_surf_points = getattr(sample_cfg, "use_surf_points", True)
        x, weights = sample_points_for_loss(
            npoints, dim=dim, use_surf_points=use_surf_points,
            gtr=(lambda x: original(x, None)),
            net=(lambda x: net(x, None)),
            deform=deform, invert_sampling=invert_sampling,
            return_weight=True)

    boundary_loss_weight = float(getattr(loss_boundary_cfg, "weight", 0))
    boundary_loss_num_points = int(getattr(loss_boundary_cfg, "num_points", 0))
    if boundary_loss_weight > 0.:
        loss_y_boundary, _ = loss_boundary(
            gtr=(lambda x: original(x, None)),
            net=(lambda x: net(x, None)),
            dim=dim,
            deform=deform,
            npoints=boundary_loss_num_points,
            invert_sampling=getattr(loss_boundary_cfg, "invert_sample", True),
            use_weights=getattr(loss_boundary_cfg, "use_weights", True),
            x=x, weights=weights
        )
        loss_y_boundary = loss_y_boundary * boundary_loss_weight
    else:
        loss_y_boundary = torch.zeros(1).float().cuda()

    deform_loss_weight = float(getattr(loss_deform_prior_cfg, "weight", 0.))
    deform_loss_num_points = int(getattr(loss_deform_prior_cfg, "num_points", 0))
    if deform_loss_weight > 0.:
        loss_deform = deform_prior_loss(
            gtr=(lambda x: original(x, None)),
            net=(lambda x: net(x, None)),
            deform=deform, dim=dim,
            npoints=deform_loss_num_points,
            use_surf_points=getattr(loss_deform_prior_cfg, "use_surf_points", True),
            invert_sampling=getattr(loss_deform_prior_cfg, "invert_sample", True),
            use_weights=getattr(loss_deform_prior_cfg, "use_weights", True),
            x=x, weights=weights
        )
        loss_deform = loss_deform * deform_loss_weight
    else:
        loss_deform = torch.zeros(1).float().cuda()

    grad_norm_weight = float(getattr(loss_grad_cfg, "weight", 0))
    grad_norm_num_points = int(getattr(loss_grad_cfg, "num_points", 0))
    if grad_norm_weight > 0.:
        loss_unit_grad_norm = loss_eikonal(
            net=(lambda x: net(x, None)),
            gtr=(lambda x: original(x, None)),
            deform=deform, dim=dim,
            npoints=grad_norm_num_points,
            use_surf_points=getattr(loss_grad_cfg, "use_surf_points", True),
            invert_sampling=getattr(loss_grad_cfg, "invert_sample", True),
            x=x, weights=weights
        )
        loss_unit_grad_norm *= grad_norm_weight
    else:
        loss_unit_grad_norm = torch.zeros(1).float().cuda()

    lap_loss_weight = float(getattr(loss_lap_cfg, "weight", 0))
    lap_loss_num_points = int(getattr(loss_lap_cfg, "num_points", 0))
    if lap_loss_weight > 0.:
        loss_lap_scaling, lap_mask = lap_loss(
            gtr=(lambda x: original(x, None)),
            net=(lambda x: net(x, None)),
            deform=deform,
            npoints=lap_loss_num_points,
            beta=beta,
            masking_thr=getattr(loss_lap_cfg, "threshold", 50),
            use_surf_points=getattr(loss_lap_cfg, "use_surf_points", True),
            invert_sampling=getattr(loss_lap_cfg, "invert_sampling", True),
            use_weights=getattr(loss_lap_cfg, "use_weights", True),
            return_mask=True,
            x=x, weights=weights
        )
        loss_lap_scaling = loss_lap_scaling * lap_loss_weight
        loss_lap_mask_perc = lap_mask.float().mean()
    else:
        loss_lap_scaling = torch.zeros(1).float().cuda()
        loss_lap_mask_perc = torch.zeros(1).float().cuda()

    km_loss_weight = float(getattr(loss_km_cfg, "weight", 0.))
    km_loss_num_points = int(getattr(loss_km_cfg, "num_points", 0))
    if km_loss_weight > 0.:
        loss_km, loss_km_mask = mean_curvature_match_loss(
            gtr=(lambda x: original(x, None)),
            net=(lambda x: net(x, None, return_both=True)),
            beta=beta,
            npoints=km_loss_num_points, dim=dim,
            masking_thr=getattr(loss_km_cfg, "threshold", 50),
            loss_type='l2', eps=0.,
            return_mask=True,
            diff_type=getattr(loss_km_cfg, "diff_type", 'rel'),
            use_surf_points=getattr(loss_km_cfg, "use_surf_points", False),
            invert_sampling=getattr(loss_km_cfg, "invert_sample", True),
            use_weight=getattr(loss_km_cfg, "use_weights", True),
            x=x, weights=weights
        )
        loss_km *= km_loss_weight
        loss_km_mask_perc = loss_km_mask.float().mean()
    else:
        loss_km = torch.zeros(1).float().cuda()
        loss_km_mask_perc = torch.zeros(1).float().cuda()

    loss_hess_weight = getattr(loss_hess_cfg, "weight", 0.)
    if loss_hess_weight > 0.:
        loss_hess = hessian_match_loss(
            lambda x: original(x, None),
            lambda x: net(x, None, return_both=True),
            beta=beta,
            dim=dim, deform=net.deform,
            npoints=getattr(loss_hess_cfg, "num_points", 5000),
            use_surf_points=getattr(loss_hess_cfg, "use_surf_points", True),
            invert_sampling=getattr(loss_hess_cfg, "invert_sample", True),
            hessian_type=getattr(loss_hess_cfg, "hess_type", "norm"),
            extend_tang=getattr(loss_hess_cfg, "extend_tang", False),
            tang_proj=getattr(loss_hess_cfg, "tang_proj", True),
            use_weights=getattr(loss_hess_cfg, "use_weights", True),
            x=x, weights=weights
        )
        loss_hess *= loss_hess_weight
    else:
        loss_hess = torch.zeros(1).cuda().float()

    loss = loss_unit_grad_norm + loss_y_boundary + loss_lap_scaling + \
           loss_deform + loss_km + loss_hess
    loss.backward()
    if grad_clip is not None:
        torch.nn.utils.clip_grad_norm_(net.parameters(), grad_clip)
    opt.step()

    return {
        'loss': loss.detach().cpu().item(),
        'loss/loss_hess': loss_hess.detach().cpu().item(),
        'loss/loss_boundary': loss_y_boundary.detach().cpu().item(),
        'loss/loss_eikonal': loss_unit_grad_norm.detach().cpu().item(),
        'loss/loss_lap_scaling': loss_lap_scaling.detach().cpu().item(),
        'loss/loss_deform': loss_deform.detach().cpu().item(),
        'loss/loss_km': loss_km.detach().cpu().item(),
        'weight/loss_boundary': boundary_loss_weight,
        'weight/loss_eikonal': grad_norm_weight,
        'weight/loss_lap': lap_loss_weight,
        'weight/loss_deform': deform_loss_weight,
        'weight/loss_km': km_loss_weight,
        'weight/loss_km_mask_perc': loss_km_mask_perc.detach().cpu().item(),
        'weight/loss_lap_mask_perc': loss_lap_mask_perc.detach().cpu().item()
    }

