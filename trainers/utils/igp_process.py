import torch
import trimesh
import numpy as np
import open3d as o3d
import torch.nn as nn
import torch.nn.functional as F
from trainers.utils.vis_utils import imf2mesh
from trainers.losses.eikonal_loss import loss_eikonal
from trainers.utils.igp_utils import sample_points_for_loss
from trainers.losses.implicit_thin_shell_losses import simple_stretch_loss, \
    hessian_match_loss


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


def deform_step(
        net, opt, original, handles_ts, targets_ts, dim=3,
        sample_cfg=None, x=None, weights=1,
        # Loss handle
        loss_h_weight=1., use_l1_loss=False, loss_h_thr=None,
        # Loss G
        loss_g_weight=1e-2, n_g_pts=5000,

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
            net(targets_ts, return_delta=True)[0] + targets_ts - handles_ts
        ).view(-1, dim).norm(dim=-1, keepdim=False)
        if loss_h_thr is not None:
            loss_h_thr = float(loss_h_thr)
            constr = F.relu(constr - loss_h_thr)
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
            gtr=(lambda x: original(x)),
            net=(lambda x: net(x)),
            deform=net.deform, invert_sampling=invert_sampling,
            return_weight=True,
            detach_weight=getattr(sample_cfg, "detach_weight", True),
            use_rejection=getattr(sample_cfg, "use_rejection", False),
        )
    else:
        # NOTE: defined in the arguement!
        pass

    if loss_g_weight > 0.:
        loss_g = loss_eikonal(
            lambda x: net(x), npoints=n_g_pts, dim=dim) * loss_g_weight
    else:
        loss_g = torch.zeros(1).cuda().float()

    if loss_hess_weight > 0.:
        loss_hess = hessian_match_loss(
            lambda x: original(x),
            lambda x: net(x, return_both=True),
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
            lambda x: original(x),
            lambda x: net(x, return_both=True),
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

    loss = loss_h + loss_g + loss_hess + loss_stretch
    loss.backward()
    if grad_clip is not None:
        torch.nn.utils.clip_grad_norm_(net.deform.parameters(), grad_clip)

    opt.step()

    return {
        'loss': loss.detach().cpu().item(),
        'loss_h': loss_h.detach().cpu().item(),
        # Repairing
        'loss_g': loss_g.detach().cpu().item(),
        # Shell energy
        'loss_hess': loss_hess.detach().cpu().item(),
        'loss_stretch': loss_stretch.detach().cpu().item()
    }
