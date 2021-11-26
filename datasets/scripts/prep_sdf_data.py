import os
import time
import trimesh
import argparse
import numpy as np
import os.path as osp
from mesh_to_sdf import mesh_to_sdf


# command line args
def get_args():
    parser = argparse.ArgumentParser(
        description='Process a mesh and dump SDF data to a directory.')
    parser.add_argument('mesh_path', type=str, help='The mesh file.')
    parser.add_argument('--out_path', type=str, default=None,
                        help='The output directory.')
    parser.add_argument('--num_uniform_points', type=int, default=5000000,
                        help='Number of point samples uniformly.')
    parser.add_argument('--num_nearsurface_points', type=int, default=5000000,
                        help='Number of point samples near surface.')
    parser.add_argument('--nearsurface_sigma', type=float, default=0.1,
                        help='STD for the Gaussian noise added near surface.')
    parser.add_argument('--save_uniform_data',
                        default=False, action='store_true')
    parser.add_argument('--save_nearsurface_data',
                        default=False, action='store_true')
    return parser.parse_args()


def load_mesh(mesh_path):
    """
    Args:
        [mesh_path]: path to the mesh
    Convert a possible scene to a mesh.

    If conversion occurs, the returned mesh has only vertex and face data.
    """
    scene_or_mesh = trimesh.load(mesh_path, force='mesh')
    if isinstance(scene_or_mesh, trimesh.Scene):
        assert len(scene_or_mesh.geometry) > 0
        # we lose texture information here
        mesh = trimesh.util.concatenate(
            tuple(trimesh.Trimesh(vertices=g.vertices, faces=g.faces)
                  for g in scene_or_mesh.geometry.values()))
    else:
        assert(isinstance(scene_or_mesh, trimesh.Trimesh))
        mesh = scene_or_mesh
    print("Mesh:", mesh)

    # Normalize
    verts = mesh.vertices.reshape(-1, 3)
    verts = verts - verts.mean(axis=0).reshape(1, 3)
    verts = verts / np.linalg.norm(verts, axis=-1).max() * 0.9
    print("Normalized vertices",
          "\n\tMax:", verts.max(),
          "\n\tMin:", verts.min(),
          "\n\tAvg:", verts.mean())
    mesh = trimesh.Trimesh(vertices=verts, faces=mesh.faces)
    return mesh


if __name__ == '__main__':
    args = get_args()
    mesh = load_mesh(args.mesh_path)
    if args.out_path is None:
        out_path = args.mesh_path[:-4] + '.npy'
        out_path_uniform = args.mesh_path[:-4] + '_uniform.npy'
        out_path_nearsurface = args.mesh_path[:-4] + '_nearsurface.npy'
    else:
        os.makedirs(args.out_path, exist_ok=True)
        out_path = osp.join(args.out_path, 'sdf.npy')
        out_path_uniform = osp.join(args.out_path, 'sdf_uniform.npy')
        out_path_nearsurface = osp.join(args.out_path, 'sdf_nearsurface.npy')
        mesh.export(osp.join(args.out_path, "mesh.obj"))

    print("Computing SDF for the uniformly sampled points.")
    print("\tnum_points:", args.num_uniform_points)
    uniform_points = np.random.uniform(
        size=(args.num_uniform_points, 3)) * 2 - 1
    s = time.time()
    uniform_sdf = mesh_to_sdf(mesh, uniform_points, sign_method='depth')
    e = time.time()
    print("Duration", e - s)

    if args.save_uniform_data:
        sdf_dataset = {
            'points': uniform_points,
            'sdf': uniform_sdf,
            "mesh": mesh
        }
        np.save(out_path_uniform, sdf_dataset)
        print("Uniform samples saved to :", out_path_uniform)

    print("Computing SDF for the near surface sampled points.")
    print("\t", "num_points:", args.num_nearsurface_points,
          "sig:", args.nearsurface_sigma)
    s = time.time()
    near_surface_points = mesh.sample(args.num_nearsurface_points)
    near_surface_points = near_surface_points + np.random.randn(
        *near_surface_points.shape) * args.nearsurface_sigma
    near_surface_sdf = mesh_to_sdf(
        mesh, near_surface_points, sign_method='depth')
    e = time.time()
    print("Duration", e - s)

    if args.save_nearsurface_data:
        sdf_dataset = {
            'points': near_surface_points,
            'sdf': near_surface_sdf,
            "mesh": mesh
        }
        np.save(out_path_nearsurface, sdf_dataset)
        print("Nearsurface samples saved to :", out_path_nearsurface)

    print("All points")
    points = np.concatenate([
        uniform_points.reshape(-1, 3), near_surface_points.reshape(-1, 3)
    ], axis=0)
    sdf = np.concatenate([uniform_sdf, near_surface_sdf])
    print("Points:", points.shape, "SDF:", sdf.shape)

    sdf_dataset = {
        'points': points,
        'sdf': sdf,
        "mesh": mesh
    }
    np.save(out_path, sdf_dataset)
    print("Final samples saved to :", out_path)
