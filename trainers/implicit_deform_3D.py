import os
import torch
import trimesh
from argparse import Namespace
from trainers.utils.vis_utils import imf2mesh
from evaluation.evaluation_metrics import CD, EMD
from trainers.implicit_deform import Trainer as BaseTrainer
from trainers.utils.igp_utils import compute_deform_weight


class Trainer(BaseTrainer):

    def __init__(self, cfg, args, original_decoder=None):
        super().__init__(cfg, args, original_decoder=original_decoder)
        self.dim = 3
        self.vis_cfg = getattr(self.cfg.trainer, "vis", Namespace())

        # same resolution as the one from
        self.res = int(getattr(self.cfg.trainer, "mc_res", 256))
        self.thr = float(getattr(self.cfg.trainer, "mc_thr", 0.))
        self.original_mesh, self.original_mesh_stats = imf2mesh(
            lambda x: self.original_decoder(x),
            res=self.res, threshold=self.thr,
            normalize=True, norm_type='res', return_stats=True
        )

        if hasattr(self.cfg.trainer, "mesh_presample"):
            self.presample_cfg = self.cfg.trainer.mesh_presample
            self.presmp_npoints = getattr(self.presample_cfg, "num_points", 10000)
        else:
            self.presmp_npoints = None

    def update(self, data, *args, **kwargs):
        if self.presmp_npoints is not None:
            uniform_pcl_orig = self.original_mesh.sample(self.presmp_npoints)
            with torch.no_grad():
                x_invert_uniform = self.decoder.deform.invert(
                    torch.from_numpy(uniform_pcl_orig).float().cuda().view(-1, 3),
                    iters=30
                ).view(1, -1, 3).cuda().float()

            weights = compute_deform_weight(
                x_invert_uniform,
                deform=lambda x: self.decoder.deform(x, None),
                y_net=lambda x: self.original_decoder(x, None),
                x_net=lambda x: self.decoder(x, None),
                surface=True,
                detach=getattr(self.presample_cfg, "detach_weight", False),
            ).cuda().float().view(1, -1)

            data.update({
                'x': x_invert_uniform,
                'weights': weights
            })
        return super().update(data, *args, **kwargs)

    def log_train(self, train_info, train_data, writer=None,
                  step=None, epoch=None, visualize=False, **kwargs):
        if writer is None:
            return

        # Log training information to tensorboard
        for k, v in train_info.items():
            if v is None:
                continue
            if step is not None:
                writer.add_scalar(k, v, step)
            else:
                assert epoch is not None
                writer.add_scalar(k, v, epoch)

        if visualize:
            with torch.no_grad():
                self.visualize(train_data, train_info,
                               writer=writer, step=step, epoch=epoch)

    def visualize(
            self, train_data, train_info,
            writer=None, step=None, epoch=None, **kwargs):
        res = int(getattr(self.cfg.trainer, "vis_mc_res", 64))
        thr = float(getattr(self.cfg.trainer, "vis_mc_thr", 0.))
        with torch.no_grad():
            print("Visualize: %s %s" % (step, epoch))
            mesh = imf2mesh(
                lambda x: self.decoder(x, None), res=res, threshold=thr,
                normalize=True, norm_type='res'
            )
            if mesh is not None:
                save_name = "mesh_%diters.obj" \
                            % (step if step is not None else epoch)
                path = os.path.join(self.cfg.save_dir, "vis", save_name)
                mesh.export(path)

    def validate(self, test_loader, epoch, *args, **kwargs):
        print("Validating : %d" % epoch)
        org_mesh_area = float(self.original_mesh.area)

        cd_gtr = 0
        emd_gtr = 0
        cd_out = 0
        emd_out = 0
        cd_ratio = 0.
        emd_ratio = 0
        area_ratio = 0.
        vol_ratio = 0.

        with torch.no_grad():
            new_mesh, new_mesh_stats = imf2mesh(
                lambda x: self.decoder(x),
                res=self.res, threshold=self.thr,
                normalize=True, norm_type='res', return_stats=True
            )
            if new_mesh is not None:
                save_name = "mesh_%diters.obj" % epoch
                path = os.path.join(self.cfg.save_dir, "val", save_name)
                new_mesh.export(path)

                area_ratio = new_mesh_stats['area'] / (self.original_mesh_stats['area'] + 1e-5)
                vol_ratio = new_mesh_stats['vol'] / (self.original_mesh_stats['vol'] + 1e-5)

                for test_data in test_loader:
                    break
                if 'gtr_verts' in test_data and 'gtr_faces' in test_data:
                    npoints = getattr(self.cfg.trainer, "val_npoints", 2048)
                    gtr_verts = test_data['gtr_verts'].detach().view(-1, 3).cpu().numpy()
                    gtr_faces = test_data['gtr_faces'].detach().view(-1, 3).cpu().numpy()
                    gtr_mesh = trimesh.Trimesh(vertices=gtr_verts, faces=gtr_faces)

                    gtr_pcl0 = gtr_mesh.sample(npoints)
                    gtr_pcl1 = gtr_mesh.sample(npoints)
                    out_pcl = new_mesh.sample(npoints)

                    cd_gtr = CD(gtr_pcl0, gtr_pcl1)
                    cd_out = CD(gtr_pcl0, out_pcl)
                    cd_ratio = cd_out / (cd_gtr + 1e-8)

                    emd_gtr = EMD(gtr_pcl0, gtr_pcl1)
                    emd_out = EMD(gtr_pcl0, out_pcl)
                    emd_ratio = emd_out / (emd_gtr + 1e-8)

        res = {
            'val/org_mesh_area': self.original_mesh_stats['area'],
            'val/org_mesh_vol': self.original_mesh_stats['vol'],
            'val/new_mesh_area': new_mesh_stats['area'],
            'val/new_mesh_vol': new_mesh_stats['vol'],
            'val/area_change_ratio': area_ratio,
            'val/vol_change_ratio': vol_ratio,
            'val/cd_gtr': cd_gtr,
            'val/emd_gtr': emd_gtr,
            'val/cd_out': cd_out,
            'val/emd_out': emd_out,
            'val/cd_ratio': cd_ratio,
            'val/emd_ratio': emd_ratio
        }
        print(res)
        return res


