import os
import tqdm
import torch
import importlib
import numpy as np
import torch.nn.functional as F
from trainers.base_trainer import BaseTrainer
from models.decoders.igp_modules import distillation, deformation, correction
from trainers.utils.utils import set_random_seed
from trainers.utils.igp_losses import loss_eikonal, loss_boundary, lap_loss, \
    mean_curvature_match_loss, get_surf_pcl
from trainers.utils.igp_process import filtering_step
from argparse import Namespace
from evaluation.evaluation_metrics import EMD_CD


class Trainer(BaseTrainer):

    def __init__(self, cfg, args, original_decoder=None):
        super().__init__(cfg, args)
        self.cfg = cfg
        self.args = args
        set_random_seed(getattr(self.cfg.trainer, "seed", 666))
        self.dim = 3

        # The networks
        if original_decoder is None:
            sn_lib = importlib.import_module(cfg.models.decoder.type)
            self.original_decoder = sn_lib.Decoder(cfg, cfg.models.decoder)
            self.original_decoder.cuda()
            self.original_decoder.load_state_dict(
                torch.load(cfg.models.decoder.path)['dec'])
            print("Original Decoder:")
            print(self.original_decoder)
        else:
            self.original_decoder = original_decoder

        # Get the wrapper for the operation
        self.wrapper_type = getattr(
            cfg.trainer, "wrapper_type", "distillation")
        if self.wrapper_type in ['distillation']:
            self.decoder, self.opt_dec, self.scheduler_dec = distillation(
                cfg, self.original_decoder,
                reload=getattr(self.cfg.trainer, "reload_decoder", True))
        elif self.wrapper_type in ['deformation']:
            self.decoder, self.opt_dec, self.scheduler_dec = deformation(
                cfg, self.original_decoder)
        elif self.wrapper_type in ['correction']:
            self.decoder, self.opt_dec, self.scheduler_dec = correction(
                cfg, self.original_decoder)
        else:
            raise ValueError("wrapper_type:", self.wrapper_type)

        # Prepare save directory
        os.makedirs(os.path.join(cfg.save_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(cfg.save_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(cfg.save_dir, "val"), exist_ok=True)

        # Set-up counter
        self.num_update_step = 0
        self.boundary_points = None
        self.beta = getattr(self.cfg.trainer, "beta", 1.)

        # whether plot histogram for network weights
        self.show_network_hist = getattr(
            self.cfg.trainer, "show_network_hist", False)

    def update(self, data, *args, **kwargs):
        if self.wrapper_type in ['distillation', 'correction']:
            return self.update_distillation_correction(data, *args, **kwargs)
        elif self.wrapper_type in ['deformation']:
            return self.update_deformation(data, *args, **kwargs)
        else:
            raise NotImplementedError

    def update_deformation(self, data, *args, **kwargs):
        return filtering_step(
            beta=self.beta,
            net=self.decoder,
            opt=self.opt_dec,
            original=self.original_decoder,
            deform=self.decoder.deform,
            dim=self.dim,
            sample_cfg=getattr(
                self.cfg.trainer, "sampling", Namespace()),
            loss_boundary_cfg=getattr(
                self.cfg.trainer, "loss_boundary", Namespace()),
            loss_grad_cfg=getattr(
                self.cfg.trainer, "loss_grad", Namespace()),
            loss_km_cfg=getattr(
                self.cfg.trainer, "loss_km", Namespace()),
            loss_lap_cfg=getattr(
                self.cfg.trainer, "loss_lap", Namespace()),
            loss_deform_prior_cfg=getattr(
                self.cfg.trainer, "loss_deform_prior", Namespace()),
            loss_hess_cfg=getattr(
                self.cfg.trainer, "loss_hess", Namespace()),
            grad_clip=getattr(self.cfg.trainer, "grad_clip", None)
        )

    def update_distillation_correction(self, data, *args, **kwargs):
        self.num_update_step += 1
        if 'no_update' in kwargs:
            no_update = kwargs['no_update']
        else:
            no_update = False
        if not no_update:
            self.decoder.train()
            self.opt_dec.zero_grad()

        boundary_loss_weight = float(getattr(
            self.cfg.trainer, "boundary_weight", 1.))
        boundary_loss_num_points = int(getattr(
            self.cfg.trainer, "boundary_num_points", 0))
        boundary_loss_points_update_step = int(getattr(
            self.cfg.trainer, "boundary_loss_points_update_step", 1))
        boundary_loss_use_surf_points = int(getattr(
            self.cfg.trainer, "boundary_loss_use_surf_points", True))
        if boundary_loss_weight > 0. and boundary_loss_num_points > 0:
            # if self.num_update_step % boundary_loss_points_update_step == 0 \
            #         or self.boundary_points is None:
            #     self.boundary_points = get_surf_pcl(
            #         lambda x: self.original_decoder(x, None),
            #         npoints=boundary_loss_num_points, dim=self.dim,
            #     ).detach().cuda().float()
            # assert self.boundary_points is not None

            if self.num_update_step % boundary_loss_points_update_step == 0:
                self.boundary_points = None
            loss_y_boundary, self.boundary_points = loss_boundary(
                (lambda x: self.original_decoder(x, None)),
                (lambda x: self.decoder(x, None)),
                npoints=boundary_loss_num_points,
                x=self.boundary_points,
                dim=self.dim,
                use_surf_points=boundary_loss_use_surf_points
            )
            loss_y_boundary = loss_y_boundary * boundary_loss_weight
        else:
            loss_y_boundary = torch.zeros(1).float().cuda()

        deform_loss_weight = float(getattr(
            self.cfg.trainer, "deform_loss_weight", 1e-4))
        deform_loss_num_points = int(getattr(
            self.cfg.trainer, "deform_loss_num_points", 5000))
        deform_loss_use_surf_pts = int(getattr(
            self.cfg.trainer, "deform_loss_use_surf_pts", False))
        if deform_loss_weight > 0. and deform_loss_num_points > 0 and \
                self.wrapper_type in ['deformation']:
            if deform_loss_use_surf_pts:
                x = get_surf_pcl(
                    lambda x: self.decoder(x, None),
                    npoints=deform_loss_num_points
                )
                print(x.shape)
            else:
                x = torch.rand(1, deform_loss_num_points, 3).cuda().float() * 2 - 1.
            delta_x, delta_s = self.decoder(x, None, return_delta=True)
            delta = torch.cat([delta_x, delta_s], dim=-1)
            loss_deform = F.mse_loss(
                delta, torch.zeros_like(delta)) * deform_loss_weight
        else:
            loss_deform = torch.zeros(1).float().cuda()

        grad_norm_weight = float(getattr(
            self.cfg.trainer, "grad_norm_weight", 1e-2))
        grad_norm_num_points = int(getattr(
            self.cfg.trainer, "grad_norm_num_points", 5000))
        if grad_norm_weight > 0. and grad_norm_num_points > 0:
            loss_unit_grad_norm = loss_eikonal(
                lambda x: self.decoder(x, None),
                npoints= grad_norm_num_points,
                use_surf_points=False, invert_sampling=False
            )
            loss_unit_grad_norm *= grad_norm_weight
        else:
            loss_unit_grad_norm = torch.zeros(1).float().cuda()

        lap_loss_weight = float(getattr(
            self.cfg.trainer, "lap_loss_weight", 1e-4))
        lap_loss_threshold = int(getattr(
            self.cfg.trainer, "lap_loss_threshold", 50))
        lap_loss_num_points = int(getattr(
            self.cfg.trainer, "lap_loss_num_points", 5000))
        if lap_loss_weight > 0. and lap_loss_num_points > 0:
            loss_lap_scaling = lap_loss(
                (lambda x: self.original_decoder(x, None)),
                (lambda x: self.decoder(x, None)),
                npoints=lap_loss_num_points,
                beta=self.beta,
                masking_thr=lap_loss_threshold,
                use_surf_points=False, invert_sampling=False,
            )
            loss_lap_scaling = loss_lap_scaling * lap_loss_weight
        else:
            loss_lap_scaling = torch.zeros(1).float().cuda()

        km_loss_weight = float(getattr(
            self.cfg.trainer, "km_loss_weight", 0.))
        km_loss_num_points = int(getattr(
            self.cfg.trainer, "km_loss_num_points", 5000))
        km_loss_diff_type = getattr(
            self.cfg.trainer, "km_loss_diff_type", 'rel')
        km_loss_use_surf_pts = int(getattr(
            self.cfg.trainer, "km_loss_use_surf_pts", False))
        km_loss_threshold = int(getattr(
            self.cfg.trainer, "km_loss_threshold", 50))
        if km_loss_weight > 0. and km_loss_num_points > 0:
            loss_km, loss_km_mask = mean_curvature_match_loss(
                (lambda x: self.original_decoder(x, None)),
                (lambda x: self.decoder(x, None, return_both=True)),
                beta=self.beta, npoints=km_loss_num_points, dim=self.dim,
                masking_thr=km_loss_threshold, loss_type='l2', eps=0.,
                return_mask=True, diff_type=km_loss_diff_type,
                use_surf_points=km_loss_use_surf_pts
            )
            loss_km *= km_loss_weight
        else:
            loss_km = torch.zeros(1).float().cuda()

        loss = loss_unit_grad_norm + loss_y_boundary + loss_lap_scaling + \
            loss_deform + loss_km
        if not no_update:
            loss.backward()
            self.opt_dec.step()

        return {
            'loss': loss.detach().cpu().item(),
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
        }

    def log_train(self, train_info, train_data, writer=None,
                  step=None, epoch=None, visualize=False, **kwargs):
        if writer is None:
            return

        # Log training information to tensorboard
        train_info = {k: (v.cpu() if not isinstance(v, float) else v)
                      for k, v in train_info.items()}
        for k, v in train_info.items():
            if not ('loss' in k):
                continue
            if step is not None:
                writer.add_scalar(k, v, step)
            else:
                assert epoch is not None
                writer.add_scalar(k, v, epoch)

        if self.show_network_hist:
            for name, p in self.decoder.named_parameters():
                if step is not None:
                    writer.add_histogram("hist/%s" % name, p, step)
                else:
                    assert epoch is not None
                    writer.add_histogram("hist/%s" % name, p, epoch)

        if visualize:
            self.visualize(train_info, train_data, writer=writer,
                  step=step, epoch=epoch, visualize=visualize, **kwargs)

    def visualize(self, train_info, train_data, writer=None,
                  step=None, epoch=None, visualize=False, **kwargs):
        raise NotImplementedError

    def validate(self, test_loader, epoch, *args, **kwargs):
        return {}

    def save(self, epoch=None, step=None, appendix=None, **kwargs):
        d = {
            'dec': self.original_decoder.state_dict(),
            'net_opt_dec': self.opt_dec.state_dict(),
            'next_dec': self.decoder.state_dict(),
            'epoch': epoch,
            'step': step
        }
        if appendix is not None:
            d.update(appendix)
        save_name = "epoch_%s_iters_%s.pt" % (epoch, step)
        path = os.path.join(self.cfg.save_dir, "checkpoints", save_name)
        torch.save(d, path)

    def resume(self, path, strict=True, **kwargs):
        ckpt = torch.load(path)
        self.original_decoder.load_state_dict(ckpt['dec'], strict=strict)
        self.decoder.load_state_dict(ckpt['next_dec'], strict=strict)
        self.opt_dec.load_state_dict(ckpt['net_opt_dec'])
        start_epoch = ckpt['epoch']
        return start_epoch

    def multi_gpu_wrapper(self, wrapper):
        self.decoder = wrapper(self.decoder)

    def epoch_end(self, epoch, writer=None, **kwargs):
        if self.scheduler_dec is not None:
            self.scheduler_dec.step(epoch=epoch)
            if writer is not None:
                writer.add_scalar(
                    'train/opt_dec_lr', self.scheduler_dec.get_lr()[0], epoch)
