import os
import tqdm
import torch
import importlib
import numpy as np
import torch.nn.functional as F
from trainers.base_trainer import BaseTrainer
from models.decoders.igp_modules import distillation, deformation
from trainers.utils.utils import set_random_seed
from trainers.utils.igp_process import deform_step, deform_mesh_o3d
from argparse import Namespace


class Trainer(BaseTrainer):

    def __init__(self, cfg, args, original_decoder=None):
        super().__init__(cfg, args)
        self.cfg = cfg
        self.args = args
        set_random_seed(getattr(self.cfg.trainer, "seed", 666))

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
        else:
            raise ValueError("wrapper_type:", self.wrapper_type)

        # Prepare save directory
        os.makedirs(os.path.join(cfg.save_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(cfg.save_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(cfg.save_dir, "val"), exist_ok=True)
        os.makedirs(os.path.join(cfg.save_dir, "vis"), exist_ok=True)

        # Set-up counter
        self.num_update_step = 0
        self.boundary_points = None

        # Set up basic parameters
        self.dim = getattr(cfg.trainer, "dim", 3)
        self.grad_clip = getattr(cfg.trainer, "grad_clip", None)
        if hasattr(cfg.trainer, "lag_mul"):
            self.lag_mult = float(cfg.trainer.lag_mul)
        else:
            self.lag_mult = None
        self.loss_h_weight = getattr(cfg.trainer, "loss_h_weight", 100)
        self.loss_h_thr = getattr(cfg.trainer, "loss_h_thr", 1e-3)

        if hasattr(cfg.trainer, "loss_g"):
            self.loss_g_cfg = cfg.trainer.loss_g
        else:
            self.loss_g_cfg = Namespace(**{})

        if hasattr(cfg.trainer, "loss_km"):
            self.loss_km_cfg = cfg.trainer.loss_km
        else:
            self.loss_km_cfg = Namespace(**{})

        if hasattr(cfg.trainer, "loss_orth"):
            self.loss_orth_cfg = cfg.trainer.loss_orth
        else:
            self.loss_orth_cfg = Namespace(**{})

        if hasattr(cfg.trainer, "loss_det"):
            self.loss_det_cfg = cfg.trainer.loss_det
        else:
            self.loss_det_cfg = Namespace(**{})

        if hasattr(cfg.trainer, "loss_hess"):
            self.loss_hess_cfg = cfg.trainer.loss_hess
        else:
            self.loss_hess_cfg = Namespace(**{})

        if hasattr(cfg.trainer, "loss_stretch"):
            self.loss_stretch_cfg = cfg.trainer.loss_stretch
        else:
            self.loss_stretch_cfg = Namespace()

        if hasattr(cfg.trainer, "sample_cfg"):
            self.sample_cfg = cfg.trainer.sample_cfg
        else:
            self.sample_cfg = None

    def update(self, data, *args, **kwargs):
        self.num_update_step += 1
        handles_ts = data['handles'].cuda().float()
        targets_ts = data['targets'].cuda().float()
        if 'x' in data and 'weights' in data:
            x_ts = data['x'].cuda().float()
            w_ts = data['weights'].cuda().float()
        else:
            x_ts = None
            w_ts = 1.

        loss_g_weight = float(getattr(self.loss_g_cfg, "weight", 1e-3))
        loss_km_weight = float(getattr(self.loss_km_cfg, "weight", 1e-3))
        loss_orth_weight = float(getattr(self.loss_orth_cfg, "weight", 0.))
        loss_det_weight = float(getattr(self.loss_det_cfg, "weight", 0.))
        loss_hess_weight = float(getattr(self.loss_hess_cfg, "weight", 0.))
        loss_stretch_weight = float(
            getattr(self.loss_stretch_cfg, "weight", 0))
        step_res = deform_step(
            self.decoder, self.opt_dec, self.original_decoder,
            handles_ts, targets_ts, dim=self.dim,
            x=x_ts, weights=w_ts,
            sample_cfg=self.sample_cfg,
            # Loss handle
            loss_h_weight=self.loss_h_weight, lag_mult=self.lag_mult,
            loss_h_thr=self.loss_h_thr,
            # Loss G
            loss_g_weight=loss_g_weight,
            n_g_pts=getattr(self.loss_g_cfg, "num_points", 5000),
            # Loss KM
            loss_km_weight=loss_km_weight,
            n_km_pts=getattr(self.loss_km_cfg, "num_points", 5000),
            km_mask_thr=getattr(self.loss_km_cfg, "mask_thr", 10),
            km_diff_type=getattr(self.loss_km_cfg, "diff_type", "abs"),
            km_use_surf_points=getattr(self.loss_km_cfg, "use_surf_points", True),
            use_lapbal=getattr(self.loss_km_cfg, "use_lapbal", True),
            km_invert_sample=getattr(self.loss_km_cfg, "invert_sample", True),

            # Loss orthogonality
            loss_orth_weight=loss_orth_weight,
            n_orth_pts=getattr(self.loss_orth_cfg, "num_points", 5000),
            orth_reg_type=getattr(self.loss_orth_cfg, "orth_reg_type", "so"),
            orth_use_surf_points=getattr(self.loss_orth_cfg, "use_surf_points", False),
            orth_invert_sample=getattr(self.loss_orth_cfg, "invert_sample", True),

            # Loss Jacobian determinant
            loss_det_weight=loss_det_weight,
            n_det_pts=getattr(self.loss_det_cfg, "num_points", 5000),
            det_use_surf_points=getattr(self.loss_det_cfg, "use_surf_points", False),
            det_invert_sample=getattr(self.loss_det_cfg, "invert_sample", True),
            det_detach_weight=getattr(self.loss_det_cfg, "detach_weight", False),

            # Loss Hessian
            loss_hess_weight=loss_hess_weight,
            n_hess_pts=getattr(self.loss_hess_cfg, "num_points", 5000),
            hess_use_surf_points=getattr(self.loss_hess_cfg, "use_surf_points", True),
            hess_invert_sample=getattr(self.loss_hess_cfg, "invert_sample", True),
            hess_type=getattr(self.loss_hess_cfg, "hess_type", 'direct'),
            hess_tang_proj=getattr(self.loss_hess_cfg, "tang_proj", True),
            hess_tang_extend=getattr(self.loss_hess_cfg, "tang_extend", False),
            hess_use_weight=getattr(self.loss_hess_cfg, "use_weight", True),
            hess_quantile=getattr(self.loss_hess_cfg, "quantile", None),
            hess_use_bending=getattr(self.loss_hess_cfg, "use_bending", True),
            hess_detach_weight=getattr(self.loss_hess_cfg, "detach_weight", False),
            hess_use_rejection=getattr(self.loss_hess_cfg, "use_rejection", False),
            hess_use_square=getattr(self.loss_hess_cfg, "use_square", False),

            # Loss stretch
            loss_stretch_weight=loss_stretch_weight,
            n_s_pts=getattr(self.loss_stretch_cfg, "num_points", 5000),
            stretch_use_surf_points=getattr(
                self.loss_stretch_cfg, "use_surf_points", True),
            stretch_invert_sample=getattr(
                self.loss_stretch_cfg, "invert_sample", True),
            stretch_extend_tang=getattr(
                self.loss_stretch_cfg, "tang_extend", True),
            stretch_proj_tang=getattr(
                self.loss_stretch_cfg, "tang_proj", True),
            stretch_loss_type=getattr(
                self.loss_stretch_cfg, "loss_type", "simple"),
            stretch_use_weight=getattr(
                self.loss_stretch_cfg, "use_weight", True),
            stretch_alpha=getattr(self.loss_stretch_cfg, "alpha", 0.),
            stretch_beta=getattr(self.loss_stretch_cfg, "beta", 1.),
            stretch_detach_weight=getattr(
                self.loss_stretch_cfg, "detach_weight", False),
            stretch_use_rejection=getattr(
                self.loss_stretch_cfg, "use_rejection", False),
            stretch_use_square=getattr(
                self.loss_stretch_cfg, "use_square", False),

            # Gradient clipping
            grad_clip=self.grad_clip,
        )
        self.lag_mult = step_res['lag_mult']
        if step_res['lag_mult']:
            step_res['lag_mult'] = step_res['lag_mult'].mean()
        step_res = {
            ('loss/%s' % k): v for k, v in step_res.items()
        }
        step_res['loss'] = step_res['loss/loss']
        step_res.update({
            "weight/loss_h_weight": self.loss_h_weight,
            "weight/loss_km_weight": loss_km_weight,
            'weight/loss_orth_weight': loss_orth_weight,
            'weight/loss_det_weight': loss_det_weight,
            'weight/loss_hess_weight': loss_hess_weight,
            'weight/loss_stretch_weight': loss_stretch_weight,
        })
        return step_res

    def log_train(self, train_info, train_data, writer=None,
                  step=None, epoch=None, visualize=False, **kwargs):
        raise NotImplementedError

    def validate(self, test_loader, epoch, *args, **kwargs):
        # TODO: compute mesh and compute the manifold harmonics to
        #       see if the high frequencies signals are dimed/suppressed
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
