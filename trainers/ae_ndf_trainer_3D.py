import os
import tqdm
import torch
import importlib
import numpy as np
import torch.nn.functional as F
from trainers.utils.diff_ops import gradient
from trainers.utils.vis_utils import imf2mesh
from trainers.base_trainer import BaseTrainer
from trainers.utils.utils import get_opt, set_random_seed


try:
    from evaluation.evaluation_metrics import EMD_CD
    eval_reconstruciton = True
except:  # noqa
    # Skip evaluation
    eval_reconstruciton = False


class Trainer(BaseTrainer):

    def __init__(self, cfg, args):
        super().__init__(cfg, args)
        self.cfg = cfg
        self.args = args
        set_random_seed(getattr(self.cfg.trainer, "seed", 666))

        # The networks
        sn_lib = importlib.import_module(cfg.models.decoder.type)
        self.decoder = sn_lib.Decoder(cfg, cfg.models.decoder)
        self.decoder.cuda()
        print("Decoder:")
        print(self.decoder)

        encoder_lib = importlib.import_module(cfg.models.encoder.type)
        self.encoder = encoder_lib.Encoder(cfg.models.encoder)
        self.encoder.cuda()
        print("Encoder:")
        print(self.encoder)

        # The optimizer
        if not (hasattr(self.cfg.trainer, "opt_enc") and
                hasattr(self.cfg.trainer, "opt_dec")):
            self.cfg.trainer.opt_enc = self.cfg.trainer.opt
            self.cfg.trainer.opt_dec = self.cfg.trainer.opt

        self.opt_enc, self.scheduler_enc = get_opt(
            self.encoder.parameters(), self.cfg.trainer.opt_enc)
        self.opt_dec, self.scheduler_dec = get_opt(
            self.decoder.parameters(), self.cfg.trainer.opt_dec)

        # Prepare save directory
        os.makedirs(os.path.join(cfg.save_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(cfg.save_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(cfg.save_dir, "val"), exist_ok=True)

        # Prepare variable for summy
        self.oracle_res = None

    def update(self, data, *args, **kwargs):
        if 'no_update' in kwargs:
            no_update = kwargs['no_update']
        else:
            no_update = False
        if not no_update:
            self.encoder.train()
            self.decoder.train()
            self.opt_enc.zero_grad()
            self.opt_dec.zero_grad()

        tr_pts = data['tr_points'].cuda()  # (B, #points, 3)smn_ae_trainer.py
        z_mu, z_sigma = self.encoder(tr_pts)
        z = z_mu + 0 * z_sigma

        bs = z.size(0)
        xyz, dist = data['xyz'].cuda(), data['dist'].cuda()
        xyz = xyz.view(bs, -1, xyz.size(-1))
        out = self.decoder(xyz, z)
        ndf_loss_weight = float(getattr(
            self.cfg.trainer, "ndf_loss_weight", 1.))
        if ndf_loss_weight > 0:
            loss_y_ndf = ((torch.abs(out) - dist) ** 2).view(bs, -1).mean()
            loss_y_ndf *= ndf_loss_weight
        else:
            loss_y_ndf = torch.zeros(1).cuda().float()

        sdf_loss_weight = float(getattr(
            self.cfg.trainer, "sdf_loss_weight", 0.))
        if 'sign' in data and sdf_loss_weight > 0:
            sign = data['sign'].cuda().float()
            loss_y_sdf = ((out - dist * sign) ** 2).view(bs, -1).mean()
            loss_y_sdf *= sdf_loss_weight
        else:
            loss_y_sdf = 0. * torch.zeros(1).to(loss_y_ndf)

        occ_loss_weight = float(getattr(
            self.cfg.trainer, "occ_loss_weight", 0.))
        if 'sign' in data and occ_loss_weight > 0:
            target = (data['sign'].cuda().float() >= 0).float()
            loss_occ = F.binary_cross_entropy(
                torch.sigmoid(out), target
            )
            loss_occ *= occ_loss_weight
        else:
            loss_occ = 0. * torch.zeros(1).cuda().float()

        grad_norm_weight = float(getattr(
            self.cfg.trainer, "grad_norm_weight", 0.))
        grad_norm_num_points = int(getattr(
            self.cfg.trainer, "grad_norm_num_points", 0))
        if grad_norm_weight > 0. and grad_norm_num_points > 0:
            xyz = torch.rand(
                bs, grad_norm_num_points, xyz.size(-1)).to(xyz) * 2 - 1
            xyz = xyz.cuda()
            xyz.requires_grad = True
            grad_norm = gradient(self.decoder(xyz, z), xyz).view(
                bs, -1, xyz.size(-1)).norm(dim=-1)
            loss_unit_grad_norm = F.mse_loss(
                grad_norm, torch.ones_like(grad_norm)) * grad_norm_weight
        else:
            loss_unit_grad_norm = 0. * torch.zeros(1).to(loss_y_ndf)
        loss = loss_unit_grad_norm + loss_y_ndf + loss_y_sdf + loss_occ

        if not no_update:
            loss.backward()
            self.opt_enc.step()
            self.opt_dec.step()

        return {
            'loss': loss.detach().cpu().item(),
            'loss_y_ndf': loss_y_ndf.detach().cpu().item(),
            'loss_y_sdf': loss_y_sdf.detach().cpu().item(),
            'loss_occ': loss_occ.detach().cpu().item(),
            'loss_grad_norm': loss_unit_grad_norm.detach().cpu().item(),
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
                writer.add_scalar('train/' + k, v, step)
            else:
                assert epoch is not None
                writer.add_scalar('train/' + k, v, epoch)

        if visualize:
            with torch.no_grad():
                print("Visualize: %s" % step)
                # TODO: use marching cube to save the meshes
                res = int(getattr(self.cfg.trainer, "vis_mc_res", 256))
                thr = float(getattr(self.cfg.trainer, "vis_mc_thr", 0.))

                mesh = imf2mesh(
                    lambda x: self.decoder(x, None), res=res, threshold=thr)
                if mesh is not None:
                    save_name = "mesh_%diters.obj" \
                                % (step if step is not None else epoch)
                    path = os.path.join(self.cfg.save_dir, "val", save_name)
                    mesh.export(path)

    def validate(self, test_loader, epoch, *args, **kwargs):
        if not eval_reconstruciton:
            return {}

    def save(self, epoch=None, step=None, appendix=None, **kwargs):
        d = {
            'opt_enc': self.opt_enc.state_dict(),
            'opt_dec': self.opt_dec.state_dict(),
            'dec': self.decoder.state_dict(),
            'enc': self.encoder.state_dict(),
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
        self.encoder.load_state_dict(ckpt['enc'], strict=strict)
        self.decoder.load_state_dict(ckpt['dec'], strict=strict)
        self.opt_enc.load_state_dict(ckpt['opt_enc'])
        self.opt_dec.load_state_dict(ckpt['opt_dec'])
        start_epoch = ckpt['epoch']
        return start_epoch

    def multi_gpu_wrapper(self, wrapper):
        self.encoder = wrapper(self.encoder)
        self.decoder = wrapper(self.decoder)

    def epoch_end(self, epoch, writer=None, **kwargs):
        if self.scheduler_dec is not None:
            self.scheduler_dec.step(epoch=epoch)
            if writer is not None:
                writer.add_scalar(
                    'train/opt_dec_lr', self.scheduler_dec.get_lr()[0], epoch)
        if self.scheduler_enc is not None:
            self.scheduler_enc.step(epoch=epoch)
            if writer is not None:
                writer.add_scalar(
                    'train/opt_enc_lr', self.scheduler_enc.get_lr()[0], epoch)

