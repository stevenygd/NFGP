import torch
import importlib
import torch.nn as nn
from trainers.utils.utils import get_opt


def distillation(cfg, net, reload=True):
    """
    For distillation model, we will initialize a completely the same model,
    reload the weight to do finetuning if needed.

    :param cfg:
    :param net:
    :param lr:
    :return:
    """
    if hasattr(cfg.models, "distill_decoder"):
        distill_decoder = cfg.models.distill_decoder
    else:
        distill_decoder = cfg.models.decoder
    dec_lib = importlib.import_module(distill_decoder.type)
    decoder = dec_lib.Decoder(cfg, distill_decoder)
    decoder.cuda()
    if reload:
        decoder.load_state_dict(net.state_dict())

    if hasattr(cfg.trainer, "distill_opt"):
        distill_opt = cfg.trainer.distill_opt
    else:
        distill_opt = cfg.trainer.opt_dec
    if hasattr(cfg.trainer, "distill_lr"):
        lr =float(getattr(cfg.trainer, "distill_lr"))
    else:
        lr = None
    opt_distll_dec, scheduler_distill_dec = get_opt(
        decoder.parameters(), distill_opt, overwrite_lr=lr)

    return decoder, opt_distll_dec, scheduler_distill_dec


class DeformationWrapper(nn.Module):
    def __init__(self, orig, cfg, deform=None, correct=None):
        super().__init__()
        assert not (deform is None and correct is None)
        self.cfg = cfg
        self.orig = orig
        self.deform = deform
        self.correct = correct
        self.nonlin_x = getattr(cfg, "nonlin_x", None)
        self.nonlin_s = getattr(cfg, "nonlin_s", None)
        self.delta_x_add = getattr(cfg, "delta_x_add", True)
        assert not self.delta_x_add
        self.use_delta_x = deform is not None
        self.use_delta_s = correct is not None

    def _nonlin_(self, x, nonlin_type):
        if not  nonlin_type:
            return x
        if nonlin_type == 'tanh':
            return torch.tanh(x)
        else:
            raise  NotImplemented

    def forward(self, x, z, return_delta=False, return_both=False):
        if self.use_delta_x:
            delta_x = self.deform(x, z)
            delta_x = self._nonlin_(delta_x, self.nonlin_x)
        else:
            delta_x = torch.zeros_like(x)

        if not self.delta_x_add:
            delta_x = delta_x - x

        if self.use_delta_s:
            delta_s = self.correct(x, z)
            delta_s = self._nonlin_(delta_s, self.nonlin_s)
        else:
            delta_s = torch.zeros_like(x).mean(dim=-1, keepdim=True)

        if return_delta and not return_both:
            return delta_x, delta_s

        out = self.orig(x + delta_x, z) + delta_s
        if return_both:
            return out, delta_x, delta_s
        else:
            return out


def deformation(cfg, net):
    param_lst = []
    if hasattr(cfg.models, "deform_decoder"):
        deform_decoder = cfg.models.deform_decoder
        d_dec_lib = importlib.import_module(deform_decoder.type)
        d_decoder = d_dec_lib.Decoder(cfg, deform_decoder)
        d_decoder.cuda()
        print(d_decoder)
        param_lst += list(d_decoder.parameters())
    else:
        d_decoder = None

    assert hasattr(cfg.trainer, "opt_deform")
    deform_opt = cfg.trainer.opt_deform
    opt_deform_dec, scheduler_deform_dec = get_opt(param_lst, deform_opt)

    assert hasattr(cfg.models, "deform_wrapper")
    out = DeformationWrapper(
        net, cfg.models.deform_wrapper, d_decoder, None)
    return out, opt_deform_dec, scheduler_deform_dec



def correction(cfg, net):
    param_lst = []
    if hasattr(cfg.models, "correct_decoder"):
        correct_decoder = cfg.models.correct_decoder
        c_dec_lib = importlib.import_module(correct_decoder.type)
        c_decoder = c_dec_lib.Decoder(cfg, correct_decoder)
        c_decoder.cuda()
        print(c_decoder)
        param_lst += list(c_decoder.parameters())
    else:
        c_decoder = None

    assert hasattr(cfg.trainer, "opt_deform")
    deform_opt = cfg.trainer.opt_deform
    opt_deform_dec, scheduler_deform_dec = get_opt(param_lst, deform_opt)

    assert hasattr(cfg.models, "deform_wrapper")
    out = DeformationWrapper(
        net, cfg.models.deform_wrapper, None, c_decoder)
    return out, opt_deform_dec, scheduler_deform_dec


def fixed_point_invert(g, y, iters=15, verbose=False):
    with torch.no_grad():
        x = y
        dim = x.size(-1)
        for i in range(iters):
            x = y - g(x)
            if verbose:
                err = (y - (x + g(x))).view(-1, dim).norm(dim=-1).mean()
                err = err.detach().cpu().item()
                print("iter:%d err:%s" % (i, err))
    return x