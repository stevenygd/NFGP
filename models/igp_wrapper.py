"""
There are three types of wrappers we provided:
1. Distillation:
    The input neural field is F, the wrapper will provide another field G that
    has the same structure as F, and (optionally) reload parameters of F to G
2. Correction:
    The input neural field is F, and the wrapper will create a correction network
    C to compose with F to create a new field G(x) = C(x) + F(x)
3. Deformation:
    The input neural field is F, and the warpper will create a deformatoin
    field D to warp the input. With that, the output field G is
    G(x) = F(D(x)).

Note that one can potentially combine multiple wrappers together. In this
repository, we will only provide function for the individual wrappers.
"""
import torch
import importlib
import torch.nn as nn
from trainers.utils.utils import get_opt


def distillation(cfg, net, reload=True):
    """
    For distillation model, we will initialize the same model,
    reload the weight to do finetuning if needed.

    :param cfg: Whole configuration file
    :param net: The neural field.
    :param reload: Whether reload the reinstantiated neural fields with
                   pretrained parameters.
    :return:
        [decoder]
        [opt_distll_dec]
        [scheduler_distill_dec]
    """
    if hasattr(cfg.models, "distill_decoder"):
        distill_decoder = cfg.models.distill_decoder
    else:
        distill_decoder = cfg.models.decoder
    dec_lib = importlib.import_module(distill_decoder.type)
    decoder = dec_lib.Net(cfg, distill_decoder)
    decoder.cuda()
    if reload:
        decoder.load_state_dict(net.state_dict())

    if hasattr(cfg.trainer, "distill_opt"):
        distill_opt = cfg.trainer.distill_opt
    else:
        distill_opt = cfg.trainer.opt
    if hasattr(cfg.trainer, "distill_lr"):
        lr = float(getattr(cfg.trainer, "distill_lr"))
    else:
        lr = None
    opt_distll_dec, scheduler_distill_dec = get_opt(
        decoder.parameters(), distill_opt, overwrite_lr=lr)

    return decoder, opt_distll_dec, scheduler_distill_dec


def deformation(cfg, net):
    """
    For deform, we return F(D(x)), where F is the [net] and D is the deformation
    network that transform the space [x].
    :param cfg: Whole configuration file.
    :param net: The neural field.
    :return:
        [out]
        [opt_deform_dec]
        [scheduler_deform_dec]
    """
    param_lst = []
    if hasattr(cfg.models, "deform_decoder"):
        deform_decoder = cfg.models.deform_decoder
        d_dec_lib = importlib.import_module(deform_decoder.type)
        d_decoder = d_dec_lib.Net(cfg, deform_decoder)
        d_decoder.cuda()
        print(d_decoder)
        param_lst += list(d_decoder.parameters())
    else:
        d_decoder = None

    if hasattr(cfg.trainer, "opt_deform"):
        deform_opt = cfg.trainer.opt_deform
    else:
        deform_opt = cfg.trainer.opt
    opt_deform_dec, scheduler_deform_dec = get_opt(param_lst, deform_opt)

    assert hasattr(cfg.models, "deform_wrapper")
    out = DeformationWrapper(
        net, cfg.models.deform_wrapper, d_decoder, None)
    return out, opt_deform_dec, scheduler_deform_dec


def correction(cfg, net):
    """
    Wrapper that produce G(x) = C(x) + F(x) for input neural field F. C(x) is
    a network defined by the configuration [cfg.models.correct_decoder]
    :param cfg: Whole configuration file.
    :param net: The neural field.
    :return:
        [out]
        [opt_deform_dec]
        [scheduler_deform_dec]
    """
    param_lst = []
    if hasattr(cfg.models, "correct_decoder"):
        correct_decoder = cfg.models.correct_decoder
        c_dec_lib = importlib.import_module(correct_decoder.type)
        c_decoder = c_dec_lib.Net(cfg, correct_decoder)
        c_decoder.cuda()
        print(c_decoder)
        param_lst += list(c_decoder.parameters())
    else:
        c_decoder = None

    if hasattr(cfg.trainer, "opt_correct"):
        deform_opt = cfg.trainer.opt_correct
    else:
        deform_opt = cfg.trainer.opt
    opt_deform_dec, scheduler_deform_dec = get_opt(param_lst, deform_opt)

    assert hasattr(cfg.models, "deform_wrapper")
    out = DeformationWrapper(
        net, cfg.models.deform_wrapper, None, c_decoder)
    return out, opt_deform_dec, scheduler_deform_dec


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
        if nonlin_type is None:
            return x
        if nonlin_type == 'tanh':
            return torch.tanh(x)
        else:
            raise NotImplemented

    def forward(self, x, return_delta=False, return_both=False):
        x_deform = x
        if self.use_delta_x:
            x_deform = self.deform(x)
            x_deform = self._nonlin_(x_deform, self.nonlin_x)
        delta_x = x_deform - x

        if self.use_delta_s:
            delta_s = self.correct(x)
            delta_s = self._nonlin_(delta_s, self.nonlin_s)
        else:
            delta_s = torch.zeros_like(x).mean(dim=-1, keepdim=True)

        if return_delta and not return_both:
            return delta_x, delta_s

        out = self.orig(x_deform) + delta_s
        if return_both:
            return out, delta_x, delta_s
        else:
            return out


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