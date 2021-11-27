import os
import yaml
import argparse
import importlib
import os.path as osp

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def load_imf(log_path, config_fpath=None, ckpt_fpath=None,
             epoch=None, verbose=False,
             return_trainer=False, return_cfg=False):
    # Load configuration
    if config_fpath is None:
        config_fpath = osp.join(log_path, "config", "config.yaml")
    with open(config_fpath) as f:
        cfg = dict2namespace(yaml.load(f))
    cfg.save_dir = "logs"

    # Load pretrained checkpoints
    ep2file = {}
    last_file, last_ep = osp.join(log_path, "latest.pt"), -1
    if ckpt_fpath is not None:
        last_file = ckpt_fpath
    else:
        ckpt_path = osp.join(log_path, "checkpoints")
        if osp.isdir(ckpt_path):
            for f in os.listdir(ckpt_path):
                if not f.endswith(".pt"):
                    continue
                ep = int(f.split("_")[1])
                if verbose:
                    print(ep, f)
                ep2file[ep] = osp.join(ckpt_path, f)
                if ep > last_ep:
                    last_ep = ep
                    last_file = osp.join(ckpt_path, f)
            if epoch is not None:
                last_file = ep2file[epoch]
    print(last_file)

    trainer_lib = importlib.import_module(cfg.trainer.type)
    trainer = trainer_lib.Trainer(cfg, None)
    trainer.resume(last_file)

    if return_trainer:
        return trainer, cfg
    else:
        imf = trainer.net
        del trainer
        return imf, cfg


def parse_hparams(hparam_lst):
    print("=" * 80)
    print("Parsing:", hparam_lst)
    out_str = ""
    out = {}
    for i, hparam in enumerate(hparam_lst):
        hparam = hparam.strip()
        k, v = hparam.split("=")[:2]
        k = k.strip()
        v = v.strip()
        print(k, v)
        out[k] = v
        out_str += "%s=%s_" % (k, v.replace("/", "-"))
    print(out)
    print(out_str)
    print("=" * 80)
    return out, out_str


def update_cfg_with_hparam(cfg, k, v):
    k_path = k.split(".")
    cfg_curr = cfg
    for k_curr in k_path[:-1]:
        assert hasattr(cfg_curr, k_curr), "%s not in %s" % (k_curr, cfg_curr)
        cfg_curr = getattr(cfg_curr, k_curr)
    k_final = k_path[-1]
    assert hasattr(cfg_curr, k_final), \
        "Final: %s not in %s" % (k_final, cfg_curr)
    v_type = type(getattr(cfg_curr, k_final))
    setattr(cfg_curr, k_final, v_type(v))


def update_cfg_hparam_lst(cfg, hparam_lst):
    hparam_dict, hparam_str = parse_hparams(hparam_lst)
    for k, v in hparam_dict.items():
        update_cfg_with_hparam(cfg, k, v)
    return cfg, hparam_str
