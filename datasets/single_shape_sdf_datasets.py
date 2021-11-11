import torch
import numpy as np
from torch.utils import data
from torch.utils.data import Dataset


class SingleShape(Dataset):

    def __init__(self, cfg, cfgdata):
        self.cfg = cfg
        self.cfgdata = cfgdata

        # Normalize based on norm, leave some space at top/bottom
        # The vertices will be within [-1, 1]
        self.data = np.load(cfg.path, allow_pickle=True).item()
        self.mesh = self.data['mesh']
        self.points = self.data['points']
        self.sdf = self.data['sdf']
        self.dim = getattr(self.cfg, "dim", 3)
        self.length = int(cfgdata.length)

        # Default display axis order
        self.display_axis_order = [0, 1, 2]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        m = torch.zeros(1, 3).float()
        s = torch.ones(1, 3).float()

        # Sample points
        idxs = np.random.choice(
            self.points.shape[0],
            int(self.cfgdata.num_sample_points)
        )
        xyz = torch.from_numpy(self.points[idxs, :]).float().view(-1, self.dim)
        dist = torch.from_numpy(self.sdf[idxs]).float().view(-1, 1)
        sign = torch.sign(dist).view(-1, 1)
        dist = torch.abs(dist).view(-1, 1)

        return {
            'idx': idx,
            'xyz': xyz,
            'dist': dist,
            'sign': sign,
            'mean': m, 'std': s, 'display_axis_order': self.display_axis_order,
        }


def get_data_loaders(cfg, args):
    tr_dataset = SingleShape(cfg, cfg.train)
    te_dataset = SingleShape(cfg, cfg.val)
    train_loader = data.DataLoader(
        dataset=tr_dataset, batch_size=cfg.train.batch_size,
        shuffle=True, num_workers=cfg.num_workers, drop_last=True,
        worker_init_fn=init_np_seed)
    test_loader = data.DataLoader(
        dataset=te_dataset, batch_size=cfg.val.batch_size,
        shuffle=False, num_workers=cfg.num_workers, drop_last=False,
        worker_init_fn=init_np_seed)

    loaders = {
        "test_loader": test_loader,
        'train_loader': train_loader,
    }
    return loaders


def init_np_seed(worker_id):
    seed = torch.initial_seed()
    np.random.seed(seed % 4294967296)