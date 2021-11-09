import torch
import numpy as np
from torch.utils import data
from torch.utils.data import Dataset

class DeformHandlesDataset(Dataset):

    def __init__(self, cfg, cfgdata, train=True):
        self.cfg = cfg
        self.train = train
        self.cfgdata = cfgdata

        self.length = int(getattr(cfgdata, "length", 1000))
        self.handles_data = np.load(cfg.path, allow_pickle=True).item()
        self.handles = self.handles_data['handles']
        self.targets = self.handles_data['targets']
        if 'gtr_verts' in self.handles_data:
            self.verts = self.handles_data['gtr_verts']
        else:
            self.verts = None
        if 'gtr_faces' in self.handles_data:
            self.faces = self.handles_data['gtr_faces']
        else:
            self.faces = None

        self.dim = self.handles.shape[-1]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.verts is not None and self.faces is not None:
            return {
                'idx': idx,
                'handles':
                    torch.from_numpy(self.handles).float().view(-1, self.dim),
                'targets':
                    torch.from_numpy(self.targets).float().view(-1, self.dim),
                'gtr_verts':
                    torch.from_numpy(self.verts).float().view(-1, self.dim),
                'gtr_faces':
                    torch.from_numpy(self.faces).float().view(-1, self.dim)
            }
        else:
            return {
                'idx': idx,
                'handles': torch.from_numpy(
                    self.handles).float().view(-1, self.dim),
                'targets': torch.from_numpy(
                    self.targets).float().view(-1, self.dim),
            }


def init_np_seed(worker_id):
    seed = torch.initial_seed()
    np.random.seed(seed % 4294967296)


def get_datasets(cfg, args):
    tr_dataset = DeformHandlesDataset(cfg, cfg.train, train=True)
    te_dataset = DeformHandlesDataset(cfg, cfg.val, train=False)
    return tr_dataset, te_dataset


def get_data_loaders(cfg, args):
    tr_dataset, te_dataset = get_datasets(cfg, args)
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
