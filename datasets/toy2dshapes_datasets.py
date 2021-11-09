import torch
import numpy as np
from torch.utils import data
from torch.utils.data import Dataset


# Define a ground truth implicit field
def pt2seg(x1, y1, x2, y2, x3, y3):
    # x3,y3 is the point
    x1 = torch.ones(1, 1) * x1
    x2 = torch.ones(1, 1) * x2
    y1 = torch.ones(1, 1) * y1
    y2 = torch.ones(1, 1) * y2
    px = x2 - x1
    py = y2 - y1

    norm = px * px + py * py

    u = ((x3 - x1) * px + (y3 - y1) * py) / float(norm)
    u = torch.clamp(u, 0, 1)

    x = x1 + u * px
    y = y1 + u * py

    dx = x - x3
    dy = y - y3

    # Note: If the actual distance does not matter,
    # if you only want to compare what this function
    # returns to other results of this function, you
    # can just return the squared distance instead
    # (i.e. remove the sqrt) to gain a little performance
    dist = (dx * dx + dy * dy) ** .5

    return dist


def SDF_line_seg(x1, y1, x2, y2, x3, y3, signed=False):
    dist = pt2seg(x1, y1, x2, y2, x3, y3).view(-1)
    if signed:
        n = torch.from_numpy(np.array([y1 - y2, x2 - x1]))
        n = n / n.norm()
        mx, my = (x1 + x2) * 0.5, (y1 + y2) * 0.5
        sign = torch.sign((mx - x3) * n[0] + (my - y3) * n[1]).view(-1)
        sign[sign >= 0] = 1.
        print("unique", torch.unique(sign))
        signed_dist = sign.float() * dist
        return signed_dist
    else:
        return dist


def square_field(x, a=0.5, b=0.5):
    #     d = SDF_line_seg(-a, 0.1, 0.5, 0.1, x[:, 0], x[:,1])
    d1 = SDF_line_seg(-a, -b, a, -b, x[:, 0], x[:, 1])
    d2 = SDF_line_seg(a, -b, a, b, x[:, 0], x[:, 1])
    d3 = SDF_line_seg(a, b, -a, b, x[:, 0], x[:, 1])
    d4 = SDF_line_seg(-a, b, -a, -b, x[:, 0], x[:, 1])
    dist = torch.min(torch.min(d1, d2), torch.min(d3, d4))
    sign = torch.sign(
        torch.max(torch.abs(x) / torch.from_numpy(np.array([a, b])), dim=-1)[0] - 1
    )
    return dist * sign


def circle_field(x, radius=0.5, center_x=0., center_y=0.):
    x = x.view(-1, 2)
    c = torch.from_numpy(np.array([center_x, center_y])).view(1, 2).to(x)
    r = (x - c).norm(dim=-1).view(-1)
    return r - radius


# Function to sample points
def square_points(length=1, num_points=1000, a=0.1, b=0.5):
    b = length * 0.5
    n = num_points // 4
    x1 = torch.cat([
        torch.ones(n, 1) *  a, 
        (torch.rand(n, 1) - 0.5) * b * 2
    ], dim=1)
    x2 = torch.cat([
        -torch.ones(n, 1) *  a, 
        (torch.rand(n, 1) - 0.5) * b * 2
    ], dim=1)
    x3 = torch.cat([
        (torch.rand(n, 1) - 0.5) * a * 2,
        torch.ones(n, 1) * b
    ], dim=1)
    x4 = torch.cat([
        (torch.rand(n, 1) - 0.5) * a * 2,
        -torch.ones(n, 1) * b 
    ], dim=1)
    return torch.cat([x1, x2, x3, x4], dim=0)


def circle_points(num_points=1000, radius=0.5, center_x=0., center_y=0.):
    x = torch.randn(num_points, 2)
    x = x / x.norm(dim=-1).view(-1, 1)
    c = torch.from_numpy(np.array([center_x, center_y])).view(1, 2)
    x = x * radius + c
    return x


class Toy2D(Dataset):

    def __init__(self, cfg, cfgdata, train=True):
        self.cfg = cfg
        self.cfgdata = cfgdata
        self.train = train

        self.tr_npoints = int(getattr(cfg, "tr_max_sample_points", 1024))
        self.te_npoints = int(getattr(cfg, "te_max_sample_points", 1024))

        self.length = int(getattr(cfgdata, "length", 1000))
        self.shape_type = getattr(cfgdata, "shape_type", "square")
        self.shape_cfg = getattr(cfgdata, "shape_cfg", None)
        self.xyz_npoints = int(getattr(cfgdata, "num_sample_points", 1024))

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.shape_type == 'square':
            if self.shape_cfg:
                a = getattr(self.shape_cfg, "a", 0.5)
                b = getattr(self.shape_cfg, "b", 0.5)
            else:
                a, b = 0.5, 0.5
            tr_out = square_points(num_points=self.tr_npoints, a=a, b=b)
            te_out = square_points(num_points=self.te_npoints, a=a, b=b)

            xyz = torch.rand(self.xyz_npoints, 2) * 2 - 1
            sdf = square_field(xyz, a=a, b=b).view(-1, 1)

        elif self.shape_type == 'circle':
            if self.shape_cfg:
                radius = getattr(self.shape_cfg, "radius", 0.5)
                c_x = getattr(self.shape_cfg, "c_x", 0.)
                c_y = getattr(self.shape_cfg, "c_y", 0.)
            else:
                radius, c_x, c_y = 0.5, 0., 0.
            tr_out = circle_points(
                num_points=self.tr_npoints, radius=radius,
                center_x=c_x, center_y=c_y)
            te_out = circle_points(
                num_points=self.te_npoints, radius=radius,
                center_x=c_x, center_y=c_y)

            xyz = torch.rand(self.xyz_npoints, 2) * 2 - 1
            sdf = circle_field(
                xyz, radius=radius, center_x=c_x, center_y=c_y).view(-1, 1)

        else:
            raise NotImplemented

        sign = torch.sign(sdf).view(-1, 1)
        dist = torch.abs(sdf).view(-1, 1)
        return {
            'idx': idx,
            'tr_points': tr_out,
            'te_points': te_out,
            'xyz': xyz,
            'dist': dist,
            'sign': sign,
        }


def init_np_seed(worker_id):
    seed = torch.initial_seed()
    np.random.seed(seed % 4294967296)


def get_datasets(cfg, args):
    tr_dataset = Toy2D(cfg, cfg.train, train=True)
    te_dataset = Toy2D(cfg, cfg.val, train=False)
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
