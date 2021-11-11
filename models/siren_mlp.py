import torch
import numpy as np
import torch.nn as nn


# https://github.com/vsitzmann/siren/blob/4df34baee3f0f9c8f351630992c1fe1f69114b5f/modules.py#L622
def sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # See supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(
                -np.sqrt(6 / num_input) / 30, np.sqrt(6 / num_input) / 30)


def first_layer_sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # See paper sec. 3.2, final paragraph, and supplement
            # Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-1 / num_input, 1 / num_input)


class Sine(nn.Module):
    def __init__(self, const=30.):
        super().__init__()
        self.const = const

    def forward(self, input):
        # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
        return torch.sin(self.const * input)


class Decoder(nn.Module):
    """ Decoder conditioned by adding.

    Example configuration:
        z_dim: 128
        hidden_size: 256
        n_blocks: 5
        out_dim: 3  # we are outputting the gradient
        sigma_condition: True
        xyz_condition: True
    """
    def __init__(self, _, cfg):
        super().__init__()
        self.cfg = cfg
        self.dim = dim = cfg.dim
        self.out_dim = out_dim = cfg.out_dim
        self.hidden_size = hidden_size = cfg.hidden_size
        self.n_blocks = n_blocks = cfg.n_blocks

        # Network modules
        self.blocks = nn.ModuleList()
        self.blocks.append(nn.Linear(dim, hidden_size))
        for _ in range(n_blocks):
            self.blocks.append(nn.Linear(hidden_size, hidden_size))
        self.blocks.append(nn.Linear(hidden_size, out_dim))
        self.act = Sine()

        # Initialization
        self.apply(sine_init)
        self.blocks[0].apply(first_layer_sine_init)
        if getattr(cfg, "zero_init_last_layer", False):
            torch.nn.init.constant_(self.blocks[-1].weight, 0)
            torch.nn.init.constant_(self.blocks[-1].bias, 0)

    # This should have the same signature as the sig condition one
    def forward(self, x):
        """
        :param x: (bs, npoints, self.dim) Input coordinate (xyz)
        :param c: (bs, self.zdim + 1) Shape latent code + sigma
        TODO: will ignore [c] for now
        :return: (bs, npoints, self.dim) Gradient (self.dim dimension)
        """
        # net = x.transpose(1, 2)  # (bs, dim, n_points)
        net = x  # (bs, n_points, dim)
        for block in self.blocks[:-1]:
            net = self.act(block(net))
        out = self.blocks[-1](net)
        return out
