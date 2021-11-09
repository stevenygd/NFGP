import torch
import numpy as np
import torch.nn as nn
from models.decoders.siren_mlp import first_layer_sine_init, sine_init, Sine


class InvertibleResBlockSIREN(nn.Module):

    def __init__(self, inp_dim, hid_dim, nblocks=1, use_scale_bias=False):
        super().__init__()
        self.dim = inp_dim
        self.nblocks = nblocks

        self.blocks = nn.ModuleList()
        self.blocks.append(
            nn.utils.spectral_norm(nn.Linear(inp_dim, hid_dim)))
        for _ in range(self.nblocks):
            self.blocks.append(
                nn.utils.spectral_norm(nn.Linear(hid_dim, hid_dim)))
        self.blocks.append(nn.utils.spectral_norm(nn.Linear(hid_dim, inp_dim)))

        self.act = Sine(const=1)

        self.use_scale_bias = use_scale_bias
        if self.use_scale_bias:
            self.y_scale = nn.Parameter(
                torch.zeros(1, 1, self.dim), requires_grad=True)
            self.y_bias = nn.Parameter(
                torch.zeros(1, 1, self.dim), requires_grad=True)
        else:
            self.y_scale, self.y_bias = 1, 0

    def forward(self, x):
        y = x
        for block in self.blocks[:-1]:
            y = self.act(block(y)) * 0.5
        y = self.blocks[-1](y) * 0.5
        if self.use_scale_bias:
            y = y * torch.tanh(self.y_scale) + self.y_bias
        return y + x


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
        for _ in range(self.n_blocks):
            self.blocks.append(
                InvertibleResBlockSIREN(
                    self.dim, self.hidden_size, nblocks=1,
                    use_scale_bias=getattr(cfg, "use_scale_bias", False),
                )
            )
        print(self)

    # This should have the same signature as the sig condition one
    def forward(self, x, _):
        """
        :param x: (bs, npoints, self.dim) Input coordinate (xyz)
        :param c: (bs, self.zdim + 1) Shape latent code + sigma
        TODO: will ignore [c] for now
        :return: (bs, npoints, self.dim) Gradient (self.dim dimension)
        """
        out = x
        for block in self.blocks:
            out = block(out)
        return out
