import torch
import numpy as np
import torch.nn as nn
from models.decoders.igp_modules import fixed_point_invert


class LipBoundedFourierFeatureEmd(nn.Module):

    def __init__(self, inp_features, out_dim, scale=1., cat_inp=True):
        super().__init__()
        self.inp_feat = inp_features
        self.cat_inp = cat_inp
        self.out_dim = out_dim
        self.scale = scale
        assert self.out_dim % 2 == 0

        # Initialize the parameters
        self.linear = nn.utils.spectral_norm(
            nn.Linear(inp_features, self.out_dim // 2, bias=False),
        )
        self.scale = scale

    def forward(self, x, return_state=False):
        """
        :param x: (bs, npoints, inp_features)
        :return: (bs, npoints, out_dim)
        """
        y = self.linear(x)
        const = self.scale * 2 * np.pi
        out = torch.cat([torch.sin(y), torch.cos(y)], dim=-1)
        const_norm = torch.ones_like(out) * const

        if self.cat_inp:
            out = torch.cat([x, out], dim=-1)
            const_norm = torch.cat([torch.ones_like(x), const_norm], dim=-1)
            out = out / const_norm
            out = out / np.sqrt(self.out_dim + 1)
        else:
            out = out / const_norm
            out = out / np.sqrt(self.out_dim)
        return out


class LipBoundedPosEnc(nn.Module):

    def __init__(self, inp_features, n_freq, cat_inp=True):
        super().__init__()
        self.inp_feat = inp_features
        self.n_freq = n_freq
        self.cat_inp = cat_inp
        self.out_dim = 2 * self.n_freq * self.inp_feat
        if self.cat_inp:
            self.out_dim += self.inp_feat

    def forward(self, x):
        """
        :param x: (bs, npoints, inp_features)
        :return: (bs, npoints, 2 * out_features + inp_features)
        """
        assert len(x.size()) == 3
        bs, npts = x.size(0), x.size(1)
        const = (2 ** torch.arange(self.n_freq) * np.pi).view(1, 1, 1, -1)
        const = const.to(x)

        # Out shape : (bs, npoints, out_feat)
        cos_feat = torch.cos(const * x.unsqueeze(-1)).view(
            bs, npts, self.inp_feat, -1)
        sin_feat = torch.sin(const * x.unsqueeze(-1)).view(
            bs, npts, self.inp_feat, -1)
        out = torch.cat(
            [sin_feat, cos_feat], dim=-1).view(
            bs, npts, 2 * self.inp_feat * self.n_freq)
        const_norm = torch.cat(
            [const, const], dim=-1).view(
            1, 1, 1, self.n_freq * 2).expand(
            -1, -1, self.inp_feat, -1).reshape(
            1, 1, 2 * self.inp_feat * self.n_freq)

        if self.cat_inp:
            out = torch.cat([out, x], dim=-1)
            const_norm = torch.cat(
                [const_norm, torch.ones(1, 1, self.inp_feat).to(x)], dim=-1)

            return out / const_norm / np.sqrt(self.n_freq * 2 + 1)
        else:

            return out / const_norm / np.sqrt(self.n_freq * 2)


class InvertibleResBlockLinear(nn.Module):

    def __init__(self, inp_dim, hid_dim, nblocks=1,
                 nonlin='leaky_relu',
                 relu_scale=1., use_scale_bias=False,
                 pos_enc_freq=None, fourier_feat_scale=None):
        super().__init__()
        self.dim = inp_dim
        self.nblocks = nblocks

        self.pos_enc_freq = pos_enc_freq
        if fourier_feat_scale is not None:
            inp_dim_af_pe = hid_dim + self.dim
            self.pos_enc = LipBoundedFourierFeatureEmd(
                self.dim, hid_dim, scale=fourier_feat_scale)
        elif self.pos_enc_freq is not None:
            inp_dim_af_pe = self.dim * (self.pos_enc_freq * 2 + 1)
            self.pos_enc = LipBoundedPosEnc(self.dim, self.pos_enc_freq)
        else:
            self.pos_enc = lambda x: x
            inp_dim_af_pe = inp_dim

        self.blocks = nn.ModuleList()
        self.blocks.append(nn.utils.spectral_norm(
            nn.Linear(inp_dim_af_pe, hid_dim)))
        for _ in range(self.nblocks):
            self.blocks.append(
                nn.utils.spectral_norm(
                    nn.Linear(hid_dim, hid_dim),
                )
            )
        self.blocks.append(
            nn.utils.spectral_norm(
                nn.Linear(hid_dim, self.dim),
            )
        )

        self.relu_scale = relu_scale
        self.nonlin = nonlin.lower()
        if self.nonlin == 'leaky_relu':
            self.act = nn.LeakyReLU()
        elif self.nonlin == 'relu':
            self.act = nn.ReLU()
        elif self.nonlin == 'elu':
            self.act = nn.ELU()
        elif self.nonlin == 'softplus':
            self.act = nn.Softplus()
        else:
            raise NotImplementedError

        self.use_scale_bias = use_scale_bias
        if self.use_scale_bias:
            self.y_scale = nn.Parameter(
                torch.zeros(1, 1, self.dim), requires_grad=True)
            self.y_bias = nn.Parameter(
                torch.zeros(1, 1, self.dim), requires_grad=True)

    def forward_g(self, x):
        orig_dim = len(x.size())
        if orig_dim == 2:
            x = x.unsqueeze(0)

        y = self.pos_enc(x)
        for block in self.blocks[:-1]:
            y = self.act(block(y)) * self.relu_scale
        y = self.blocks[-1](y)
        if self.use_scale_bias:
            y = y * torch.tanh(self.y_scale) + self.y_bias

        if orig_dim == 2:
            y = y.squeeze(0)

        return y

    def forward(self, x):
        return x + self.forward_g(x)

    def invert(self, y, verbose=False, iters=15):
        return fixed_point_invert(
            lambda x: self.forward_g(x), y, iters=iters, verbose=verbose
        )


class InvertivleScaleAndShiftLayer(nn.Module):

    def __init__(self, dim):
        super(InvertivleScaleAndShiftLayer, self).__init__()
        self.a = nn.Parameter(
            torch.zeros(1, 1, dim), requires_grad=True)
        self.b = nn.Parameter(
            torch.zeros(1, 1, dim), requires_grad=True)

    def forward(self, x):
        return x * torch.exp(self.a) + self.b

    def invert(self, y, *args, **kwargs):
        return (y - self.b) * torch.exp(- self.a)



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
        if getattr(cfg, "use_scale_layer_first", False):
            if getattr(cfg, "use_scale_layer", False):
                self.blocks.append(InvertivleScaleAndShiftLayer(self.dim))

        for _ in range(self.n_blocks):
            self.blocks.append(
                InvertibleResBlockLinear(
                    self.dim, self.hidden_size, nblocks=1,
                    nonlin=getattr(cfg, "nonlin", False),
                    relu_scale=getattr(cfg, "relu_scale", 1.),
                    use_scale_bias=getattr(cfg, "use_scale_bias", False),
                    pos_enc_freq=getattr(cfg, "pos_enc_freq", None),
                    fourier_feat_scale=getattr(cfg, "fourier_feat_scale", None)
                )
            )
            if getattr(cfg, "use_scale_layer", False):
                self.blocks.append(InvertivleScaleAndShiftLayer(self.dim))

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

    def invert(self, y, verbose=False, iters=15):
        x = y
        for block in self.blocks[::-1]:
            x = block.invert(x, verbose=verbose, iters=iters)
        return x