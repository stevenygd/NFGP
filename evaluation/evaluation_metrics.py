import torch
import warnings
import numpy as np
from numpy.linalg import norm
from scipy.stats import entropy
from sklearn.neighbors import NearestNeighbors
from scipy.optimize import linear_sum_assignment


def CD(x, y, dist=None, return_dist=False):
    """
    Compute chamfer distance
    :param x:  (bs, npts, dim)
    :param y:  (bs, mpts, dim)
    :param dist: (bs, npts, mpts) precomputed pairwise distances
    :param return_dist: Whether return the pairwise distances
    :return: (bs,) Batch of chamfer distance
    """
    with torch.no_grad():
        if dist is None:
            bs, npts, mpts, dim = x.size(0), x.size(1), y.size(2), x.size(2)
            dim = x.shape[-1]
            x = x.reshape(bs, npts, 1, dim)
            y = y.reshape(bs, 1, mpts, dim)
            dist = (x - y).norm(dim=-1, keepdim=False)  # (bs, npts, mpts)
        cd = 0.5 * (
                dist.min(dim=1, keepdim=False)[0].mean(dim=1, keepdim=False) +
                dist.min(dim=1, keepdim=False)[0].mean(dim=1, keepdim=False))
        cd = cd.view(-1)
    if return_dist:
        return cd, dist
    return cd, None


def EMD(x, y, dist=None, return_dist=False):
    """ Computing earth mover distance bewteen two batch of points
    :param x:  (bs, npts, dim)
    :param y:  (bs, mpts, dim)
    :param dist: (bs, npts, mpts) precomputed pairwise distances
    :param return_dist: Whether return the pairwise distances
    :return: (bs,) Batch of EMDs
    """
    with torch.no_grad():
        bs, npts, mpts, dim = x.size(0), x.size(1), y.size(2), x.size(2)
        assert npts == mpts, "EMD only works if two point clouds are equal size"
        if dist is None:
            dim = x.shape[-1]
            x = x.reshape(bs, npts, 1, dim)
            y = y.reshape(bs, 1, mpts, dim)
            dist = (x - y).norm(dim=-1, keepdim=False)  # (bs, npts, mpts)

        emd_lst = []
        dist_np = dist.cpu().detach().numpy()
        for i in range(bs):
            d_i = dist_np[i]
            r_idx, c_idx = linear_sum_assignment(d_i)
            emd_i = d_i[r_idx, c_idx].mean()
            emd_lst.append(emd_i)
        emd = np.stack(emd_lst).reshape(-1)
        emd_torch = torch.from_numpy(emd).to(x)
    return emd_torch


def EMD_CD(sample_pcs, ref_pcs, batch_size, reduced=True):
    N_sample = sample_pcs.shape[0]
    N_ref = ref_pcs.shape[0]
    assert N_sample == N_ref, "REF:%d SMP:%d" % (N_ref, N_sample)

    cd_lst = []
    emd_lst = []
    iterator = range(0, N_sample, batch_size)

    for b_start in iterator:
        b_end = min(N_sample, b_start + batch_size)
        sample_batch = sample_pcs[b_start:b_end]
        ref_batch = ref_pcs[b_start:b_end]

        cd, dist = CD(sample_batch, ref_batch)
        emd = EMD(sample_batch, ref_batch, dist=dist)
        cd_lst.append(cd)
        emd_lst.append(emd)

    cd = torch.cat(cd_lst)
    emd = torch.cat(emd_lst)
    if reduced:
        cd = cd.mean()
        emd = emd.mean()

    results = {
        'CD': cd,
        'EMD': emd,
    }
    return results