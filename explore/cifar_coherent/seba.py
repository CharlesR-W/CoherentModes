# SPDX-License-Identifier: GPL-3.0-or-later
"""Port of gfroyland/SEBA, by Gary Froyland, Christopher P. Rock,
and Konstantinos Sakellariou.

Source: https://github.com/gfroyland/SEBA (matlab/SEBA.m).
Paper: Froyland, Rock & Sakellariou (2019), "Sparse eigenbasis approximation:
multiple feature extraction across spatiotemporal scales with application to
coherent set identification", Communications in Nonlinear Science and Numerical
Simulation 77:81–107.

Upstream copyright notice, retained verbatim:

SEBA is a collection of MATLAB functions for the computation of
coherent sets in time dependent velocity fields. 

SEBA is copyright by Gary Froyland, Christopher P. Rock, and Konstantinos Sakellariou.

This program is free software; you can redistribute it and/or modify it
under the terms of the GNU General Public License as published by the
Free Software Foundation; either version 3 of the License, or (at your
option) any later version.

This program is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General
Public License for more details.

You should have received a copy of the GNU General Public License along
with this program; if not, see <http://www.gnu.org/licenses/>

Please note that although SEBA is freely available, it is copyrighted
by the authors and may not be sold or included in commercial products
without a license.

If you use SEBA in any program or publication, please acknowledge
its authors by adding a reference to: 

Gary Froyland, Christopher P. Rock, and Konstantinos Sakellariou.
Sparse eigenbasis approximation: multiple feature extraction across spatiotemporal scales with application to coherent set identification. 
Communications in Nonlinear Science and Numerical Simulation, 77:81-107, 2019.
"""
from __future__ import annotations

import torch


def seba_reconstruct(v, rot, perm=None, clip=True):
    """Schema-v2 reconstruction, before/after reliability permutation."""
    z = v.double() @ rot.double()
    s = z.sign() * (z.abs() - .99 / len(v) ** .5).clamp_min(0)
    norms = s.norm(dim=0)
    if bool((norms == 0).any()):
        raise ValueError('SEBA threshold produced an empty column')
    s /= norms
    signs = s.sum(0).sign()
    signs[signs == 0] = 1
    s *= signs
    s /= s.max(0).values
    if perm is not None:
        s = s[:, perm]
    return s.clamp_min(0) if clip else s



def seba(v: torch.Tensor, r: int, max_iter: int = 5000, tol: float = 1e-14,
         return_transform=False):
    """Port of gfroyland/SEBA matlab/SEBA.m (Algorithm 3.1).

    QR, near-constant perturbation, normalized soft threshold, polar update,
    parity, max scaling and reliability sort follow the reference. Returned
    rotation acts on the ORIGINAL input columns (reference uses R transpose).
    We save the rotation that produced the last thresholded S, not its next
    polar update. Tiny reference perturbations are only a numerical tie-break.
    """
    original = v[:, :r].double()
    v, triangular = torch.linalg.qr(original)
    gen = torch.Generator(device=v.device).manual_seed(0)
    for j in range(r):
        if float(v[:, j].max() - v[:, j].min()) < 1e-14:
            v[:, j] += (torch.rand(len(v), generator=gen, device=v.device, dtype=v.dtype) - .5) * 1e-12
    rot = torch.eye(r, dtype=v.dtype, device=v.device)
    for iteration in range(max_iter):
        z = v @ rot
        s = z.sign() * (z.abs() - .99 / len(v) ** .5).clamp_min(0)
        norms = s.norm(dim=0)
        if bool((norms == 0).any()):
            raise ValueError('SEBA threshold produced an empty column')
        s /= norms
        u, _, wt = torch.linalg.svd(v.T @ s)
        new = u @ wt
        if float((new - rot).norm()) <= tol:
            break
        if iteration != max_iter - 1:
            rot = new
    # Translate QR coordinates back to the saved eigenbasis. Reconstruct without
    # the O(1e-12) tie-break so the saved transform is an exact output contract.
    transform = torch.linalg.solve_triangular(triangular, rot, upper=True)
    raw = seba_reconstruct(original, transform, clip=False)
    perm = raw.min(0).values.argsort(descending=True, stable=True)
    result = raw[:, perm].clamp_min(0)
    return (result, transform, perm) if return_transform else result



def seba_labels(s: torch.Tensor, thresh: float = 0.0) -> torch.Tensor:
    """Hard assignment: argmax set if its value > thresh, else -1."""
    m, a = s.max(1)
    a[m <= thresh] = -1
    return a

