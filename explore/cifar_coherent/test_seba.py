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
import torch
from seba import seba


def reference_seba(v):
    """Independent literal NumPy translation of authors' MATLAB SEBA.m.

    Reference: https://github.com/gfroyland/SEBA/blob/master/matlab/SEBA.m
    Fixed comparison inputs have no constant columns, avoiding RNG differences.
    """
    import numpy as np
    v, _ = np.linalg.qr(v)
    n, r = v.shape
    rnew = np.eye(r)
    rotation = np.zeros((r, r))
    iteration = 0
    while np.linalg.norm(rnew - rotation) > 1e-14 and iteration < 5000:
        iteration += 1
        rotation = rnew
        z = v @ rotation.T
        s = np.sign(z) * np.maximum(np.abs(z) - .99 / np.sqrt(n), 0)
        s /= np.linalg.norm(s, axis=0)
        p, _, qt = np.linalg.svd(s.T @ v, full_matrices=False)
        rnew = p @ qt
    s *= np.sign(s.sum(axis=0))
    s /= s.max(axis=0)
    return np.maximum(s[:, np.argsort(-s.min(axis=0), kind='stable')], 0)



def test_reference():
    import numpy as np
    torch.manual_seed(18)
    v, _ = torch.linalg.qr(torch.randn(137, 6, dtype=torch.float64))
    for r in (2, 3, 4, 6):
        expected = reference_seba(v[:, :r].numpy())
        actual = seba(v, r).numpy()
        # Reliability can tie at exactly zero: compare after optimal assignment.
        from itertools import permutations
        error = min(np.max(np.abs(actual - expected[:, p])) for p in permutations(range(r)))
        assert error < 2e-7, error



if __name__ == '__main__':
    torch.set_num_threads(1)
    test_reference()
    print('SEBA reference checks passed')
