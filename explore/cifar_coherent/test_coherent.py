"""Planted check: three groups; sites 0-1 separate A|B|C, site 2 merges B and C.
Block [0,1] should find a 3-way split; block [0,2] should find A vs (B,C) as the
most coherent split (B/C separation is broken by site 2)."""
import torch
from test_seba import test_reference

from coherent import block_modes, centred_cosine_gram, diffusion_operator, knn_dist, seba, seba_labels


def planted(sep, n=60, d=40, seed=0):
    g = torch.Generator().manual_seed(seed)
    centres = torch.randn(3, d, generator=g) * torch.tensor(sep)[:, None]
    lab = torch.arange(3).repeat_interleave(n)
    return centres[lab] + 0.3 * torch.randn(3 * n, d, generator=g), lab


def test_planted():
    x0, lab = planted([3.0, 3.0, 3.0], seed=0)
    x1, _ = planted([3.0, 3.0, 3.0], seed=1)
    xb, _ = planted([3.0, 3.0, 3.0], seed=2)
    x2 = xb.clone()
    x2[lab == 2] = x2[lab == 1][torch.randperm(60)] + 0.01 * torch.randn(60, 40)  # merge B, C
    cs = [centred_cosine_gram(x) for x in (x0, x1, x2)]
    kap = [1.0 / float(knn_dist(c, 10).median()) for c in cs]
    ms = [diffusion_operator(c, k) for c, k in zip(cs, kap)]

    def purity(labels):
        ok = 0
        for s in labels.unique():
            if s < 0:
                continue
            ok += torch.bincount(lab[labels == s], minlength=3).max().item()
        return ok / len(lab)

    vals, vecs = block_modes(ms[:2], 4)
    assert purity(seba_labels(seba(vecs, 3))) > 0.95
    vals2, vecs2 = block_modes(ms, 4)
    # 2 SEBA sets of the merged block are A and (B, C)
    lab2 = seba_labels(seba(vecs2, 2))
    merged = lab.clone(); merged[merged == 2] = 1
    agree = max((lab2 == merged).float().mean(), (lab2 == 1 - merged).float().mean())
    assert agree > 0.95, agree
    # the B/C split is much less coherent with site 2 included
    assert vals2[2] < vals[2] - 0.05, (vals, vals2)
    print("vals [0,1]:", vals.numpy().round(3), " vals [0,2]:", vals2.numpy().round(3))




def test_sinkhorn():
    from coherent import sinkhorn_operator
    torch.manual_seed(7)
    x = torch.cat([torch.randn(90, 12) * .1 + 3, torch.randn(30, 12)])
    c = centred_cosine_gram(x)
    m = sinkhorn_operator(c, 5)
    assert torch.max(torch.abs(m.sum(1) - 1)) < 3e-12
    assert torch.max(torch.abs(m - m.T)) < 1e-14
    val, vec = block_modes([m], 4)
    assert abs(float(val[0]) - 1) < 1e-11
    assert torch.max(torch.abs(vec[:, 0] - 1 / len(x)**.5)) < 1e-10




def test_reconstruction():
    from coherent import sinkhorn_operator, seba_reconstruct
    torch.manual_seed(27)
    x, _ = planted([1., 1., 1.], n=80)
    _, v = block_modes([sinkhorn_operator(centred_cosine_gram(x), 5)], 9)
    v[:, 0] = 1 / len(v)**.5
    for r in (2, 3, 4, 6, 8):
        s, rotation, perm = seba(v, r, return_transform=True)
        scale = v[:, 1:].abs().max(0).values
        quant = torch.round(127 * v[:, 1:] / scale).to(torch.int8)
        decoded = torch.cat([v[:, :1], quant.double() * scale / 127], 1)
        got = seba_reconstruct(decoded[:, :r], rotation.float(), perm)
        agreement = float((seba_labels(s) == seba_labels(got)).double().mean())
        assert agreement >= .99, (r, agreement)
        exact = seba_reconstruct(v[:, :r], rotation, perm)
        assert torch.max(torch.abs(s - exact)) < 1e-12


def test_time_block():
    from coherent import sinkhorn_operator
    torch.manual_seed(11)
    # A persistent binary distinction versus an orthogonal distinction visible
    # only at the first of four checkpoints. No labels enter any kernel.
    n = 160
    a = torch.arange(n) % 2 * 2 - 1
    b = torch.arange(n) // 2 % 2 * 2 - 1
    xs = [torch.stack([a * 2., b * (3. if t == 0 else .03)], 1)
          + .06 * torch.randn(n, 2) for t in range(4)]
    ms = [sinkhorn_operator(centred_cosine_gram(x), 2.) for x in xs]
    _, v = block_modes(ms, 3)
    corr_a = abs(float(torch.dot(v[:, 1], a.double()) / n**.5))
    corr_b = abs(float(torch.dot(v[:, 1], b.double()) / n**.5))
    assert corr_a > .98 and corr_b < .05, (corr_a, corr_b)


def test_near_identity_saved_operator():
    """Reviewer regression: global-30 in a realistic high-dimensional cell."""
    import numpy as np
    import tempfile
    from pathlib import Path
    from coherent import sinkhorn_operator
    from schema_v2 import block_record
    torch.manual_seed(12)
    x=torch.randn(2000,256,dtype=torch.float64)
    m=sinkhorn_operator(centred_cosine_gram(x),30)
    with tempfile.TemporaryDirectory() as directory:
        path=Path(directory)/'operators.npy'
        np.save(path,m.numpy())
        loaded=torch.from_numpy(np.load(path))
    assert loaded.dtype==torch.float64 and torch.equal(loaded,m)
    rec,v=block_record([loaded],0,0)
    assert rec['vals'][0]==1
    assert torch.max(torch.abs(v[:,0]-1/2000**.5))<1e-15
    assert torch.max(torch.abs(v[:,1:].sum(0)))<1e-12
    assert torch.max(torch.abs(v.T@v-torch.eye(9)))<1e-12
    assert torch.max(torch.abs(m@v-v*torch.from_numpy(rec['vals']).double()))<1e-8
    # Reconstruct from EXACTLY the exported int16/f32 arrays, not in-memory
    # rotations/scales. This is the reviewer counterexample that defeated int8.
    from coherent import seba_reconstruct
    decoded=np.concatenate([np.full((2000,1),1/2000**.5),rec['vecs'].T.astype('f8')*rec['scale']/32767],1)
    for ri,r in enumerate((2,3,4,6,8)):
        reconstructed=seba_reconstruct(torch.from_numpy(decoded[:,:r]),torch.from_numpy(rec['seba_rot'][ri,:r,:r]),
                                       torch.from_numpy(rec['seba_perm'][ri,:r].astype('i8')))
        agreement=np.mean(seba_labels(reconstructed).numpy()==rec['seba_label'][ri])
        assert agreement>=.99,(r,agreement)
    p=sinkhorn_operator(centred_cosine_gram(x),30,remove_diagonal=True)
    assert torch.max(torch.abs(p.diag()))==0 and torch.max(torch.abs(p.sum(1)-1))<3e-12
    _,vv=block_modes([p*.5+torch.eye(2000,dtype=torch.float64)*.5],9,shared_stationary=True)
    assert torch.max(torch.abs(vv[:,0]-1/2000**.5))<1e-15


def test_zero_diagonal_fixed_lazy():
    from coherent import sinkhorn_operator,fixed_lazy_operator
    torch.manual_seed(78)
    x=torch.cat([torch.randn(90,12)*.1+3,torch.randn(30,12)]).double()
    # Unequal density and nearly disconnected two-group geometry.
    groups=torch.cat([torch.randn(40,8)*.02+2,torch.randn(40,8)*.02-2]).double()
    for data in (x,groups):
        p=sinkhorn_operator(centred_cosine_gram(data),10,remove_diagonal=True)
        eig=torch.linalg.eigvalsh(p)
        assert float((p.sum(1)-1).abs().max())<3e-12
        assert torch.equal(p.diag(),torch.zeros(len(p),dtype=p.dtype))
        assert eig[0]>=-1-1e-12 and abs(float(eig[-1])-1)<1e-11
        if data is groups:assert eig[-2]>.999
    # Four square vertices: with diagonal removed, almost all kernel mass
    # crosses between x-axis and y-axis vertices, giving a near-bipartite graph.
    square=torch.tensor([[1.,0.],[-1.,0.],[0.,1.],[0.,-1.]],dtype=torch.float64)
    c=square@square.T
    p=sinkhorn_operator(c,20,remove_diagonal=True)
    eig=torch.linalg.eigvalsh(p)
    assert eig[0]<-.99999 and abs(float(eig[-1])-1)<1e-12
    m=fixed_lazy_operator(c,20);q=m@m
    alternating=torch.tensor([1.,1.,-1.,-1.],dtype=torch.float64)/2
    assert (q@alternating).norm()<1e-12
    singular=torch.linalg.svdvals(q-torch.ones_like(q)/4)
    assert .24999<float(singular[0])<.25001
    # Fixed self-mass equalises diffusion mass, not mixing rates or eigengaps.
    high=centred_cosine_gram(torch.randn(200,256,dtype=torch.float64))
    old=[sinkhorn_operator(high,k) for k in (30,5)]
    old_mass=[float(1-a.diag().mean()) for a in old]
    assert old_mass[0]<old_mass[1]*1e-5
    new=[fixed_lazy_operator(high,k) for k in (30,5)]
    contributions=[float(1-a.diag().mean()) for a in new]
    assert contributions==[.5,.5]
    avg=sum(new)/2
    assert abs(float((torch.eye(200)-avg).trace())-100)<1e-10


def test_final_kernel_bank():
    from sweep_math import final_kernel_bank
    from coherent import symmetric_balance
    torch.manual_seed(91)
    bank=final_kernel_bank(centred_cosine_gram(torch.randn(400,24,dtype=torch.float64)))
    assert len(bank)==5 and bank[0][1]['kappa'] is None
    assert all(info['self_weight'] is None for _,info in bank[:3])
    assert abs(bank[4][1]['nn_mass']-.5)<1e-9
    for logk,info in bank:
        p,diag=symmetric_balance(logk,return_info=True)
        assert float((p.sum(1)-1).abs().max())<2e-12
        assert torch.equal(p.diag(),torch.zeros(len(p),dtype=p.dtype))
        assert 0<info['nn_mass']<1
        m=p/2;m.diagonal().add_(.5)
        vals,v=block_modes([m],9,shared_stationary=True)
        assert vals[0]==1 and float((v[:,0]-1/400**.5).abs().max())<1e-14
        assert diag['row_error']<2e-12


if __name__ == "__main__":
    torch.set_num_threads(1)
    test_planted()
    test_sinkhorn()
    test_reference()
    test_reconstruction()
    test_time_block()
    test_near_identity_saved_operator()
    test_zero_diagonal_fixed_lazy()
    test_final_kernel_bank()
    print("all coherent checks passed")

