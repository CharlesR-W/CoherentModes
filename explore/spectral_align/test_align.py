import sys
from pathlib import Path
import numpy as np
import torch
import align

torch.set_num_threads(1)


def basis(n, seed):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, n-1, generator=g, dtype=torch.float64)
    return torch.linalg.qr(x-x.mean(0)).Q


def test_identity_and_chance():
    n, k = 45, 6
    a = basis(n, 0)
    e = torch.linspace(1, .1, n, dtype=torch.float64)
    assert align.projector_distances(a, a, [k]).abs().max() < 1e-12
    assert align.heat_distances(e, a, e, a).abs().max() < 1e-12
    vals = [float(align.projector_distances(a, basis(n, s), [k])) for s in range(1, 101)]
    assert abs(np.mean(vals)-(1-k/(n-1))) < .015


def test_planted_repeated_and_separated():
    n, r = 37, 4
    u = basis(n, 0)
    base = torch.cat([torch.ones(1), torch.full((r,), .85), torch.linspace(.3, .1, n-r-1)]).double()
    j = torch.ones(n, n, dtype=torch.float64)/n
    def operator(e, q):
        return j+(q*e[1:])@q.T
    repeated, separated, complements = [], [], []
    for seed in range(80):
        g = torch.Generator().manual_seed(seed+111)
        # Isotropic, independent symmetric perturbations split the repeated signal eigenvalue.
        modes = []
        modes_sep = []
        for _ in range(2):
            noise = torch.randn(n-1, n-1, generator=g, dtype=torch.float64)
            noise = (noise+noise.T)*1e-6
            m = operator(base,u)+u@noise@u.T
            modes.append(align.spectrum(m, n-1)[1])
            sep = base.clone(); sep[1:r+1] = torch.tensor([.9,.8,.7,.6])
            modes_sep.append(align.spectrum(operator(sep,u)+u@noise@u.T,n-1)[1])
        repeated.append(align.projector_distances(*modes, list(range(1,r+1))).numpy())
        separated.append(align.projector_distances(*modes_sep,list(range(1,r+1))).numpy())
        qa = torch.linalg.qr(torch.randn(n-r-1,n-r-1,generator=g,dtype=torch.float64)).Q
        qb = torch.linalg.qr(torch.randn(n-r-1,n-r-1,generator=g,dtype=torch.float64)).Q
        a = torch.cat([u[:,:r],u[:,r:]@qa],1)
        b = torch.cat([u[:,:r],u[:,r:]@qb],1)
        complements.append(align.projector_distances(a,b,[6,10,20]).numpy())
    avg = np.mean(repeated,0)
    np.testing.assert_allclose(avg[:r-1],1-np.arange(1,r)/r,atol=.055)
    assert avg[-1] < 1e-8
    assert np.max(separated) < 1e-7
    k = np.array([6,10,20]); expected = 1-(r+(k-r)**2/(n-1-r))/k
    np.testing.assert_allclose(np.mean(complements,0),expected,atol=.015)


def test_dtw():
    # Rotating operator path, B duplicates A's interior positions.
    q = basis(18,3)
    positions_a = np.array([0,.2,.5,.8,1.])
    positions_b = np.array([0,.2,.2,.5,.8,.8,1.])
    def modes(t):
        return (q[:,0]*np.cos(t)+q[:,1]*np.sin(t))[:,None]
    d = np.array([[float(align.projector_distances(modes(a),modes(b),[1])) for b in positions_b] for a in positions_a])
    fit = align.dtw_alignment(d)
    assert fit['cost'] < 1e-12
    assert all(positions_a[i] == positions_b[j] for i,j in fit['path'])
    fit = align.dtw_alignment([[1,.1],[1,1]])
    assert fit['path'] == [[0,0],[0,1],[1,1]]
    assert abs(fit['cost']-.7) < 1e-12
    fit = align.dtw_alignment(np.full((4,6),2.))
    assert abs(fit['cost']-2.) < 1e-12


def test_chunked_and_heat():
    n=17
    u=torch.stack([basis(n,i) for i in range(5)])
    e=torch.stack([torch.cat([torch.ones(1),torch.linspace(.9-.02*i,.1,n-1)]).double() for i in range(5)])
    p,s=align.pairwise_spectral(e,u,[1,3,8],chunk=2)
    kernels=[align.heat_kernel(e[i],u[i],4) for i in range(5)]
    h=align.streamed_kernel_gram(5,n,lambda i,a,b:kernels[i][a:b],dtype=torch.float64,row_chunk=3)
    for i in range(5):
        for j in range(5):
            torch.testing.assert_close(p[i,j],align.projector_distances(u[i],u[j],[1,3,8]),atol=1e-12,rtol=1e-12)
            torch.testing.assert_close(s[i,j],align.eigenvalue_distances(e[i],e[j],[1,3,8]))
            torch.testing.assert_close(h[i,j],align.heat_distances(e[i],u[i],e[j],u[j],[4])[0],atol=1e-12,rtol=1e-12)
            ma=align.heat_kernel(e[i],u[i],1)+1/n
            mb=align.heat_kernel(e[j],u[j],1)+1/n
            dense=align.cosine_distance(torch.linalg.matrix_power(ma,4)-1/n,torch.linalg.matrix_power(mb,4)-1/n)
            torch.testing.assert_close(h[i,j],dense,atol=1e-12,rtol=1e-12)


def test_small_spectrum_heat_no_underflow():
    n=17
    ea=torch.cat([torch.ones(1),torch.linspace(.60,.1,n-1)]).double()
    eb=torch.cat([torch.ones(1),torch.linspace(.57,.1,n-1)]).double()
    ua,ub=basis(n,1),basis(n,2)
    ka=align.normalized_heat_kernel(ea,ua.float(),256)
    kb=align.normalized_heat_kernel(eb,ub.float(),256)
    assert abs(float(ka.square().sum())-1)<1e-5
    direct=align.cosine_distance(align.heat_kernel(ea,ua,256),align.heat_kernel(eb,ub,256))
    streamed=align.streamed_kernel_gram(2,n,None,load_rows=lambda a,b:torch.stack([ka[a:b],kb[a:b]]))
    assert torch.isfinite(streamed).all()
    assert abs(float(streamed[0,1])-float(direct))<1e-5


def test_model_equivalence():
    sys.path.insert(0,str(Path.home()/'code/ModelZoo/src'))
    from zoo.models import ResNet20
    from nets import build_resnet
    torch.manual_seed(4); a=ResNet20('relu',10).eval()
    torch.manual_seed(4); b=build_resnet(3).eval()
    assert list(a.state_dict())==list(b.state_dict())
    b.load_state_dict(a.state_dict())
    x=torch.randn(2,3,32,32)
    torch.testing.assert_close(a(x),b(x),rtol=0,atol=0)
