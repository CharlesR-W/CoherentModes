"""Assert the numerical identities and counterexamples in draft B.40--B.53.

Run from the repository root:
    uv run --with numpy python -I checks/theorems_b3_check.py

Nonzero reference values are checked numerically, not merely printed. Exact-zero
identities and roundoff diagnostics are checked with absolute tolerances rather
than pinned to BLAS-dependent last bits. No file I/O or optional dependencies.
"""

import numpy as np


def check_reference(actual, expected, *, rtol=5e-8, atol=1e-12):
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)

rng=np.random.default_rng(614)
N=24
X=rng.normal(size=(N,3))
A=np.tanh(X@rng.normal(size=(3,3)))
H=np.tanh(X@rng.normal(size=(3,4)))
V=rng.normal(size=(4,5))/2
W=rng.normal(size=(5,3))/2
S=np.linalg.pinv(A)@H
D=A@S-H

def logits(h): return np.tanh(h@V)@W

def logsoftmax(z):
    z=z-z.max(axis=1,keepdims=True)
    return z-np.log(np.exp(z).sum(axis=1,keepdims=True))
z=logits(H); lp=logsoftmax(z); p=np.exp(lp)
q=0.
for x in range(N):
    J=W.T@np.diag(1-np.tanh(H[x]@V)**2)@V.T
    F=np.diag(p[x])-np.outer(p[x],p[x])
    d=J@D[x]
    q+=d@F@d
pred=q/(2*N)
check_reference(np.linalg.norm(D), 3.75714159887)
check_reference(pred, 0.00915415565648)
print('nonlinear network: N=%d, ||residual||F=%.12g, KL quadratic coefficient=%.12g'%(N,np.linalg.norm(D),pred))
ratios=[]
actual_kls=[]
quadratic_kls=[]
for t in [.1,.03,.01,.003,.001]:
    kl=np.sum(p*(lp-logsoftmax(logits(H+t*D))))/N
    ratio=kl/(t*t*pred); ratios.append(ratio)
    actual_kls.append(kl)
    quadratic_kls.append(t*t*pred)
    print('t=%.3g KL=%.12g quadratic=%.12g ratio=%.10f'%(t,kl,t*t*pred,ratio))
check_reference(ratios, [0.9857628505, 0.9958458141, 0.9986276011,
                         0.9995896098, 0.9998633324], rtol=2e-7)
check_reference(actual_kls, [9.02382657341e-5, 8.20451483312e-6,
                            9.14159250356e-7, 8.23535899278e-8,
                            9.15290458037e-9], atol=1e-14, rtol=2e-7)
check_reference(quadratic_kls, [9.15415565648e-5, 8.23874009083e-6,
                               9.15415565648e-7, 8.23874009083e-8,
                               9.15415565648e-9], atol=1e-14)
assert abs(ratios[-1]-1)<.002
assert abs(ratios[-1]-1)<abs(ratios[0]-1)/30
# Any orthonormal basis: residual = sum Pi residual.
Psi=np.linalg.qr(rng.normal(size=(N,N)))[0]
parts=[Psi[:,i:i+1]@(Psi[:,i:i+1].T@D) for i in range(N)]
err=np.linalg.norm(sum(parts)-D)
print('orthogonal block reconstruction error=%.3g'%err)
assert err<1e-12
# Ideal binary readability, uniform isotropic Gaussian-output Fisher.
N=8
Psi=np.linalg.qr(rng.normal(size=(N,N)))[0]
a=np.arange(1,N+1)/N
H=Psi@np.diag(a)
keep=np.array([0,2,4])
A=Psi[:,keep]
T=A@A.T; M=np.eye(N)-T
E=M@H
unread=np.array([np.linalg.norm(M@Psi[:,i])**2 for i in range(N)])
gamma=1.7
q=gamma*np.linalg.norm(E)**2
sens2=N*gamma*a*a
pred=np.sum(sens2*unread)/N
print('ideal binary q=%.12g weighted prediction=%.12g absolute error=%.3g'%(q,pred,abs(q-pred)))
check_reference([q, pred], [4.4890625, 4.4890625])
assert np.allclose(q,pred,rtol=1e-12,atol=1e-12)
# Leakage/cancellation invalidates a scalar unreadability formula.
Psi=np.eye(2); H=np.ones((2,1)); A=H.copy(); M=np.eye(2)-A@np.linalg.pinv(A)
q=np.linalg.norm(M@H)**2
naive=sum(np.linalg.norm(M@Psi[:,i])**2 for i in range(2))
print('mixing counterexample: exact residual energy=%.3g naive weighted unreadability=%.12g'%(q,naive))
assert q<1e-28 and abs(naive-1)<1e-12
# Reviewed revisions: genuine spectral blocks, approximate commutation,
# site-homogeneous anisotropic Fisher, and distinguishable ideal stitches.
def sinkhorn(K):
    v=np.ones(len(K))/np.sqrt(K.sum()/len(K))
    for it in range(50000):
        vn=np.sqrt(v/(K@v))
        if np.max(np.abs(vn-v))<1e-15:
            v=vn
            break
        v=vn
    P=v[:,None]*K*v[None,:]
    assert np.max(np.abs(P.sum(axis=1)-1))<1e-12
    return P

def operator(H,beta):
    U=H/np.linalg.norm(H,axis=1,keepdims=True)
    return sinkhorn(np.exp(beta*(U@U.T)))

def spectral_blocks(P,tol=1e-8):
    w,U=np.linalg.eigh(P)
    order=np.argsort(-w); w=w[order]; U=U[:,order]
    result=[]; i=0
    while i<len(w):
        j=i+1
        while j<len(w) and abs(w[j]-w[i])<tol:
            j+=1
        result.append((w[i:j],U[:,i:j]))
        i=j
    return result

def projection(H): return H@np.linalg.pinv(H)

rng=np.random.default_rng(11)
H=rng.normal(size=(30,4)); blocks=spectral_blocks(operator(H,4))
assert all(Psi.shape[1]==1 for _,Psi in blocks)
M=np.eye(30)-projection(H)
residual_energy=np.linalg.norm(M@H)**2
naive=sum(np.linalg.norm(M@Psi)**2*np.linalg.norm(Psi.T@H)**2 for _,Psi in blocks)
print('spectral self-stitch: residual energy=%.12g naive predictor=%.12g'%(residual_energy,naive))
# The theoretical residual is exactly zero; its printed 2.60e-29 is roundoff.
assert residual_energy<1e-25
check_reference(naive, 39.3566151378)
# Exact two-site spectral counterexamples.
for source,scale,expected_energy,expected_naive in [([2.,1.],1.,0.,.9),([1.,2.],np.sqrt(10)/3,2.,1.)]:
    H2=scale*np.array([[2.],[1.]])
    A2=np.array(source)[:,None]; M2=np.eye(2)-projection(A2)
    B2=spectral_blocks(operator(H2,4))
    energy=np.linalg.norm(M2@H2)**2
    diagonal=sum(np.linalg.norm(M2@Psi)**2*np.linalg.norm(Psi.T@H2)**2 for _,Psi in B2)
    print('two-site spectral example: energy=%.12g diagonal=%.12g'%(energy,diagonal))
    assert abs(energy-expected_energy)<1e-12 and abs(diagonal-expected_naive)<1e-12

N=24; theta=2*np.pi*np.arange(N)/N
Z=np.c_[np.cos(theta),np.sin(theta),.7*np.cos(2*theta),.7*np.sin(2*theta)]
a=rng.uniform(.5,2,size=N); H=a[:,None]*Z
blocks=spectral_blocks(operator(H,3))
assert len(blocks)==13
assert sorted(Psi.shape[1] for _,Psi in blocks)==[1,1]+[2]*11
check_reference(np.sum(Z*Z,axis=1), np.full(N,1.49))
min_gap=np.min(np.diff(sorted(w[0] for w,_ in blocks)))
check_reference(min_gap, 0.00015599257286)
A=np.c_[np.cos(theta),np.cos(2*theta)]
rng.normal(size=(N,2))  # Preserve the reviewer's deterministic Gaussian-head draw.
T=projection(A); M=np.eye(N)-T; E=M@H
comm=max(np.linalg.norm(T@Psi@Psi.T-Psi@Psi.T@T,2) for _,Psi in blocks)
assert comm<1e-12
W=np.diag([1.,1.,2.,2.])+.3*rng.normal(size=(4,4))
g=W.T@W

def block_bounds(E,M,gfun,with_commutator=False,T=None):
    lo=hi=0.; diag=0.; q=sum(E[x]@gfun(x)@E[x] for x in range(N)); cs=[]
    for _,Psi in blocks:
        Q=sum(np.kron(np.outer(Psi[x],Psi[x]),gfun(x)) for x in range(N))
        gam=np.linalg.eigvalsh(Q)
        L=Psi.T@H; ell=np.linalg.eigvalsh(L@L.T)
        ell=np.maximum(ell,0)
        U=np.linalg.norm(M@Psi)**2
        c=0.
        if with_commutator:
            Pi=Psi@Psi.T; correction=(T@Pi-Pi@T)@H
            c=np.linalg.norm(correction)
            F=M@Psi@L; Eb=Pi@E
            assert abs(np.linalg.norm(Eb)-np.linalg.norm(F))<=c+1e-12
            assert c<=np.linalg.norm(T@Pi-Pi@T,2)*np.linalg.norm(H)+1e-12
            cs.append(c)
        lo+=max(gam[0],0)*max(np.sqrt(ell[0]*U)-c,0)**2
        hi+=max(gam[-1],0)*(np.sqrt(ell[-1]*U)+c)**2
        Eb=Psi@Psi.T@E
        diag+=sum(Eb[x]@gfun(x)@Eb[x] for x in range(N))
    return lo,q,hi,diag,cs
lo,q,hi,diag,_=block_bounds(E,M,lambda x:g)
print('circle: min gap=%.12g max commutator=%.12g'%(min_gap,comm))
print('circle Gaussian: lower=%.12g q=%.12g upper=%.12g width=%.12g'%(lo,q,hi,hi-lo))
check_reference([lo,q,hi,hi-lo], [28.8021689113,98.690860218,
                                  249.710801551,220.90863264])
assert lo<=q<=hi and abs(diag-q)<1e-10
V=rng.normal(size=(4,5))
def gsoft(x):
    z=H[x]@V; p=np.exp(z-z.max()); p/=p.sum()
    return V@(np.diag(p)-np.outer(p,p))@V.T
lo,q,hi,diag,_=block_bounds(E,M,gsoft)
eta=abs(q-diag)/diag
print('circle softmax: eta=%.12g corrected lower=%.12g q=%.12g upper=%.12g width=%.12g'%(eta,(1-eta)*lo,q,(1+eta)*hi,(1+eta)*hi-(1-eta)*lo))
check_reference([eta,(1-eta)*lo,q,(1+eta)*hi,(1+eta)*hi-(1-eta)*lo],
                [0.266100238141,0.698942839274,11.9891342564,
                 46.9780311363,46.279088297])
assert eta<1 and (1-eta)*lo<=q<=(1+eta)*hi
# A single perturbed source/projector is used on BOTH sides of each commutator.
Ap=A+1e-3*rng.normal(size=A.shape)
Tp=projection(Ap); Mp=np.eye(N)-Tp; Ep=Mp@H
lo,q,hi,diag,cs=block_bounds(Ep,Mp,lambda x:g,True,Tp)
comm_p=max(np.linalg.norm(Tp@Psi@Psi.T-Psi@Psi.T@Tp,2) for _,Psi in blocks)
print('circle perturbed source: max commutator=%.12g max correction=%.12g interval=[%.12g,%.12g] q=%.12g'%(comm_p,max(cs),lo,hi,q))
check_reference([comm_p,max(cs),lo,hi,q],
                [0.00137234126938,0.00509480410441,28.7086783711,
                 250.3877768,98.6903500313])
assert comm_p>1e-5 and lo<=q<=hi and abs(diag-q)<1e-10
# Binary spectral readability under a non-isotropic constant form.
A=np.c_[np.cos(theta),np.sin(theta)]; T=projection(A); M=np.eye(N)-T; E=M@H
g=np.diag([1.5,1.5,.4,.4]); q=np.einsum('ni,ij,nj->',E,g,E)
pred=0.
for _,Psi in blocks:
    Pi=Psi@Psi.T
    assert min(np.linalg.norm(T@Pi-Pi),np.linalg.norm(T@Pi))<1e-10
    L=Psi.T@H; u=np.linalg.norm(M@Psi)**2/Psi.shape[1]
    s2=N*np.trace(L@g@L.T); pred+=s2*u/N
print('circle binary anisotropic: q=%.12g predictor=%.12g'%(q,pred))
check_reference([q,pred], [14.6650169533,14.6650169533])
assert abs(q-pred)<1e-10
# Nonlinear null term agrees on both sites but not at their stitched mean.
H=np.array([[1.,0,1],[0,1,1]])
A=np.ones((2,1)); Hs=A@np.linalg.pinv(A)@H

def za(h): return np.c_[h[:,0]-h[:,1],h[:,2]]
def zb(h):
    d=h[:,0]-h[:,1]
    return np.c_[d+d*d-1,h[:,2]]
assert np.allclose(za(H),zb(H))
lp=logsoftmax(za(H)); p=np.exp(lp)
kla=np.sum(p*(lp-logsoftmax(za(Hs))))/2
klb=np.sum(p*(lp-logsoftmax(zb(Hs))))/2
print('distinguishing full stitch: KL_A=%.12g KL_B=%.12g'%(kla,klb))
check_reference([kla,klb], [0.0936226307057,0.216890415242])
assert klb>kla+.1

# Quoted B.49 evaluation budgets and B.50's informative-bound threshold.
alpha=.05; epsilon=.1; k=1
hoeffding=lambda squared_range: int(np.ceil(squared_range**2/(2*epsilon**2)*np.log(2*k/alpha)))
bernstein=lambda squared_range,variance: int(np.ceil(2*(variance+squared_range*epsilon/3)/epsilon**2*np.log(2*k/alpha)))
assert hoeffding(4)==2952
assert hoeffding(100)==1844440
assert bernstein(100,100)==76237
activation_threshold=1/(2*10*(1+2*np.sqrt(10000)))
check_reference(activation_threshold,0.000248756218905)
print('sample budgets: Hoeffding=2952, localized Hoeffding=1844440, localized Bernstein=76237')
print('all B3 checks passed')
