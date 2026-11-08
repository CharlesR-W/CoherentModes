# Asserted finite-data checks for theorems/drafts/A1-exactness.md.
# Run from the repository root:
#   uv run --with numpy python -I checks/theorems_a1_check.py
# Includes counterexamples and regressions requested by the Sol and Opus reviews.
import itertools
import numpy as np

rng = np.random.default_rng(1729)
def sinkhorn(K):
    v = np.ones(len(K)) / np.sqrt(K.sum(axis=1))
    for it in range(20000):
        v = np.sqrt(v / (K @ v))
        P = K * v[:,None] * v[None,:]
        if np.max(np.abs(P.sum(axis=1)-1)) < 2e-14:
            return P, v
    raise AssertionError('scaling did not converge')
def unit(Z):
    return Z / np.linalg.norm(Z,axis=1,keepdims=True)
def R(b):
    return np.expm1(1.5*b)**2 + b*np.expm1(3*b) + np.exp(3*b)*(np.expm1(b)-b)
def align(A,B):
    return np.sum(A*B)/(np.linalg.norm(A)*np.linalg.norm(B))
N,d=12,4
e=np.ones(N)/np.sqrt(N); C=np.eye(N)-np.outer(e,e)
Z1=unit(rng.normal(size=(N,d))); Z2=unit(rng.normal(size=(N,d)))
G1=Z1@Z1.T; G2=Z2@Z2.T
K1=np.exp(1.7*G1); K2=np.exp(1.7*G2)
P1,v1=sinkhorn(K1); P2,v2=sinkhorn(K2)
assert np.allclose(v1*(K1@v1),1)
Ps,vs=sinkhorn(13*K1)
assert np.allclose(Ps,P1) and np.allclose(vs,v1/np.sqrt(13))
assert np.linalg.eigvalsh(P1).min()>=-1e-13
assert np.allclose(P1@e,e) and np.allclose(P1,P1.T)
# Convex gradient and Hessian used for existence proof.
x=np.log(v1); h=rng.normal(size=N); step=1e-5
fun=lambda x: .5*np.exp(x)@K1@np.exp(x)-x.sum()
D2=.5*np.sum(P1*(h[:,None]+h[None,:])**2)
D2fd=(fun(x+step*h)-2*fun(x)+fun(x-step*h))/step**2
assert np.isclose(D2,D2fd,rtol=2e-4)
# Nonnegative symmetric positive-diagonal extension, reducible case.
Pr,_=sinkhorn(np.diag([1.,2.,3.]))
assert np.allclose(Pr,np.eye(3))
# Power and counting-measure diffusion identity.
l,V=np.linalg.eigh(P1)
for t in [0,1,2,5]:
    Pt=np.linalg.matrix_power(P1,t)
    rhs=np.sum(l**(2*t)*(V[2,:]-V[7,:])**2)
    assert np.isclose(np.linalg.norm(Pt[:,2]-Pt[:,7])**2,rhs)
    assert np.allclose(Pt.sum(1),1)
Q=P2@P1
assert abs(np.linalg.svd(Q,compute_uv=False)[0]-1)<1e-12
assert np.allclose(C@Q@C,Q-np.outer(e,e))
assert np.linalg.norm(Q@P2,2)<=1+1e-12
D1=K1.sum(1); D2=K2.sum(1)
S1=K1/np.sqrt(D1[:,None]*D1[None,:]); S2=K2/np.sqrt(D2[:,None]*D2[None,:])
assert np.linalg.norm(S2@S1,2)<1-1e-7
assert abs(np.linalg.norm(S1@S1,2)-1)<1e-12
f=rng.normal(size=N); f/=np.linalg.norm(f)
a=1-np.linalg.norm(P1@f)**2; b=np.linalg.norm(P1@f)**2-np.linalg.norm(Q@f)**2
assert np.isclose(a+b,1-np.linalg.norm(Q@f)**2) and min(a,b)>=-1e-13
assert np.linalg.norm((np.eye(N)-P1)@f)**2<=a+1e-12
# Balanced set identity for all nontrivial sizes; SVD relaxation and minmax on singular subspaces.
Uc,sc,Vtc=np.linalg.svd(C@Q@C)
for m in range(1,N):
    A=np.zeros(N); B=np.zeros(N); A[:m]=1; B[-m:]=1; alpha=m/N
    fa=(A-alpha)/np.sqrt(N*alpha*(1-alpha)); fb=(B-alpha)/np.sqrt(N*alpha*(1-alpha))
    score=B@Q@A/m+(1-B)@Q@(1-A)/(N-m)-1
    assert np.isclose(score,fb@Q@fa) and score<=sc[0]+1e-12
assert np.isclose(Uc[:,0]@Q@Vtc[0],sc[0])
for k in [1,3,7]:
    S=Vtc[:k].T
    assert np.isclose(np.linalg.eigvalsh(S.T@Q.T@Q@S)[0],sc[k-1]**2)
# Exact reconstruction, indicator, PCA loadings.
Psi=np.linalg.qr(rng.normal(size=(N,N)))[0]; F=rng.normal(size=(N,5)); r=4
A=F.T@Psi; Pi=Psi[:,:r]@Psi[:,:r].T
assert np.allclose(Psi@A.T,F)
assert np.isclose(np.linalg.norm(F-Pi@F)**2,np.linalg.norm(A[:,r:])**2)
ind=np.zeros(N); ind[:4]=.5; load=F.T@ind
assert np.allclose(np.outer(ind,load)[:4],F[:4].mean(0))
U,sv,Vt=np.linalg.svd(C@F,full_matrices=False)
assert np.allclose((C@F).T@U,Vt.T*sv)
# Gradient: multi-slot chain rule, block invariance, ranks, tail bound, frozen finite update.
s,p,q=3,5,4
Us=[rng.normal(size=(N,p)) for _ in range(s)]; Ys=[rng.normal(size=(N,q)) for _ in range(s)]
W=rng.normal(size=(q,p)); Ds=[u@W.T-y for u,y in zip(Us,Ys)]
grad=sum(de.T@u for de,u in zip(Ds,Us))
pieces=[sum(np.outer(de.T@Psi[:,i],u.T@Psi[:,i]) for de,u in zip(Ds,Us)) for i in range(N)]
assert np.allclose(sum(pieces),grad)
assert all(np.linalg.matrix_rank(piece,tol=1e-9)<=s for piece in pieces)
GB=sum(de.T@Pi@u for de,u in zip(Ds,Us))
Rot=np.linalg.qr(rng.normal(size=(r,r)))[0]; Psir=Psi[:,:r]@Rot
assert np.allclose(GB,sum(de.T@Psir@Psir.T@u for de,u in zip(Ds,Us)))
E=np.eye(N)-Pi
bound=np.sqrt(sum(np.linalg.norm(E@de)**2 for de in Ds)*sum(np.linalg.norm(E@u)**2 for u in Us))
assert np.linalg.norm(grad-GB)<=bound+1e-12
loss=lambda W: .5*sum(np.linalg.norm(u@W.T-y)**2 for u,y in zip(Us,Ys))
eps=1e-5
num=np.empty_like(W)
for i in range(q):
    for j in range(p):
        dw=np.zeros_like(W); dw[i,j]=eps
        num[i,j]=(loss(W+dw)-loss(W-dw))/(2*eps)
assert np.allclose(num,grad,rtol=1e-7,atol=1e-7)
dW=rng.normal(size=W.shape)
assert np.allclose(Us[0]@(W+dW).T-Us[0]@W.T,Psi@(dW@(Us[0].T@Psi)).T)
# Probe variances exactly by Rademacher enumeration and degree-four-exact Gaussian quadrature.
Jx=rng.normal(size=(3,4)); Jy=rng.normal(size=(3,4)); M=Jx.T@Jy; Sym=(M+M.T)/2
for name,nodes,weights,mu4 in [('rad',[-1,1],[.5,.5],1),('gauss',[-np.sqrt(3),0,np.sqrt(3)],[1/6,2/3,1/6],3)]:
    ids=np.array(list(itertools.product(range(len(nodes)),repeat=4)))
    probes=np.array(nodes)[ids]; probs=np.array(weights)[ids].prod(1)
    vals=np.einsum('bi,ij,bj->b',probes,M,probes)
    mean=probs@vals; var=probs@(vals-mean)**2
    expected=2*np.linalg.norm(Sym)**2+(mu4-3)*np.sum(np.diag(Sym)**2)
    assert np.isclose(mean,np.trace(M)) and np.isclose(var,expected)
    vals2=(vals[:,None]+vals[None,:])/2
    assert np.isclose(np.sum(probs[:,None]*probs[None,:]*(vals2-mean)**2),expected/2)
Jbank=rng.normal(size=(N,3,4)); probes=rng.normal(size=(7,4))
sketch=np.einsum('xqi,ri->xrq',Jbank,probes).reshape(N,-1)/np.sqrt(7)
assert np.linalg.eigvalsh(sketch@sketch.T).min()>-1e-12
# Explicit low-temperature remainder across dimensions/configurations and eigenspace/CKA limits.
maxratio=0
for n in [2,5,12,30]:
    Z=unit(rng.normal(size=(n,4))); G=Z@Z.T; Cn=np.eye(n)-np.ones((n,n))/n
    for beta in [1e-4,1e-3,.01,.1,.5,1.]:
        P,v=sinkhorn(np.exp(beta*G))
        error=np.linalg.norm(Cn@P@Cn-beta/n*Cn@G@Cn)
        assert error<=R(beta)+1e-13
        assert R(beta)<=(21/4*np.exp(3)+.5*np.exp(4))*beta**2*(1+1e-12)
        assert np.max(np.abs(np.sqrt(n)*v-1))<=np.expm1(1.5*beta)+1e-12
        maxratio=max(maxratio,error/R(beta))
beta=1e-5
A1=C@sinkhorn(np.exp(beta*G1))[0]@C; A2=C@sinkhorn(np.exp(beta*G2))[0]@C
H1=C@G1@C; H2=C@G2@C
assert abs(align(A1,A2)-align(H1,H2))<1e-5
lam,V=np.linalg.eigh(H1); lpb,Vpb=np.linalg.eigh(N*A1/beta)
r=2; gap=lam[-r]-lam[-r-1]; eta=N*R(beta)/beta
assert eta<gap/2
Pi=V[:,-r:]@V[:,-r:].T; Pip=Vpb[:,-r:]@Vpb[:,-r:].T
assert np.linalg.norm(Pip-Pi)<=np.sqrt(2)*eta/(gap-eta)
# Large-temperature angular rate and spectral coalescence.
Z=np.eye(6); Z[1]=np.array([.6,.8,0,0,0,0]); G=Z@Z.T; n=len(Z)
Cn=np.eye(n)-np.ones((n,n))/n; gap=np.min(1-G[np.triu_indices(n,1)])
for beta in [1,5,10,30]:
    P,_=sinkhorn(np.exp(beta*(G-1)))
    bd=min(1,2*(n-1)*np.exp(-beta*gap))
    assert np.linalg.norm(P-np.eye(n),2)<=bd+1e-12
    assert np.linalg.norm(Cn@P@Cn-Cn,2)<=bd+1e-12
    assert np.linalg.norm(np.linalg.matrix_power(P,3)-np.eye(n),2)<=3*bd+1e-12
# Boundary/counterexamples: non-PSD positivity, fractional power, reducible degrees, duplicate rows.
K=np.array([[1.,2.],[2.,1.]])
assert np.allclose(np.linalg.eigvalsh(sinkhorn(K)[0]),[-1/3,1])
P0=np.array([[.75,.25,0],[.25,.5,.25],[0,.25,.75]])
Pe=.99*P0+.01*np.ones((3,3))/3
l,V=np.linalg.eigh(Pe); sq=(V*np.sqrt(l))@V.T
formula=1/3-np.sqrt(.99)*(np.sqrt(3)/4-1/12)
assert np.isclose(sq[0,2],formula) and sq[0,2]<0 and Pe.min()>0 and l.min()>0
D1=np.array([1.,2.]); D2=D1[::-1]
assert np.allclose(np.diag(D1)/np.sqrt(np.outer(D1,D1)),np.eye(2))
for beta in [0,1,10]:
    assert np.allclose(sinkhorn(np.exp(beta*np.ones((2,2))))[0],np.ones((2,2))/2)
print(f'explicit low-beta bound: maximum observed error/bound = {maxratio:.6g}')
print(f'low-beta CKA difference = {abs(align(A1,A2)-align(H1,H2)):.6g}')
print(f'fractional-power negative entry = {sq[0,2]:.9f}')
print('Gaussian variance: exact degree-four quadrature; Rademacher variance: exhaustive enumeration; both m=1 and m=2')

# Review regression 1: PSD (indeed positive definite) is NOT a substitute
# for strict entrywise positivity in the degree-mismatch theorem.
Kdiag1=np.diag([1.,2.]); Kdiag2=np.diag([2.,1.])
assert np.linalg.eigvalsh(Kdiag1).min()>0
assert np.linalg.eigvalsh(Kdiag2).min()>0
def degree_normalize(K):
    degrees=K.sum(1)
    return K/np.sqrt(degrees[:,None]*degrees[None,:])
degree_ratios=Kdiag1.sum(1)/Kdiag2.sum(1)
assert not np.allclose(degree_ratios,degree_ratios[0])
assert np.isclose(np.linalg.norm(degree_normalize(Kdiag2)@degree_normalize(Kdiag1),2),1)

# Review regression 2: general duplicate-class limit, including unequal
# class sizes, a singleton class, and the all-identical endpoint.
rngrev=np.random.default_rng(8591)
for sizes in [[3,1,1],[2,5,1],[1,1,1],[7]]:
    m=len(sizes); n=sum(sizes)
    reps=np.eye(m)
    Zdup=np.repeat(reps,sizes,axis=0)
    Gdup=Zdup@Zdup.T
    labels=np.repeat(np.arange(m),sizes)
    pidup=np.zeros((n,n))
    for c,nc in enumerate(sizes):
        ind=(labels==c).astype(float)
        pidup+=np.outer(ind,ind)/nc
    assert np.allclose(pidup@pidup,pidup)
    Cdup=np.eye(n)-np.ones((n,n))/n
    for beta in [0.,1.,5.,15.,40.]:
        Pdup,_=sinkhorn(np.exp(beta*(Gdup-1)))
        assert np.allclose(Pdup@pidup,Pdup)
        assert np.allclose(Pdup@(np.eye(n)-pidup),0,atol=1e-13)
        if m==1:
            assert np.allclose(Pdup,pidup)
        else:
            angular_gap=1.  # orthonormal class representatives
            kappa=max(sum(np.sqrt(nd/nc) for d,nd in enumerate(sizes) if d!=c)
                      for c,nc in enumerate(sizes))
            errbd=min(1,2*kappa*np.exp(-beta*angular_gap))
            assert np.linalg.norm(Pdup-pidup,2)<=errbd+1e-12
            assert np.linalg.norm(Pdup-pidup)<=np.sqrt(m)*errbd+1e-12
            assert np.linalg.norm(Cdup@Pdup@Cdup-(pidup-np.ones((n,n))/n),2)<=errbd+1e-12
            assert np.linalg.norm(np.linalg.matrix_power(Pdup,3)-pidup,2)<=3*errbd+1e-12
    assert np.allclose(Pdup,pidup,atol=1e-12)
# Different duplicate partitions need not have high-temperature alignment one.
Pa=np.kron(np.eye(2),np.ones((2,2))/2)
perm=[0,2,1,3]; Pb=Pa[np.ix_(perm,perm)]
Cc=np.eye(4)-np.ones((4,4))/4
assert np.isclose(align(Cc@Pa@Cc,Cc@Pb@Cc),0)

# Review regression 3: retained sites are not the full gradient; direct terms
# are not use-mediated terms. Total adjoints still use the original loss.
uall=np.array([[1.],[2.],[3.]])
wscalar=1.5
adj=uall*wscalar  # loss = sum_j (W u_j)^2 / 2 + W^2 / 2
other_view=np.array([[1.],[0.],[1.]])  # zero in another view need not mean zero input
kept=np.linalg.norm(other_view,axis=1)>0
gret=(adj[kept].T@uall[kept]).item()
gexcluded=(adj[~kept].T@uall[~kept]).item()
gdirect=wscalar
fullgrad=wscalar*((uall**2).sum()+1)
assert np.isclose(gret+gexcluded+gdirect,fullgrad)
assert not np.isclose(gret,fullgrad)
# Sol's scalar regularizer counterexample: use gradient 1, full gradient 2.
scalar_loss=lambda w: .5*(w*1.)**2+.5*w**2
scalar_eps=1e-5
scalar_full=(scalar_loss(1+scalar_eps)-scalar_loss(1-scalar_eps))/(2*scalar_eps)
scalar_use=(1.*1.)*1.
assert np.isclose(scalar_full,2) and np.isclose(scalar_use,1)
assert not np.isclose(scalar_full,scalar_use)

# Review regression 4: damped iteration's projective and actual convergence,
# derivative in log and original coordinates, and PSD local rate <= 1/2.
def osc(x):
    return np.max(x)-np.min(x)
def dh(a,b):
    return osc(np.log(a)-np.log(b))
def log_step(x,K):
    return .5*(x-np.log(K@np.exp(x)))
max_hilbert_ratio=0.
for _ in range(100):
    n=int(rngrev.integers(2,15)); beta=float(rngrev.uniform(0,9))
    Z=unit(rngrev.normal(size=(n,4)))
    K=np.exp(beta*(Z@Z.T-1))  # scalar-invariant, avoids large entries
    P,v=sinkhorn(K)
    logK=np.log(K)
    Delta=max(np.ptp(logK[i]-logK[j]) for i in range(n) for j in range(n))
    assert Delta<=4*beta+1e-12
    qK=.5*(1+np.tanh(Delta/4)); qSphere=.5*(1+np.tanh(beta))
    wa=np.exp(rngrev.normal(size=n)); wb=np.exp(rngrev.normal(size=n))
    ga=np.sqrt(wa/(K@wa)); gb=np.sqrt(wb/(K@wb))
    assert dh(ga,gb)<=qK*dh(wa,wb)+1e-12
    assert qK<=qSphere+1e-12
    max_hilbert_ratio=max(max_hilbert_ratio,dh(ga,gb)/(qK*dh(wa,wb)))
    initial=wa.copy(); wk=initial.copy()
    for k in range(8):
        wk=np.sqrt(wk/(K@wk))
        assert np.max(np.abs(np.log(wk)-np.log(v)))<=.5*qK**k*dh(initial,v)+1e-11
    # Exact local derivative, with finite differences in an arbitrary direction.
    h=rngrev.normal(size=n); eps=1e-6
    analytic=.5*(np.eye(n)-P)
    measured=(log_step(np.log(v)+eps*h,K)-log_step(np.log(v)-eps*h,K))/(2*eps)
    assert np.allclose(measured,analytic@h,atol=2e-8)
    assert np.linalg.norm(analytic,2)<=.5+1e-12
    assert np.linalg.norm(analytic@np.ones(n))<1e-12
    orig=.5*(np.eye(n)-v[:,None]*P/v[None,:])
    assert np.allclose(orig,(v[:,None]*analytic)/v[None,:])
# A clustered, near-reducible PSD kernel still has the damped local bound.
z=np.repeat([[1.,0.],[-1.,0.]], [10,10],axis=0)
k=np.exp(20*(z@z.T-1)); P,v=sinkhorn(k)
assert np.linalg.eigvalsh(P)[-2]>1-1e-12
assert np.linalg.norm(.5*(np.eye(len(z))-P),2)<=.5+1e-12

# Optional A.5 strengthening: singular-pair closeness from two contractions.
for _ in range(30):
    z1=unit(rngrev.normal(size=(10,4))); z2=unit(rngrev.normal(size=(10,4)))
    p1,_=sinkhorn(np.exp(3*(z1@z1.T-1)))
    p2,_=sinkhorn(np.exp(3*(z2@z2.T-1)))
    uu,ss,vv=np.linalg.svd(p2@p1)
    for i in range(1,len(ss)):
        assert np.linalg.norm(ss[i]*uu[:,i]-vv[i])<=2*np.sqrt(max(0,1-ss[i]**2))+1e-12
print(f'review regression Hilbert contraction: maximum observed ratio/bound = {max_hilbert_ratio:.6g}')
print('all A1 checks passed')
