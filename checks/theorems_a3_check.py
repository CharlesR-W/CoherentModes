"""Asserted numerical checks for the A3 sampling/cost theorem draft."""
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import numpy as np
from pathlib import Path
rng=np.random.default_rng(4417)
def psqrt(A):
    d,u=np.linalg.eigh((A+A.T)/2)
    return (u*np.sqrt(np.maximum(d,0)))@u.T
def invsqrt(A):
    d,u=np.linalg.eigh((A+A.T)/2)
    return (u*(1/np.sqrt(d)))@u.T
def scale(K,w):
    s=np.ones(len(w))
    for j in range(4000):
        row=s*(K@(w*s))
        if np.max(np.abs(row-1))<2e-13: return s
        s/=np.sqrt(row)
    raise AssertionError('scaling failed')
def slope(ns, errors): return np.polyfit(np.log(ns),np.log(errors),1)[0]
# A.43/A.44: population common-space lifts on five distinct support points.
x=rng.normal(size=(5,3)); x/=np.linalg.norm(x,axis=1,keepdims=True)
beta=.35; K=np.exp(beta*x@x.T); F=psqrt(K); w=np.array([.1,.17,.23,.31,.19]); s=scale(K,w)
C=F.T@np.diag(w*s*s)@F
assert abs(np.linalg.eigvalsh(C)[-1]-1)<1e-11
ns=np.array([128,512,2048,8192]); raw=[]; normal=[]; eigen=[]
Craw=F.T@np.diag(w)@F
for n in ns:
    errs=[]
    for rep in range(350):
        wn=rng.multinomial(int(n),w)/n; sn=scale(K,wn)
        Cn=F.T@np.diag(wn*sn*sn)@F
        errs.append([np.linalg.norm(F.T@np.diag(wn)@F-Craw,2),np.linalg.norm(Cn-C,2),np.max(np.abs(np.linalg.eigvalsh(Cn)-np.linalg.eigvalsh(C)))])
        assert np.max(np.abs(np.linalg.eigvalsh(Cn)-np.linalg.eigvalsh(C)))<=np.linalg.norm(Cn-C,2)+1e-12
    z=np.mean(errs,axis=0);raw.append(z[0]);normal.append(z[1]);eigen.append(z[2])
for name,errs in [('raw',raw),('normalized',normal),('eigenvalue',eigen)]:
    a=slope(ns,errs); print(name,'mean-errors',np.round(errs,6),'log-log slope',round(a,3));assert -.65<a<-.35
# Finite-N probability/counting conversion and extension at training points.
z=x[rng.choice(5,size=33,p=w)]; kn=np.exp(beta*z@z.T); sn=scale(kn,np.ones(33)/33); P=sn[:,None]*kn*sn[None,:]/33
lam,U=np.linalg.eigh(P); i=-2
ext=(sn[:,None]*kn*sn[None,:]/33)@(np.sqrt(33)*U[:,i])/lam[i]
assert np.max(np.abs(ext-np.sqrt(33)*U[:,i]))<1e-10
print('Nystrom eigenfunction extension counting/probability conversion passed')
# A.47/A.48: two-state independent conditional views; centered rank one.
pp=np.array([[.7,.2,.1],[.1,.3,.6]]); pq=np.array([[.2,.6,.2],[.65,.25,.1]])
J=.4*np.outer(pp[0],pq[0])+.6*np.outer(pp[1],pq[1])
vp=rng.normal(size=(3,2));vp/=np.linalg.norm(vp,axis=1,keepdims=True)
vq=rng.normal(size=(3,2));vq/=np.linalg.norm(vq,axis=1,keepdims=True)
Fp=psqrt(np.exp(vp@vp.T));Fq=psqrt(np.exp(vq@vq.T)); rho=.08

def ccas(J):
    wp=J.sum(1);wq=J.sum(0); A=Fp-wp@Fp;B=Fq-wq@Fq
    Cp=A.T@np.diag(wp)@A;Cq=B.T@np.diag(wq)@B;Cqp=B.T@J.T@A
    V=invsqrt(Cq+rho*np.eye(3))@Cqp@invsqrt(Cp+rho*np.eye(3))
    return Cp,Cq,Cqp,V
Cp,Cq,Cqp,V=ccas(J);assert np.linalg.svd(V,compute_uv=False)[1]<1e-12
cca_errors=[];tails=[]
for n in ns:
    es=[]; ts=[]
    for rep in range(200):
        Jn=rng.multinomial(int(n),J.ravel()).reshape(3,3)/n
        Cpn,Cqn,Cqpn,Vn=ccas(Jn)
        e=max(np.linalg.norm(Cpn-Cp,2),np.linalg.norm(Cqn-Cq,2),np.linalg.norm(Cqpn-Cqp,2))
        error=np.linalg.norm(Vn-V,2); tail=np.linalg.svd(Vn,compute_uv=False)[1]
        bound=e*(rho**-1+np.sqrt(np.e)*rho**-1.5)
        assert error<=bound+1e-12 and tail<=error+1e-12
        es.append(error);ts.append(tail)
    cca_errors.append(np.mean(es));tails.append(np.mean(ts))
for name,errs in [('CCA',cca_errors),('CCA beyond rank-one tail',tails)]:
    a=slope(ns,errs);print(name,'mean-errors',np.round(errs,6),'log-log slope',round(a,3));assert -.65<a<-.35
# A.52: ridge-leverage Nystrom deterministic identities and error.
N=60;v=rng.normal(size=(N,4));v/=np.linalg.norm(v,axis=1,keepdims=True);KK=np.exp(.4*v@v.T);tau=.08
A=KK/N;d,U=np.linalg.eigh(A); scores=(U**2)@(d/(d+tau)); deff=scores.sum();eta=.05;m=int(np.ceil(10*deff*np.log(2*N/eta)))
I=rng.choice(N,m,p=scores/deff);Ck=KK[:,I];W=KK[np.ix_(I,I)];ny=Ck@np.linalg.pinv(W,rcond=1e-11)@Ck.T
res=(KK-ny+KK.T-ny.T)/(2*N)
assert np.linalg.eigvalsh(res)[0]>-1e-7 and np.linalg.norm(res,2)<tau+1e-7
print('ridge-leverage Nystrom bound passed; effective dimension',round(deff,3),'budget',m,'N',N)
# A.49: exact pairwise/variance identity and root-n rate.
mu=np.array([.3,.7]); vv=np.array([[1.,0.],[0.,1.]]);theta=2*(1-np.linalg.norm(mu@vv)**2)
noise_errors=[]
for n in ns:
    errs=[]
    for rep in range(350):
        yy=vv[rng.choice(2,int(n),p=mu)]; avg=yy.mean(0); est=2*n/(n-1)*(1-np.dot(avg,avg));errs.append(abs(est-theta))
    noise_errors.append(np.mean(errs))
a=slope(ns,noise_errors);print('noise U-statistic slope',round(a,3));assert -.65<a<-.35
Y=rng.normal(size=(21,4));Y/=np.linalg.norm(Y,axis=1,keepdims=True);dist=np.sum((Y[:,None]-Y[None,:])**2,axis=-1)
assert abs(dist.sum()/(21*20)-2*21/20*np.mean(np.sum((Y-Y.mean(0))**2,axis=1)))<1e-12
# A.54: unbiasedness and variance, correlated shared sketches.
Ji=rng.normal(size=(4,3));Jj=rng.normal(size=(4,3));Bij=(Ji.T@Jj+Jj.T@Ji)/2;mj=20
G=rng.normal(size=(6000,mj,3));qi=np.einsum('ab,rmb->rma',Ji,G);qj=np.einsum('ab,rmb->rma',Jj,G)
hat=np.mean(np.sum(qi*qj,axis=-1),axis=1);true=np.sum(Ji*Jj);var=2*np.sum(Bij*Bij)/mj
assert abs(hat.mean()-true)<5*np.sqrt(var/6000)
assert abs(hat.var()/var-1)<.08
print('Gaussian shared-probe variance measured/theory',round(hat.var()/var,3))
# Draft format checks.
text=(Path(__file__).resolve().parents[1] / 'theorems/drafts/A3-sampling.md').read_text()
assert '\\{' not in text and '\\}' not in text
for label in range(40,57): assert f'A.{label} ' in text
assert text.count('**What it buys.**')==15
assert text.count('**Tractability and sampling.**')==15
# Reviewed refinements: all additional checks use their own RNG to preserve prior results.
rng2 = np.random.default_rng(9055)
# A.44 refined constants and deterministic stability implication on a concentration event.
beta2 = 0.5
points = rng2.normal(size=(6, 3)); points /= np.linalg.norm(points, axis=1, keepdims=True)
K2 = np.exp(beta2 * points @ points.T); Phi2 = psqrt(K2)
w2 = np.array([.1, .12, .18, .2, .15, .25]); s2 = scale(K2, w2)
B0 = np.exp(beta2) * s2.max()**2; B_univ = np.exp(4*beta2)
assert B0 <= B_univ + 1e-12
n2, eta2 = 20000, .1
h2 = (1+np.sqrt(2*np.log(2/eta2)))/np.sqrt(n2)
eps2 = B0*h2; t2 = (np.exp(2*beta2)+.5)*(-np.log1p(-eps2))
E2 = eps2+np.expm1(2*t2)*(1+eps2)
Pop2 = Phi2.T @ np.diag(w2*s2*s2) @ Phi2
for _ in range(40):
    wn2 = rng2.multinomial(n2, w2)/n2; sn2 = scale(K2, wn2)
    Emp2 = Phi2.T @ np.diag(wn2*sn2*sn2) @ Phi2
    assert np.max(np.abs(np.log(sn2/s2))) <= t2+1e-12
    assert np.linalg.norm(Emp2-Pop2, 2) <= E2+1e-12
print('refined Sinkhorn bound passed; B0/universal bound', round(B0/B_univ, 4))
# A.55 cross-layer product identity and both square-root perturbation bounds.
other = rng2.normal(size=(6, 4)); other /= np.linalg.norm(other, axis=1, keepdims=True)
K3 = np.exp(.8*other@other.T); Phi3 = psqrt(K3); s3 = scale(K3, w2)
A0 = np.sqrt(w2)[:,None]*s2[:,None]*Phi2
B0map = np.sqrt(w2)[:,None]*s3[:,None]*Phi3
Sa, Sb, Xi = A0.T@A0, B0map.T@B0map, B0map.T@A0
Qlift = psqrt(Sb)@Xi@psqrt(Sa)
Qsite = (B0map@B0map.T)@(A0@A0.T)
assert np.max(np.abs(np.linalg.svd(Qsite, compute_uv=False)-np.linalg.svd(Qlift, compute_uv=False))) < 1e-11
qerrors=[]; floor_checks=0
for n in ns:
    es=[]
    for _ in range(150):
        wn = rng2.multinomial(int(n), w2)/n
        san, sbn = scale(K2, wn), scale(K3, wn)
        An = np.sqrt(wn)[:,None]*san[:,None]*Phi2
        Bn = np.sqrt(wn)[:,None]*sbn[:,None]*Phi3
        San, Sbn, Xin = An.T@An, Bn.T@Bn, Bn.T@An
        qn = psqrt(Sbn)@Xin@psqrt(San)
        site = (Bn@Bn.T)@(An@An.T)
        assert np.max(np.abs(np.linalg.svd(site, compute_uv=False)-np.linalg.svd(qn, compute_uv=False))) < 1e-10
        ea, eb, ex = np.linalg.norm(San-Sa,2), np.linalg.norm(Sbn-Sb,2), np.linalg.norm(Xin-Xi,2)
        err = np.linalg.norm(qn-Qlift,2)
        assert err <= np.sqrt(ea)+ex+np.sqrt(eb)+1e-11
        nua,nub=np.linalg.eigvalsh(Sa)[0],np.linalg.eigvalsh(Sb)[0]
        if ea<=nua/2 and eb<=nub/2:
            assert err <= ea/np.sqrt(nua)+ex+eb/np.sqrt(nub)+1e-11
            floor_checks+=1
        es.append(err)
    qerrors.append(np.mean(es))
qslope=slope(ns,qerrors)
assert -.7<qslope<-.3
# Force a small perturbation to test the finite-feature-floor implication as well.
wt=(1-1e-5)*w2+1e-5*np.ones(6)/6
At=np.sqrt(wt)[:,None]*scale(K2,wt)[:,None]*Phi2
Bt=np.sqrt(wt)[:,None]*scale(K3,wt)[:,None]*Phi3
Sat,Sbt,Xit=At.T@At,Bt.T@Bt,Bt.T@At
ea,eb,ex=np.linalg.norm(Sat-Sa,2),np.linalg.norm(Sbt-Sb,2),np.linalg.norm(Xit-Xi,2)
assert ea<nua/2 and eb<nub/2
assert np.linalg.norm(psqrt(Sbt)@Xit@psqrt(Sat)-Qlift,2)<=ea/np.sqrt(nua)+ex+eb/np.sqrt(nub)+1e-12
print('Q lift and perturbation bounds passed; observed finite-support slope',round(qslope,3),'sampled floor cases',floor_checks)
# A.47 nonleading canonical-function inequality and centering.
def pinvsqrt(M):
    vals,U=np.linalg.eigh((M+M.T)/2)
    inv=np.zeros_like(vals);np.divide(1,np.sqrt(np.maximum(vals,0)),out=inv,where=vals>1e-11)
    return (U*inv)@U.T
H=np.array([[1,1,1,1],[1,1,-1,-1],[1,-1,1,-1],[1,-1,-1,1]],float)/2
transition=np.ones((4,4))/4+(H[1:].T*np.array([.4,.2,.1]))@H[1:]
joint=transition/4
xp=rng2.normal(size=(4,3));xp/=np.linalg.norm(xp,axis=1,keepdims=True)
xq=rng2.normal(size=(4,3));xq/=np.linalg.norm(xq,axis=1,keepdims=True)
FP=psqrt(np.exp(xp@xp.T));FQ=psqrt(np.exp(xq@xq.T))
def finite_cov(joint):
    wp=joint.sum(1);wq=joint.sum(0)
    X=FP-wp@FP;Y=FQ-wq@FQ
    return X.T@np.diag(wp)@X,Y.T@np.diag(wq)@Y,Y.T@joint.T@X,wp,wq
Cp0,Cq0,Cqp0,wp0,wq0=finite_cov(joint)
V0=pinvsqrt(Cq0)@Cqp0@pinvsqrt(Cp0)
U0,sv0,Vh0=np.linalg.svd(V0);assert np.max(np.abs(sv0[:3]-[.4,.2,.1]))<1e-9
up=Vh0[1];ap=pinvsqrt(Cp0)@up
for rho in [.1,.03,.01]:
    jn=rng2.multinomial(10000,joint.ravel()).reshape(4,4)/10000
    Cpn,Cqn,Cqpn,wpn,wqn=finite_cov(jn)
    vn=invsqrt(Cqn+rho*np.eye(4))@Cqpn@invsqrt(Cpn+rho*np.eye(4))
    un=np.linalg.svd(vn)[2][1];un*=1 if un@up>=0 else -1
    an=invsqrt(Cpn+rho*np.eye(4))@un
    fp=(FP-wp0@FP)@ap; fpn=(FP-wp0@FP)@an
    f_emp=(FP-wpn@FP)@an
    assert np.linalg.norm((f_emp-wp0@f_emp)-fpn)<1e-12
    err=np.sqrt(wp0@((fpn-fp)**2));e=np.linalg.norm(Cpn-Cp0,2)
    bias=np.linalg.norm((psqrt(Cp0)@invsqrt(Cp0+rho*np.eye(4))-np.eye(4))@up)
    bound=np.sqrt(1+e/rho)*np.linalg.norm(un-up)+np.sqrt(np.e)*e/(2*rho**1.5)+bias
    assert err<=bound+1e-11
print('nonleading canonical-function inequality and marginal centering passed')
# A.49 exact-binomial verification of the Bernstein U-statistic tail.
from math import lgamma
prare=.005;theta_small=4*prare*(1-prare)
for n in [200,1000,10000]:
    counts=np.arange(n+1,dtype=float)
    probs=np.array([np.exp(lgamma(n+1)-lgamma(k+1)-lgamma(n-k+1)+k*np.log(prare)+(n-k)*np.log1p(-prare)) for k in range(n+1)])
    assert abs(probs.sum()-1)<5e-11
    estimates=4*counts*(n-counts)/(n*(n-1))
    for rel in [.2,.5,1.]:
        u=rel*theta_small
        exact=probs[np.abs(estimates-theta_small)>u].sum()
        bern=min(1.,2*np.exp(-(n//2)*u*u/(8*theta_small+8*u/3)))
        assert exact<=bern+1e-10
print('Bernstein U-statistic bounds passed against exact binomial tails')
# A.56 repeated within-document positions do not create independent samples.
D,T=60,7
idx=rng2.choice(6,D,p=w2); document_features=(s2[:,None]*Phi2)[idx]
rows=np.repeat(document_features,T,axis=0)
assert np.max(np.abs(rows.mean(0)-document_features.mean(0)))<1e-12
assert np.max(np.abs(rows.T@rows/(D*T)-document_features.T@document_features/D))<1e-12
print('document block weighting and duplicated-position counterexample passed')
# A.51 projective contraction, explicit initial bound, and no coherence implication.
c=np.ones(6);vstar=scale(K2,np.ones(6)/6)/np.sqrt(6)
def osc(v):return np.max(v)-np.min(v)
assert osc(np.log(c)-np.log(vstar))<=2*beta2+1e-12
for _ in range(12):
    d0=osc(np.log(c)-np.log(vstar)); rr=1/(K2@c);c=1/(K2.T@rr)
    d1=osc(np.log(c)-np.log(vstar))
    assert d1 <= (1-np.exp(-2*beta2))**2*d0+2e-11
u,vv=H[1],H[2];JJ=np.ones((4,4))/4;aa=.999
P1=JJ+aa*np.outer(u,u);P2=JJ+aa*np.outer(vv,vv)
assert np.min(P1)>=0 and np.min(P2)>=0
assert abs(np.linalg.eigvalsh(P1)[-2]-aa)<1e-12
assert abs(np.linalg.eigvalsh(P2)[-2]-aa)<1e-12
assert np.linalg.norm(P2@P1-JJ)<1e-12
print('Sinkhorn projective contraction and orthogonal-slow-block counterexample passed')

print('all A3 checks passed')

