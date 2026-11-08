# Numerical checks for THEORY.md v1: bistochastic product has trivial singular value 1 (symmetric normalization does not);
# centered low-beta operator ~ (beta/N) C G C and centered alignment -> linear CKA; dual KCCA eig(R_p R_q) = canonical correlations^2.
# Run: uv run --with numpy python -I checks/cm_check.py
import numpy as np
rng=np.random.default_rng(0)
N,d=60,5
def unit(A): return A/np.linalg.norm(A,axis=1,keepdims=True)
Z1=unit(rng.normal(size=(N,d))); Z2=unit(rng.normal(size=(N,d))+0.5*Z1)
def sinkhorn(K,it=5000):
    v=np.ones(len(K))
    for _ in range(it):
        v=np.sqrt(v/(K@v))
    P=K*v[:,None]*v[None,:]; return P
C=np.eye(N)-np.ones((N,N))/N
# 1. bistochastic: shared trivial eigenvector, product top singular value 1
P1=sinkhorn(np.exp(2.0*Z1@Z1.T)); P2=sinkhorn(np.exp(2.0*Z2@Z2.T))
print("row sums",P1.sum(1).min(),P1.sum(1).max())
Q=P2@P1; s=np.linalg.svd(Q,compute_uv=False); print("top sv of Q",s[:3])
# compare with symmetric normalization product
def symn(K): D=K.sum(1); return K/np.sqrt(D[:,None]*D[None,:])
S1,S2=symn(np.exp(2.0*Z1@Z1.T)),symn(np.exp(2.0*Z2@Z2.T)); print("top sv sym-norm product",np.linalg.svd(S2@S1,compute_uv=False)[:3])
# 2. beta->0: C P_beta C vs (beta/N) C G C
for b in [1e-2,1e-3]:
    Pb=sinkhorn(np.exp(b*Z1@Z1.T)); A=C@Pb@C; B=(b/N)*C@(Z1@Z1.T)@C
    print("beta",b,"rel err",np.linalg.norm(A-B)/np.linalg.norm(B))
# alignment -> linear CKA of normalized rows
def cka(G1,G2): return np.sum((C@G1@C)*(C@G2@C))/np.sqrt(np.sum((C@G1@C)**2)*np.sum((C@G2@C)**2))
b=1e-3; A1=C@sinkhorn(np.exp(b*Z1@Z1.T))@C; A2=C@sinkhorn(np.exp(b*Z2@Z2.T))@C
al=np.sum(A1*A2)/np.sqrt(np.sum(A1**2)*np.sum(A2**2)); ck=cka(Z1@Z1.T,Z2@Z2.T); print("centered alignment",al,"linear CKA",ck); assert abs(al-ck)<1e-3
# 3. KCCA dual form: eig(R_p R_f) = squared regularized canonical correlations (linear features for check)
Xp=rng.normal(size=(N,3)); Xf=Xp@rng.normal(size=(3,4))+0.3*rng.normal(size=(N,4))
Xp-=Xp.mean(0); Xf-=Xf.mean(0); rho=0.1
Cpp=Xp.T@Xp/N; Cff=Xf.T@Xf/N; Cfp=Xf.T@Xp/N
def isqrt(M): w,V=np.linalg.eigh(M); return V@np.diag(w**-0.5)@V.T
V=isqrt(Cff+rho*np.eye(4))@Cfp@isqrt(Cpp+rho*np.eye(3)); cc=np.linalg.svd(V,compute_uv=False)
Kp,Kf=Xp@Xp.T,Xf@Xf.T
Rp=Kp@np.linalg.inv(Kp+N*rho*np.eye(N)); Rf=Kf@np.linalg.inv(Kf+N*rho*np.eye(N))
ev=np.sort(np.linalg.eigvals(Rp@Rf).real)[::-1][:3]
print("primal canon corr^2",cc**2,"dual eig(RpRf)",ev)
# 4. variates: eigenvectors of R_p R_f are the past canonical variates (up to scale)
w,Vd=np.linalg.eig(Rp@Rf); o=np.argsort(-w.real); sd=Vd[:,o[0]].real
Wp=isqrt(Cpp+rho*np.eye(3))@np.linalg.svd(V)[2][0]; sp=Xp@Wp
vc=abs(sd@sp)/np.linalg.norm(sd)/np.linalg.norm(sp); print("variate cosine |<dual,primal>|",vc); assert vc>1-1e-6
# 5. indicator block vs site-group gradient differ (THEORY 7.2)
U=np.array([[1.],[-1.]]); De=U.copy(); Pi=np.ones((2,2))/2; DA=np.eye(2)
gg,ib=(De.T@DA@U).item(),(De.T@Pi@U).item(); print("group grad",gg,"indicator-block contribution",ib); assert gg==2.0 and abs(ib)<1e-12
# 6. reconstruction tail identity and assertions
F=rng.normal(size=(N,4)); Psi=np.linalg.qr(rng.normal(size=(N,N)))[0]; A=F.T@Psi; r=10
assert abs(np.linalg.norm(F-Psi[:,:r]@A[:,:r].T)**2-(A[:,r:]**2).sum())<1e-8
assert abs(s[0]-1)<1e-9 and P1.sum(1).std()<1e-9
assert np.allclose(np.sort(cc**2)[::-1],ev,atol=1e-8)
# 7. merge example: contrast slow at L, annihilated at L'
e1,e2=np.eye(2); HL=np.array([e1,e1,e2,e2]); HLp=np.array([e1]*4); b=10.0
PL=sinkhorn(np.exp(b*HL@HL.T)); PLp=sinkhorn(np.exp(b*HLp@HLp.T)); c=np.array([1,1,-1,-1])/2
n1,n2=np.linalg.norm(PL@c),np.linalg.norm(PLp@(PL@c)); print("contrast norm after P^L",n1,"(tanh(b/2)=%.5f)"%np.tanh(b/2),"after P^L'",n2); assert abs(n1-np.tanh(b/2))<1e-6 and n2<1e-12
print("all checks passed")
