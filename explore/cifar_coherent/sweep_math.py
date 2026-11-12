"""Kernel-rule sweep: labels enter only the final matched-Jaccard validation."""
import math
import numpy as np
import torch
from coherent import symmetric_balance,block_modes,seba,seba_labels


def mass_calibration(c,target,k=15):
    d=1-c.double();d.diagonal().fill_(float('inf'))
    near=d.topk(k,largest=False,dim=1).values
    shifted=d-near[:,:1];nshifted=near-near[:,:1]
    def mass(kappa):
        return float((torch.exp(-kappa*nshifted).sum(1)/torch.exp(-kappa*shifted).sum(1)).quantile(.5))
    lo,hi=0.,1.
    while mass(hi)<target:
        hi*=2
        if hi>1e8:raise ValueError('Mass target is unattainable due to distance ties')
    for _ in range(45):
        mid=(lo+hi)/2
        if mass(mid)<target:lo=mid
        else:hi=mid
    return (lo+hi)/2


def perplexity_logkernel(c,perplexity):
    d=1-c.double();d.diagonal().fill_(float('inf'))
    d-=d.min(1).values[:,None]
    moment_d=d.clone();moment_d.diagonal().fill_(0)
    n=len(d);low=torch.zeros(n,device=d.device,dtype=d.dtype);high=torch.ones_like(low)
    target=math.log(perplexity)
    def conditional(beta):
        logits=-beta[:,None]*d
        logz=torch.logsumexp(logits,dim=1)
        logp=logits-logz[:,None]
        entropy=logz+beta*(torch.exp(logp)*moment_d).sum(1)
        return entropy,logp
    for _ in range(40):
        ent,_=conditional(high);bad=ent>target
        if not bool(bad.any()):break
        high[bad]*=2
    else:raise ValueError('Perplexity bracketing did not converge')
    for _ in range(50):
        beta=(low+high)/2;ent,_=conditional(beta)
        low=torch.where(ent>target,beta,low);high=torch.where(ent>target,high,beta)
    beta=(low+high)/2;ent,logp=conditional(beta)
    assert float((ent-target).abs().max())<1e-7
    logk=torch.logaddexp(logp,logp.T)-math.log(2)
    return logk,dict(kappa_quantiles=beta.quantile(torch.tensor([0.,.25,.5,.75,1.],device=beta.device,dtype=beta.dtype)).cpu().tolist(),
                     kappa_median=float(beta.quantile(.5)),kappa_q10=float(beta.quantile(.1)),kappa_q90=float(beta.quantile(.9)),
                     perplexity_error=float((torch.exp(ent)-perplexity).abs().max()))


def kernel_rules(c):
    d=1-c.double();d.diagonal().fill_(float('inf'))
    for kap in (2,3,5,8,12,16,20,25,30):
        yield f'global {kap}',-kap*d,dict(family='global',kappa=kap)
    for target in (.05,.1,.2,.5):
        kap=mass_calibration(c,target)
        yield f'mass {target:g}',-kap*d,dict(family='mass',target=target,kappa=kap)
    for perp in (15,30,100,300):
        logk,info=perplexity_logkernel(c,perp)
        yield f'perplexity {perp}',logk,dict(family='perplexity',perplexity=perp,**info)
    for k in (15,50):
        idx=d.topk(k,largest=False,dim=1).indices
        mask=torch.zeros_like(c,dtype=torch.bool);mask.scatter_(1,idx,True)
        mask=mask|mask.T
        weights=mask.double()+1e-6;weights.diagonal().fill_(0)
        yield f'union kNN {k}',torch.log(weights),dict(family='union_knn',k=k,floor=1e-6)


def final_kernel_bank(c):
    """The five final, label-free selected rules, with explicit null conventions."""
    from coherent import sinkhorn_operator
    d=1-c.double();d.diagonal().fill_(float('inf'))
    near=d.topk(15,largest=False,dim=1)
    mask=torch.zeros_like(c,dtype=torch.bool);mask.scatter_(1,near.indices,True);mask=mask|mask.T
    weights=torch.where(mask,torch.ones_like(c),torch.full_like(c,1e-6));weights.diagonal().fill_(0)
    bank=[(torch.log(weights),dict(kappa=None,kappa_q10=None,kappa_q90=None,self_weight=None))]
    for perplexity in (100,300):
        logk,info=perplexity_logkernel(c,perplexity)
        bank.append((logk,dict(kappa=info['kappa_median'],kappa_q10=info['kappa_q10'],kappa_q90=info['kappa_q90'],self_weight=None)))
    ks=[1/float(near.values[:,-1].quantile(.5)),mass_calibration(c,.5)]
    for k in ks:
        bank.append((-k*d,dict(kappa=k,kappa_q10=k,kappa_q90=k,self_weight=float(sinkhorn_operator(c,k).diag().quantile(.5)))))
    for logk,info in bank:
        probabilities=torch.exp(logk-torch.logsumexp(logk,dim=1)[:,None])
        info['nn_mass']=float(probabilities.gather(1,near.indices).sum(1).quantile(.5))
    return bank


def summarize(c,labels):
    from scipy.optimize import linear_sum_assignment
    import time
    rows=[]
    for rule,logk,params in kernel_rules(c):
        start=time.time()
        p,info=symmetric_balance(logk,return_info=True)
        m=p*.5;m.diagonal().add_(.5)
        val,v=block_modes([m],9,shared_stationary=True)
        ipr=(1/v[:,1:].pow(4).sum(0)).cpu().tolist()
        sb={}
        for r in (4,8):
            hard=seba_labels(seba(v.cpu(),r)).numpy()
            sizes=np.array([(hard==j).sum() for j in range(r)])
            intersection=np.array([[np.sum((hard==j)&(labels==cl)) for cl in range(10)] for j in range(r)])
            union=sizes[:,None]+np.bincount(labels,minlength=10)[None,:]-intersection
            jac=intersection/np.maximum(union,1)
            js,cls=linear_sum_assignment(-jac)
            sb[str(r)]=dict(sizes=sizes.tolist(),unassigned=int((hard<0).sum()),matched_classes=cls.tolist(),
                            matched_jaccard=jac[js,cls].tolist(),mean_matched_jaccard=float(jac[js,cls].mean()))
        rows.append(dict(rule=rule,**params,gaps=(1-val[1:]).cpu().tolist(),participation=ipr,seba=sb,
                         sinkhorn_iterations=info['iterations'],row_error=info['row_error'],newton_steps=info['newton_steps'],
                         sinkhorn_method=info['method'],seconds=time.time()-start))
        print(rule,'gap1',rows[-1]['gaps'][0],'PN1',ipr[0],'seconds',time.time()-start,flush=True)
    return rows
