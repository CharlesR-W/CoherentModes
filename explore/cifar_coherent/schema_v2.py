"""Shared schema-v2 computation/export, used by fixture and Modal workers.

JSON conventions: comparison files have `rows`, each with step/mode/a/b/r and
agreement keys; decod.json has `steps`, `sites`, `views` and `results` (one legacy
protocol result per checkpoint). Retention padding is NaN; transform padding 0,
permutation padding -1. Float64 operators/eigensystems; float32 exported scalars.
"""
from pathlib import Path
import json
import numpy as np
import torch
from construction import FINGERPRINT
from coherent import (block_modes, seba, seba_labels, seba_reconstruct,
                      centred_cosine_gram, kappa_modes, sinkhorn_operator, fixed_lazy_operator)

STEPS = [0, 64, 256, 1024, 2560, 6400, 25600, 64000]
SITES = ['stem'] + [f'b{i}' for i in range(1, 10)] + ['pool', 'logits']
MODES = ['kNN graph, k = 15', 'per-point perplexity 100', 'per-point perplexity 300', 'kNN scale', 'half-mass, calibrated']
RANKS = [2, 3, 4, 6, 8]
CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
CHANNELS = [16]*4 + [32]*3 + [64]*4 + [10]

def pairs(n):
    return [(a,b) for a in range(n) for b in range(a,n)]


def metadata(kappa, mass, self_weight, kappa_q10, kappa_q90, fixture=False):
    return dict(schema_version=2, construction_fingerprint=FINGERPRINT, steps=STEPS, sites=SITES, modes=MODES,
                kappa=np.asarray(kappa,dtype=object).tolist(), nn_mass=np.asarray(mass).tolist(), self_weight=np.asarray(self_weight,dtype=object).tolist(),
                kappa_q10=np.asarray(kappa_q10,dtype=object).tolist(),kappa_q90=np.asarray(kappa_q90,dtype=object).tolist(),default_mode=0,
                seba_r=RANKS, n_vecs=8, layer_pairs=pairs(12), time_pairs=pairs(8),
                n=2000, classes=CLASSES, fixture=fixture,
                rule='Cosine distance 1-C; midpoint medians. 0: union 15NN, weight1 on edges,1e-6 elsewhere, zero diagonal. 1/2: per-row conditional perplexity100/300 then symmetric mean affinity. 3: kappa=1/median15th-neighbour distance. 4: calibrate median raw nonself15NN mass=.5. All: log-domain symmetric scaling P of zero-diagonal affinity, M=(I+P)/2. kappa/q10/q90 are null for graph, per-point quantiles for perplexity, identical scalars for modes3/4. self_weight is null for modes0–2, retained-vMF Sinkhorn median diagonal for3/4. nn_mass uses each actual symmetric pre-Sinkhorn kernel.',
                quantization='int16, psi = vecs * scale / 32767; no renormalization after decoding',
                seba='Reference Algorithm 3.1; mu=.99/sqrt(N); columns normalized within iteration; rotation acts on original signed unit eigenvectors; clip negatives only after reliability ordering.',
                json_layout='Comparison rows: step, mode, a, b, r and agreement keys. decod.results: one protocol result per step; results[i].views[view][site].ridge_acc/knn_acc. curves.checkpoints: manifest entries.')


def allocate(nm, npair, n, nc):
    return dict(vals=np.empty((nm,npair,9),np.float32),
                vecs=np.empty((nm,npair,8,n),np.int16),
                scale=np.empty((nm,npair,8),np.float32),
                seba_label=np.empty((nm,npair,5,n),np.int8),
                seba_rot=np.zeros((nm,npair,5,8,8),np.float32),
                seba_perm=np.full((nm,npair,5,8),-1,np.int8),
                retention_psi=np.empty((nm,npair,8,nc),np.float32),
                retention_seba=np.full((nm,npair,5,8,nc),np.nan,np.float32))


def block_record(ms, a, b, eigen=None):
    """The one real computation path: mean, spectrum, SEBA, all-cell retention."""
    values, vectors = block_modes(ms[a:b+1], 9, shared_stationary=True) if eigen is None else eigen
    assert abs(float(values[0])-1) < 1e-6
    # The analytical stationary function fixes the trivial mode, even if the
    # numerical eigensolver returned a slightly mixed near-degenerate vector.
    assert float((vectors[:,0] - 1/len(vectors)**.5).abs().max()) < 1e-5
    vectors[:,0] = 1/len(vectors)**.5
    v = vectors.cpu().double()
    scale = v[:,1:].abs().max(0).values
    quant = torch.round(32767*v[:,1:]/scale).to(torch.int16)
    result = dict(vals=values.cpu().numpy().astype('f4'), vecs=quant.T.numpy(), scale=scale.numpy().astype('f4'),
                  seba_label=np.empty((5,len(v)),np.int8), seba_rot=np.zeros((5,8,8),np.float32),
                  seba_perm=np.full((5,8),-1,np.int8), retention_psi=None,
                  retention_seba=np.full((5,8,len(ms)),np.nan,np.float32))
    ss=[]
    for ri,r in enumerate(RANKS):
        s,rot,perm=seba(v,r,return_transform=True)
        result['seba_label'][ri]=seba_labels(s).numpy().astype('i1')
        result['seba_rot'][ri,:r,:r]=rot.numpy()
        result['seba_perm'][ri,:r]=perm.numpy()
        ss.append(s)
    f=torch.cat([v[:,1:]]+ss,1).to(ms[0].device)
    ret=torch.stack([(f*(m@f)).sum(0)/f.square().sum(0) for m in ms],1).cpu().numpy().astype('f4')
    result['retention_psi']=ret[:8]
    at=8
    for ri,r in enumerate(RANKS):
        result['retention_seba'][ri,:r]=ret[at:at+r];at+=r
    return result, vectors


def put_record(arrays, mode, pair, rec):
    for key in arrays:
        arrays[key][mode,pair]=rec[key]


def quantize_channels(x):
    x=np.asarray(x,dtype='f4')
    lo=x.min(0); hi=x.max(0)
    q=np.rint(255*(x-lo)/np.maximum(hi-lo,1e-30)).clip(0,255).astype('u1')
    return q,lo,hi


def point_arrays(channels, logits, labels, fc_w, fc_b):
    z=torch.as_tensor(logits).double(); y=torch.as_tensor(labels).long()
    pred=z.argmax(1).numpy().astype('i1')
    result=dict(loss=torch.nn.functional.cross_entropy(z,y,reduction='none').numpy().astype('f4'),
                pred=pred,correct=(pred==np.asarray(labels)),fc_w=np.asarray(fc_w,dtype='f4'),fc_b=np.asarray(fc_b,dtype='f4'))
    for name,x in zip(SITES,channels):
        q,lo,hi=quantize_channels(x)
        result[f'chan_{name}']=q;result[f'chan_{name}_lo']=lo;result[f'chan_{name}_hi']=hi
    return result


def comparison(a,b,r):
    """Mean squared principal cosines; robust to nonorthogonal projected bases."""
    qa=torch.linalg.qr(a[:,:r]-a[:,:r].mean(0)).Q
    qb=torch.linalg.qr(b[:,:r]-b[:,:r].mean(0)).Q
    return float((qa.T@qb).square().sum()/r)


def fixture(root):
    """N=2000, real dense operators; repeated identical synthetic cells reuse results.

    The fixture intentionally has no variation over checkpoints/sites. This saves
    thousands of redundant eigensolves without inventing any spectra/transforms.
    It is a parser/UI fixture, never evidence about either trained network.
    """
    import time
    start=time.time(); root=Path(root);root.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1);torch.manual_seed(41)
    n=2000
    planted=torch.arange(n)%10
    x=torch.randn(10,64)[planted]*1.3 + .8*torch.randn(n,64)
    c=centred_cosine_gram(x)
    from sweep_math import final_kernel_bank
    from coherent import symmetric_balance
    bank=final_kernel_bank(c)
    infos=[info for _,info in bank]
    layers=allocate(5,78,n,12); times=allocate(5,36,n,8)
    for mi,(logk,info) in enumerate(bank):
        m=symmetric_balance(logk)*.5;m.diagonal().add_(.5)
        rec,v=block_record([m],0,0)
        for pi in range(78):
            for key in layers:
                val=rec[key]
                if key.startswith('retention_'): val=np.repeat(val,12,axis=-1)
                layers[key][mi,pi]=val
        for pi in range(36):
            for key in times:
                val=rec[key]
                if key.startswith('retention_'): val=np.repeat(val,8,axis=-1)
                times[key][mi,pi]=val
        print('fixture mode',mi,'seconds',round(time.time()-start,1),flush=True)
    for step in STEPS: np.savez_compressed(root/f'layer_{step}.npz',**layers)
    for site in SITES: np.savez_compressed(root/f'time_{site}.npz',**times)
    rng=np.random.default_rng(1)
    labels=rng.integers(0,10,n)
    w=rng.normal(0,.1,(10,64)); bias=np.zeros(10)
    logits=x.numpy()@w.T
    chans=[x.numpy()[:,:width] for width in CHANNELS[:-1]]+[logits]
    points=point_arrays(chans,logits,labels,w,bias)
    err=torch.softmax(torch.as_tensor(logits),1)-torch.nn.functional.one_hot(torch.as_tensor(labels),10)
    mean_w=err.T@x.double()/n;mean_b=err.mean(0)
    norm=(err.square().sum(1)*(x.double().square().sum(1)+1)).sqrt()
    dot=(err*(x.double()@mean_w.T+mean_b)).sum(1)
    mean_norm=(mean_w.square().sum()+mean_b.square().sum()).sqrt()
    points.update(jac_fro=np.full(n,np.linalg.norm(w),np.float32),grad_dot=dot.numpy().astype('f4'),
                  grad_cos=(dot/(norm*mean_norm)).numpy().astype('f4'),grad_norm=norm.numpy().astype('f4'))
    gradgram=(err@err.T)*(x.double()@x.double().T+1)/(norm[:,None]*norm[None,:])
    for step in STEPS:
        np.save(root/f'gradgram_{step}.npy',gradgram.numpy().astype('f2'))
        np.savez_compressed(root/f'points_{step}.npz',**points)
        np.save(root/f'umap_{step}.npy',np.tile(x[:,:2].numpy().astype('f4')[None],(12,1,1)))
    np.savez_compressed(root/'probe.npz',idx=np.arange(n),labels=labels,images=rng.integers(0,256,(n,32,32,3),dtype='u1'))
    def grid(key):return np.tile(np.array([info[key] for info in infos],dtype=object)[:,None,None],(1,8,12))
    (root/'meta.json').write_text(json.dumps(metadata(grid('kappa'),grid('nn_mass'),grid('self_weight'),grid('kappa_q10'),grid('kappa_q90'),fixture=True),indent=1))
    avgrows=[dict(step=t,mode=mi,a=a,b=b,r=r,right=1.,left=1.,end_right=1.,end_left=1.)
             for mi in (0,1) for t in STEPS for a,b in pairs(12) if a<b for r in (3,6)]
    # Identical cells: all powers have exactly the same eigenvectors.
    (root/'avgprod.json').write_text(json.dumps(dict(fixture=True,modes=[0,1],rows=avgrows)))
    from coherent import diffusion_operator
    rows=[]
    for mi in (3,4):
        k=infos[mi]['kappa']
        _,old=block_modes([diffusion_operator(c,k)],7)
        _,retained=block_modes([sinkhorn_operator(c,k)],7,shared_stationary=True)
        _,new=block_modes([fixed_lazy_operator(c,k)],7,shared_stationary=True)
        for a,b in pairs(12):
            for r in (3,6):rows.append(dict(step=64000,mode=mi,a=a,b=b,r=r,agreement=comparison(old[:,1:],new[:,1:],r),retained_agreement=comparison(retained[:,1:],new[:,1:],r)))
    (root/'normalisation_compare.json').write_text(json.dumps(dict(fixture=True,rows=rows)))
    old_steps=sorted(int(p.stem.split('_')[1]) for p in (root.parent/'r20s0').glob('stats_*.json')) or STEPS
    # Actual validation fits on synthetic data, reused across identical cells.
    xx=x.double(); yy=torch.nn.functional.one_hot(torch.as_tensor(labels),10).double()
    xx=torch.nn.functional.normalize(xx-xx[:1000].mean(0),dim=1)
    results=[]
    views={view:{} for view in ('flat','gap')}
    for site,width in zip(SITES,CHANNELS):
        z=torch.as_tensor(logits).double() if site=='logits' else xx[:,:width]
        weight=torch.linalg.solve(z[:1000].T@z[:1000]+torch.eye(width),z[:1000].T@yy[:1000])
        ridge=float(((z[1000:]@weight).argmax(1)==torch.as_tensor(labels[1000:])).double().mean())
        idx=(z[1000:]@z[:1000].T).topk(20,dim=1).indices
        knn=float((yy[idx].sum(1).argmax(1)==torch.as_tensor(labels[1000:])).double().mean())
        for view in views:views[view][site]=dict(ridge_acc=ridge,knn_acc=knn)
    for step in old_steps: results.append(dict(step=step,views=views))
    (root/'decod.json').write_text(json.dumps(dict(fixture=True,steps=old_steps,sites=SITES,views=['flat','gap'],results=results,protocol='SYNTHETIC FIXTURE ONLY: 1000/1000 split, ridge lambda=1, 20NN. Real data uses the full decodability.py protocol.')))
    all_steps=sorted(set(range(17))|{32,64,128,256,512,1024}|set(range(1280,64001,1280)))
    (root/'curves.json').write_text(json.dumps(dict(fixture=True,checkpoints=[dict(step=t,train_acc=.1,val_acc=.1,train_loss=2.4,val_loss=2.4) for t in all_steps])))
    print('fixture complete',root,round(time.time()-start,1),'seconds')


def validate(root, require_derivatives=True):
    root=Path(root);meta=json.loads((root/'meta.json').read_text());n=meta['n']
    assert meta['schema_version']==2 and n==2000
    assert meta.get('construction_fingerprint')==FINGERPRINT
    assert meta['steps']==STEPS and meta['sites']==SITES and meta['modes']==MODES
    assert meta['layer_pairs']==[list(p) for p in pairs(12)] and meta['time_pairs']==[list(p) for p in pairs(8)]
    assert meta['seba_r']==RANKS and meta['n_vecs']==8 and len(meta['classes'])==10
    assert np.asarray(meta['kappa']).shape==(5,8,12)
    assert np.asarray(meta['nn_mass']).shape==(5,8,12)
    assert np.asarray(meta['self_weight']).shape==(5,8,12)
    for field in ('kappa','kappa_q10','kappa_q90'):
        a=np.asarray(meta[field],dtype=object);assert a.shape==(5,8,12)
        assert all(v is None for v in a[0].ravel())
        assert np.isfinite(a[1:].astype(float)).all()
    sw=np.asarray(meta['self_weight'],dtype=object)
    assert all(v is None for v in sw[:3].ravel()) and np.isfinite(sw[3:].astype(float)).all()
    assert np.max(np.abs(np.asarray(meta['nn_mass'])[4]-.5))<1e-6
    min_agree=1.; blocks=0
    torch.set_num_threads(1)
    for family,names,npair,nc in [('layer',STEPS,78,12),('time',SITES,36,8)]:
        expected=allocate(5,npair,n,nc)
        for name in names:
            with np.load(root/f'{family}_{name}.npz') as arrays:
                assert set(arrays.files)==set(expected)
                for key,template in expected.items():
                    a=arrays[key];assert a.shape==template.shape,(name,key,a.shape)
                    assert a.dtype==template.dtype,(name,key,a.dtype)
                    if key!='retention_seba':assert np.isfinite(a).all(),(name,key)
                assert np.max(np.abs(arrays['vals'][:,:,0]-1))<1e-6
                assert np.all(np.diff(arrays['vals'],axis=-1)<=1e-6)
                vals=arrays['vecs'];scale=arrays['scale']
                rotations=arrays['seba_rot'];perms=arrays['seba_perm'];labels=arrays['seba_label'];ret=arrays['retention_seba']
                for mi in range(5):
                    for pi in range(npair):
                        v=np.concatenate([np.full((n,1),1/n**.5), vals[mi,pi].T.astype('f8')*scale[mi,pi]/32767],1)
                        assert np.max(np.abs(np.linalg.norm(v[:,1:],axis=0)-1))<.00002
                        assert np.all(vals[mi,pi].max(axis=1)==32767)
                        for ri,r in enumerate(RANKS):
                            assert ((labels[mi,pi,ri]>=-1)&(labels[mi,pi,ri]<r)).all()
                            assert np.all(rotations[mi,pi,ri,r:,:]==0) and np.all(rotations[mi,pi,ri,:,r:]==0)
                            p=perms[mi,pi,ri,:r];assert sorted(p)==list(range(r))
                            assert (perms[mi,pi,ri,r:]==-1).all()
                            assert np.isfinite(ret[mi,pi,ri,:r]).all() and np.isnan(ret[mi,pi,ri,r:]).all()
                            s=seba_reconstruct(torch.from_numpy(v[:,:r]),torch.from_numpy(rotations[mi,pi,ri,:r,:r]),torch.from_numpy(p.astype('i8')))
                            agree=float(np.mean(seba_labels(s).numpy()==labels[mi,pi,ri]));min_agree=min(min_agree,agree)
                            assert agree>=.99,(family,name,mi,pi,r,agree)
                        blocks+=1
    for step in STEPS:
        with np.load(root/f'points_{step}.npz') as p:
            expected=dict(loss=((n,),'f4'),pred=((n,),'i1'),correct=((n,),'?'),fc_w=((10,64),'f4'),fc_b=((10,),'f4'))
            if require_derivatives:
                expected.update({key:((n,),'f4') for key in ('jac_fro','grad_dot','grad_cos','grad_norm')})
            for site,width in zip(SITES,CHANNELS):
                expected[f'chan_{site}']=((n,width),'u1')
                for suffix in ('lo','hi'):expected[f'chan_{site}_{suffix}']=((width,),'f4')
            for key,(shape,dtype) in expected.items():
                assert p[key].shape==shape and p[key].dtype==np.dtype(dtype),(step,key)
                assert np.isfinite(p[key]).all(),(step,key)
        u=np.load(root/f'umap_{step}.npy');assert u.shape==(12,n,2) and u.dtype==np.float32 and np.isfinite(u).all()
        if require_derivatives:
            g=np.load(root/f'gradgram_{step}.npy')
            assert g.shape==(n,n) and g.dtype==np.float16 and np.isfinite(g).all()
            assert np.max(np.abs(g.diagonal().astype('f4')-1))<.001
            assert np.max(np.abs(g.astype('f4')-g.T.astype('f4')))<.001
            assert np.max(np.abs(g))<=1.001
    with np.load(root/'probe.npz') as p:
        assert p['idx'].shape==(n,) and p['labels'].shape==(n,) and p['images'].shape==(n,32,32,3)
    documents={}
    for file in ('avgprod.json','normalisation_compare.json','decod.json','curves.json'):
        documents[file]=json.loads((root/file).read_text(),parse_constant=lambda s:(_ for _ in ()).throw(ValueError(s)))
    avg=documents['avgprod.json']['rows']
    wanted={(mi,t,a,b,r) for mi in (0,1) for t in STEPS for a,b in pairs(12) if a<b for r in (3,6)}
    assert len(avg)==len(wanted) and {(row['mode'],row['step'],row['a'],row['b'],row['r']) for row in avg}==wanted
    for row in avg:
        for key in ('right','left','end_right','end_left'):assert -1e-10<=row[key]<=1+1e-10
    normal=documents['normalisation_compare.json']['rows']
    wanted={(mi,64000,a,b,r) for mi in (3,4) for a,b in pairs(12) for r in (3,6)}
    assert len(normal)==len(wanted) and {(row['mode'],row['step'],row['a'],row['b'],row['r']) for row in normal}==wanted
    for row in normal:
        for key in ('agreement','retained_agreement'):assert -1e-10<=row[key]<=1+1e-10
    decod=documents['decod.json'];steps=[0,4,16,64,256,1024,2560,6400,12800,25600,32000,33280,48640,64000]
    assert decod['steps']==steps and [r['step'] for r in decod['results']]==steps
    for row in decod['results']:
        for view in ('flat','gap'):
            assert set(row['views'][view])==set(SITES)
            for site in SITES:
                for score in ('ridge_acc','knn_acc'):assert 0<=row['views'][view][site][score]<=1
    checkpoints=documents['curves.json']['checkpoints']
    wanted=sorted(set(range(17))|{32,64,128,256,512,1024}|set(range(1280,64001,1280)))
    assert [r['step'] for r in checkpoints]==wanted
    for row in checkpoints:
        for key in ('train_acc','val_acc'):assert 0<=row[key]<=1
        for key in ('train_loss','val_loss'):assert np.isfinite(row[key]) and row[key]>=0
    report=dict(blocks=blocks,minimum_seba_label_agreement=min_agree,complete=True)
    print(json.dumps(report));return report


if __name__=='__main__':
    import sys
    if sys.argv[1]=='fixture':fixture(sys.argv[2])
    elif sys.argv[1]=='validate':validate(sys.argv[2])
