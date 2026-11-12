"""Schema-v2 detached Modal pipeline. See SPEC.md section 2 and README.

Commands: deploy | launch [stage] | status | fetch [tag]
Stages: all (seed0 core), derivatives, layouts (seed1), validate.
Independent units are checkpoints (extraction, layer blocks, products, decodability)
and sites (time blocks). All scratch stays on the volume under /work/<tag>;
only public schema files are in /analysis/<tag>.
"""
from pathlib import Path
import json
import sys
import subprocess
import time
import modal
from construction import FINGERPRINT,SWEEP_FINGERPRINT,cached_completion

HERE=Path(__file__).resolve().parent
APP='cm-cifar-coherent-v2'
TAG='r20s0b'
STEPS=[0,64,256,1024,2560,6400,25600,64000]
DECODE_STEPS=[0,4,16,64,256,1024,2560,6400,12800,25600,32000,33280,48640,64000]
SITES=['stem']+[f'b{i}' for i in range(1,10)]+['pool','logits']
RUN='resnet20-relu-seed0'
image=(modal.Image.debian_slim(python_version='3.12')
       .pip_install('torch','numpy','scipy','umap-learn')
       .env({'PYTHONPATH':'/root/src:/root/cc','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'})
       .add_local_dir(str(Path.home()/'code/ModelZoo/src'),remote_path='/root/src'))
for name in ('coherent.py','seba.py','schema_v2.py','decodability.py','sweep_math.py','construction.py'):
    image=image.add_local_file(str(HERE/name),remote_path='/root/cc/'+name)
app=modal.App(APP)
vol=modal.Volume.from_name('cm-cifar-coherent')
zoo=modal.Volume.from_name('crw-modelzoo-train').read_only()
VOLS={'/out':vol,'/zoo':zoo}
# Prices checked https://modal.com/pricing, 2026-11-12. Conservative CPU charge
# treats each requested vCPU as a full physical core; excludes cold-start time.
GPU_RATE=.000583  # A100-40GB; conservative upper bound for remaining L40S units
CPU_RATE=.0000131
MEM_RATE=.00000222


def environment():
    import torch
    assert torch.cuda.is_available()
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False


def paths(tag=TAG):
    public=Path('/out/analysis')/tag;work=Path('/out/work')/tag
    public.mkdir(parents=True,exist_ok=True);work.mkdir(parents=True,exist_ok=True)
    return public,work


def model_data(run,step):
    import numpy as np
    import torch
    from zoo.data import build_data
    from zoo.io import load_checkpoint
    from zoo.models import build_model
    ck=load_checkpoint(Path(f'/out/runs/{run}/checkpoints/step_{step}.pt'))
    model=build_model(ck['config']);model.load_state_dict(ck['model_state']);model.eval().cuda()
    data=build_data(ck['config'],Path('/zoo/data'),'cpu')
    y=data.test_y.numpy();rng=np.random.RandomState(0)
    idx=np.sort(np.concatenate([rng.choice(np.where(y==c)[0],200,replace=False) for c in range(10)]))
    return model,data,idx


def activations(model,x):
    h=model.act(model.bn1(model.conv1(x)));acts=[h]
    for block in model.blocks:h=block(h);acts.append(h)
    pool=h.mean((2,3));return acts+[pool,model.fc(pool)]


def finished(work,name,start,cpu=2,mem=12,gpu=True,**extra):
    seconds=time.time()-start
    row=dict(unit=name,seconds=seconds,estimated_usd=seconds*((GPU_RATE if gpu else 0)+cpu*CPU_RATE+mem*MEM_RATE),
             started_at=start,completed_at=time.time(),**extra)
    if name.startswith(('extract_final5_','layer_','time_','products_')) or name in ('normalization','validation'):
        row['construction_fingerprint']=FINGERPRINT
    (work/f'timing_{name}.json').write_text(json.dumps(row,allow_nan=False));vol.commit()
    return row


@app.function(image=image,gpu='A100-40GB',cpu=2,memory=12288,timeout=900,max_containers=8,retries=0,volumes=VOLS)
def extract(step:int,run:str=RUN,tag:str=TAG,pcs_only:bool=False):
    import numpy as np
    import torch
    from coherent import centred_cosine_gram,symmetric_balance
    from sweep_math import final_kernel_bank
    from schema_v2 import point_arrays
    environment();vol.reload();root,work=paths(tag);name=f'extract_{step}' if pcs_only else f'extract_final5_{step}';start=time.time()
    required=[root/f'pcs_{step}.npy'] if pcs_only else [work/f'operators_{step}_{si}.npy' for si in range(12)]
    cached=cached_completion(work/f'timing_{name}.json',required,require_fingerprint=not pcs_only)
    if cached:return cached
    model,data,idx=model_data(run,step)
    storage=[[] for _ in SITES]
    with torch.inference_mode():
        for at in range(0,len(idx),128):
            aa=activations(model,data.normalize(data.test_x[idx[at:at+128]]).cuda())
            for dst,a in zip(storage,aa):dst.append(a.cpu())
    acts=[torch.cat(a) for a in storage]
    if pcs_only:
        pcs=[]
        for a in acts:
            c=centred_cosine_gram(a.cuda())
            val,v=torch.linalg.eigh(c);pcs.append((v[:,-50:]*val[-50:].clamp_min(0).sqrt()).flip(1).cpu().numpy())
        np.save(root/f'pcs_{step}.npy',np.stack(pcs).astype('f4'))
    else:
        info_rows=[];near_one=[]
        for si,a in enumerate(acts):
            c=centred_cosine_gram(a.cuda())
            bank=final_kernel_bank(c);info_rows.append([info for _,info in bank])
            matrices=[]
            for logk,info in bank:
                p,diagnostics=symmetric_balance(logk,return_info=True)
                m=p*.5;m.diagonal().add_(.5);matrices.append(m)
                info.update(sinkhorn_iterations=diagnostics['iterations'],row_error=diagnostics['row_error'])
            if step==64000:near_one.append([int((torch.linalg.eigvalsh(m)>.995).sum())-1 for m in matrices])
            operators=np.stack([m.cpu().numpy() for m in matrices])
            np.save(work/f'operators_{step}_{si}.npy',operators)
            np.save(work/f'gram_{step}_{si}.npy',c.cpu().numpy())
            print('extracted',step,SITES[si],flush=True)
        fields={key:[[info[key] for info in cell] for cell in info_rows] for key in ('kappa','kappa_q10','kappa_q90','nn_mass','self_weight','sinkhorn_iterations','row_error')}
        (work/f'kappa_{step}.json').write_text(json.dumps(dict(**fields,
            nontrivial_eigenvalues_above_0995=near_one,variant='final5 zero-diagonal fixed-lazy')))
        channels=[a.mean((2,3)).numpy() if a.ndim==4 else a.numpy() for a in acts]
        points=point_arrays(channels,acts[-1].numpy(),data.test_y[idx].numpy(),model.fc.weight.detach().cpu().numpy(),model.fc.bias.detach().cpu().numpy())
        if (root/f'points_{step}.npz').exists():
            with np.load(root/f'points_{step}.npz') as previous:
                for key in ('jac_fro','grad_dot','grad_cos','grad_norm'):
                    if key in previous:points[key]=previous[key]
        np.savez_compressed(root/f'points_{step}.npz',**points)
    if step==0:
        np.savez_compressed(root/'probe.npz',idx=idx,labels=data.test_y[idx].numpy(),images=data.test_x[idx].permute(0,2,3,1).numpy())
    return finished(work,name,start)


@app.function(image=image,gpu='A100-40GB',cpu=2,memory=12288,timeout=1800,max_containers=8,retries=0,volumes=VOLS)
def spectra(family:str,index:int,tag:str=TAG):
    import numpy as np
    import torch
    from schema_v2 import allocate,pairs,block_record,put_record
    environment();vol.reload();root,work=paths(tag);start=time.time()
    name=f'{family}_{STEPS[index] if family=="layer" else SITES[index]}'
    cached=cached_completion(work/f'timing_{name}.json',[root/f'{name}.npz'])
    if cached:return cached
    nc=12 if family=='layer' else 8
    ps=pairs(nc);arrays=allocate(5,len(ps),2000,nc)
    for mi in range(5):
        ms=[]
        for j in range(nc):
            step,si=(STEPS[index],j) if family=='layer' else (STEPS[j],index)
            m=np.load(work/f'operators_{step}_{si}.npy',mmap_mode='r')[mi]
            ms.append(torch.tensor(m,device='cuda',dtype=torch.float64))
        for pi,(a,b) in enumerate(ps):
            rec,_=block_record(ms,a,b)
            put_record(arrays,mi,pi,rec)
        print(name,'mode',mi,'seconds',time.time()-start,flush=True)
    np.savez_compressed(root/f'{name}.npz',**arrays)
    return finished(work,name,start)


@app.function(image=image,gpu='A100-40GB',cpu=2,memory=12288,timeout=1200,max_containers=8,retries=0,volumes=VOLS)
def products(step:int,tag:str=TAG):
    import numpy as np
    import torch
    from coherent import block_modes
    from schema_v2 import comparison
    environment();vol.reload();root,work=paths(tag);start=time.time();name=f'products_{step}'
    cached=cached_completion(work/f'timing_{name}.json',[work/f'avgprod_{step}.json'])
    if cached:return cached
    # Explicit constant projection (left AND right) prevents the trivial singular
    # value 1 from swamping weak nontrivial directions in a long product.
    def center(q):return q-q.mean(0,keepdim=True)-q.mean(1,keepdim=True)+q.mean()
    rows=[]
    for mi in (0,1):
        ms=[torch.tensor(np.load(work/f'operators_{step}_{si}.npy',mmap_mode='r')[mi],device='cuda',dtype=torch.float64) for si in range(12)]
        cs=[center(m) for m in ms]
        for a in range(11):
            q=cs[a]
            for b in range(a+1,12):
                q=cs[b]@q
                end=cs[b]@cs[a]
                _,v=block_modes(ms[a:b+1],7,shared_stationary=True);v=v[:,1:]
                u,sv,vh=torch.linalg.svd(q,full_matrices=False)
                ue,se,vhe=torch.linalg.svd(end,full_matrices=False)
                for r in (3,6):
                    rows.append(dict(step=step,mode=mi,a=a,b=b,r=r,right=comparison(v,vh.T,r),left=comparison(v,u,r),
                                     end_right=comparison(v,vhe.T,r),end_left=comparison(v,ue,r),
                                     product_sigma_r=float(sv[r-1]),end_sigma_r=float(se[r-1])))
    (work/f'avgprod_{step}.json').write_text(json.dumps(rows,allow_nan=False))
    return finished(work,name,start)


@app.function(image=image,gpu='A100-40GB',cpu=2,memory=12288,timeout=1800,max_containers=1,retries=0,volumes=VOLS)
def normalization(tag:str=TAG):
    import numpy as np
    import torch
    from coherent import block_modes,diffusion_operator,sinkhorn_operator
    from schema_v2 import comparison,pairs
    environment();vol.reload();root,work=paths(tag);start=time.time();name='normalization'
    cached=cached_completion(work/f'timing_{name}.json',[root/'normalisation_compare.json'])
    if cached:return cached
    ks=json.loads((work/'kappa_64000.json').read_text())['kappa'];rows=[]
    for mi in (3,4):
        grams=[torch.tensor(np.load(work/f'gram_64000_{si}.npy'),device='cuda',dtype=torch.float64) for si in range(12)]
        old=[diffusion_operator(c,ks[si][mi]) for si,c in enumerate(grams)]
        retained=[sinkhorn_operator(c,ks[si][mi]) for si,c in enumerate(grams)]
        new=[torch.tensor(np.load(work/f'operators_64000_{si}.npy',mmap_mode='r')[mi],device='cuda',dtype=torch.float64) for si in range(12)]
        for a,b in pairs(12):
            _,vo=block_modes(old[a:b+1],7);_,vn=block_modes(new[a:b+1],7,shared_stationary=True)
            _,vr=block_modes(retained[a:b+1],7,shared_stationary=True)
            for r in (3,6):rows.append(dict(step=64000,mode=mi,a=a,b=b,r=r,agreement=comparison(vo[:,1:],vn[:,1:],r),
                                            retained_agreement=comparison(vr[:,1:],vn[:,1:],r)))
    (root/'normalisation_compare.json').write_text(json.dumps(dict(rows=rows,
        convention='Discard each leading eigenvector (analytical constant for the Sinkhorn constructions; only a leading-vector convention for the legacy average, which has no shared stationary mode), project the remaining columns off the constant, orthonormalize, compare mean squared principal cosines.'),allow_nan=False))
    return finished(work,name,start)


@app.function(image=image,gpu='L40S',cpu=4,memory=32768,timeout=1250,max_containers=4,retries=0,volumes=VOLS)
def decode(step:int,tag:str=TAG):
    environment();vol.reload();root,work=paths(tag);start=time.time();name=f'decode_{step}'
    if (work/f'timing_{name}.json').exists():return json.loads((work/f'timing_{name}.json').read_text())
    from decodability import measure
    # Invoke the unchanged fit/evaluation protocol locally inside this GPU unit.
    res=measure.local(step,tag,RUN)
    for site in ('pool','logits'):res['views']['gap'][site]=res['views']['flat'][site]
    res['protocol']+=' In schema v2, GAP at pool/logits is an alias of flat (these sites already have no spatial dimensions).'
    (work/f'decod_{step}.json').write_text(json.dumps(res,allow_nan=False))
    # Intermediate protocol outputs are not part of the public schema.
    for suffix in ('','.json','_partial.json'):
        path=root/f'decodability_{step}{suffix}'
        if path.is_file():path.unlink()
    return finished(work,name,start,cpu=4,mem=32)


@app.function(image=image,gpu='L40S',cpu=2,memory=16384,timeout=900,max_containers=8,retries=0,volumes=VOLS)
def derivatives(step:int,tag:str=TAG):
    import numpy as np
    import torch
    from torch.func import functional_call,grad,jacrev,vmap
    environment();vol.reload();root,work=paths(tag);start=time.time();name=f'derivatives_{step}'
    if (work/f'timing_{name}.json').exists():return json.loads((work/f'timing_{name}.json').read_text())
    model,data,idx=model_data(RUN,step)
    params={n:p.detach() for n,p in model.named_parameters() if p.requires_grad}
    buffers={n:b.detach() for n,b in model.named_buffers()}
    x=data.normalize(data.test_x[idx]).cuda();y=data.test_y[idx].cuda()
    def logits(p,b,im):return functional_call(model,(p,b),(im.unsqueeze(0),)).squeeze(0)
    def loss(p,b,im,lab):return torch.nn.functional.cross_entropy(logits(p,b,im).unsqueeze(0),lab.unsqueeze(0))
    pergrad=vmap(grad(loss),in_dims=(None,None,0,0))
    perjac=vmap(jacrev(logits,argnums=2),in_dims=(None,None,0))
    g=torch.empty((len(idx),sum(p.numel() for p in params.values())),device='cuda')
    jacnorm=torch.empty(len(idx),device='cuda')
    for at in range(0,len(idx),16):
        dd=pergrad(params,buffers,x[at:at+16],y[at:at+16])
        g[at:at+16]=torch.cat([a.flatten(1) for a in dd.values()],1)
        jj=perjac(params,buffers,x[at:at+16]);jacnorm[at:at+16]=jj.flatten(1).norm(dim=1)
    g=g.double()
    norm=g.norm(dim=1);mean=g.mean(0);mean_norm=mean.norm()
    if bool((norm==0).any()) or float(mean_norm)==0:raise ValueError('Gradient cosine undefined for zero gradient')
    dot=g@mean;cos=dot/(norm*mean_norm)
    gn=g/norm[:,None];gram=(gn@gn.T).clamp(-1,1)
    assert float((gram.diag()-1).abs().max())<1e-5
    np.save(root/f'gradgram_{step}.npy',gram.cpu().numpy().astype('f2'))
    with np.load(root/f'points_{step}.npz') as p:points={k:p[k] for k in p.files}
    points.update(jac_fro=jacnorm.cpu().numpy().astype('f4'),grad_dot=dot.cpu().numpy().astype('f4'),
                  grad_cos=cos.cpu().numpy().astype('f4'),grad_norm=norm.cpu().numpy().astype('f4'))
    np.savez_compressed(root/f'points_{step}.npz',**points)
    return finished(work,name,start,mem=16)


@app.function(image=image,cpu=4,memory=8192,timeout=1200,max_containers=8,retries=0,volumes=VOLS)
def embed(step:int,tag:str='r20s1b'):
    import numpy as np
    import umap
    vol.reload();root,work=paths(tag);start=time.time();name=f'embed_{step}'
    if (work/f'timing_{name}.json').exists():return json.loads((work/f'timing_{name}.json').read_text())
    pcs=np.load(root/f'pcs_{step}.npy');relations=[{i:i for i in range(2000)} for _ in range(11)]
    model=umap.AlignedUMAP(n_neighbors=15,min_dist=.1,random_state=0,alignment_regularisation=.02)
    model.fit(list(pcs),relations=relations)
    np.save(work/f'umap_raw_{step}.npy',np.stack(model.embeddings_).astype('f4'))
    return finished(work,name,start,cpu=4,mem=8,gpu=False)


@app.function(image=image,cpu=2,memory=4096,timeout=1800,volumes=VOLS)
def validation(tag=TAG,require_derivatives=True):
    from schema_v2 import validate
    vol.reload();root,work=paths(tag);start=time.time()
    result=validate(root,require_derivatives)
    (root/'validation.json').write_text(json.dumps(result))
    return finished(work,'validation',start,gpu=False,mem=4,**result)


@app.function(image=image,gpu='A100-40GB',cpu=2,memory=12288,timeout=420,max_containers=8,retries=0,volumes=VOLS)
def sweep_cell(step:int,si:int):
    import numpy as np
    import torch
    from coherent import centred_cosine_gram
    from sweep_math import summarize
    environment();vol.reload();root,work=paths();start=time.time();name=f'sweep_{step}_{si}'
    cached=cached_completion(work/f'timing_{name}.json',[work/f'{name}.json'],expected_fingerprint=SWEEP_FINGERPRINT)
    if cached:return cached
    cache=work/f'gram_{step}_{si}.npy'
    model,data,idx=model_data(RUN,step)
    if cache.exists():c=torch.tensor(np.load(cache),device='cuda',dtype=torch.float64)
    else:
        xx=[]
        with torch.inference_mode():
            for at in range(0,len(idx),128):
                xx.append(activations(model,data.normalize(data.test_x[idx[at:at+128]]).cuda())[si].cpu())
        c=centred_cosine_gram(torch.cat(xx).cuda());np.save(cache,c.cpu().numpy())
    rows=summarize(c,data.test_y[idx].numpy())
    for row in rows:row.update(step=step,site=SITES[si],site_index=si)
    (work/f'{name}.json').write_text(json.dumps(rows,allow_nan=False));vol.commit()
    seconds=time.time()-start
    result=dict(unit=name,seconds=seconds,estimated_usd=seconds*(.000583+2*CPU_RATE+12*MEM_RATE),
                construction_fingerprint=SWEEP_FINGERPRINT,started_at=start,completed_at=time.time())
    (work/f'timing_{name}.json').write_text(json.dumps(result));vol.commit();return result


@app.function(image=image,cpu=1,memory=2048,timeout=300,volumes=VOLS)
def audit_provenance(tag=TAG):
    """One-time audited migration of the in-flight final-five completion records.

    Never infer a fingerprint from shapes. This migration requires EVERY relevant
    output and completion timestamp to follow the last final-five extraction;
    it records that provenance was adopted by audit rather than emitted at run
    time. Future reuse accepts only exact fingerprints, without this migration.
    """
    import numpy as np
    from schema_v2 import MODES
    vol.reload();root,work=paths(tag)
    extraction=[work/f'timing_extract_final5_{s}.json' for s in STEPS]
    assert all(p.exists() for p in extraction)
    cutoff=max(json.loads(p.read_text()).get('completed_at',p.stat().st_mtime) for p in extraction)
    units=[]
    for family,names in [('layer',STEPS),('time',SITES)]:
        for n in names:units.append((work/f'timing_{family}_{n}.json',root/f'{family}_{n}.npz'))
    units += [(work/f'timing_products_{s}.json',work/f'avgprod_{s}.json') for s in STEPS]
    units += [(work/'timing_normalization.json',root/'normalisation_compare.json')]
    stale=[];rows=[]
    for marker,output in units:
        assert marker.is_file() and output.is_file(),(str(marker),str(output))
        mtime=marker.stat().st_mtime;otime=output.stat().st_mtime
        if mtime<cutoff or otime<cutoff:stale.append(marker.name)
        rows.append(dict(marker=marker.name,output=output.name,marker_mtime=mtime,output_mtime=otime))
    assert not stale,('Stale construction results: rerun these units',stale)
    for aggregate in (root/'avgprod.json',root/'meta.json'):
        assert aggregate.is_file() and aggregate.stat().st_mtime>=cutoff,('Stale aggregate',str(aggregate))
    meta=json.loads((root/'meta.json').read_text());assert meta['modes']==MODES
    for marker,output in units:
        if output.suffix=='.npz':
            with np.load(output) as z:assert z['vecs'].dtype==np.int16
    for path in extraction+[m for m,_ in units]:
        data=json.loads(path.read_text())
        if data.get('construction_fingerprint')!=FINGERPRINT:
            data.update(construction_fingerprint=FINGERPRINT,provenance_adopted_by_timestamp_audit=True,
                        completed_at=path.stat().st_mtime)
            path.write_text(json.dumps(data))
    meta['construction_fingerprint']=FINGERPRINT;(root/'meta.json').write_text(json.dumps(meta,indent=1))
    report=dict(construction_fingerprint=FINGERPRINT,latest_final_extraction_mtime=cutoff,
                stale_units=stale,checked_downstream_units=len(units),checked_extractions=len(extraction),checked_aggregate_outputs=2,rows=rows,
                final_five_launch='fc-01M77KA5BY6WGXZ3N1TM1CTZFQ')
    (work/'provenance_audit.json').write_text(json.dumps(report,indent=1));vol.commit();return report


@app.function(image=image,cpu=.25,memory=1024,timeout=12*3600,volumes=VOLS)
def coordinate(stage='all'):
    import numpy as np
    import shutil
    from schema_v2 import metadata,MODES
    root,work=paths(TAG)
    log=work/'coordinator.log'
    def say(text):
        with log.open('a') as f:f.write(time.strftime('%Y-%m-%d %H:%M:%S ')+text+'\n')
        print(text,flush=True);vol.commit()
    def spent():
        vol.reload()
        return sum(json.loads(p.read_text())['estimated_usd'] for p in work.glob('timing_*.json'))
    def guard():
        usd=spent();say(f'estimated completed compute ${usd:.3f}')
        if usd>18:raise RuntimeError('Pause at $18 seed0 compute estimate; leaves reserve under $25 total ceiling.')
    if stage in ('all','extract'):
        say('extract checkpoints');list(extract.map(STEPS));guard()
        vol.reload()
        kr=[json.loads((work/f'kappa_{s}.json').read_text()) for s in STEPS]
        k=np.array([r['kappa'] for r in kr]).transpose(2,0,1);mass=np.array([r['nn_mass'] for r in kr]).transpose(2,0,1)
        sw=np.array([r['self_weight'] for r in kr],dtype=object).transpose(2,0,1)
        q10=np.array([r['kappa_q10'] for r in kr],dtype=object).transpose(2,0,1)
        q90=np.array([r['kappa_q90'] for r in kr],dtype=object).transpose(2,0,1)
        (root/'meta.json').write_text(json.dumps(metadata(k,mass,sw,q10,q90),indent=1,allow_nan=False))
        source=Path('/out/analysis/r20s0')
        # Use exactly the existing layouts, not a re-fit or an extra alignment.
        for step in STEPS:shutil.copy2(source/f'umap_{step}.npy',root/f'umap_{step}.npy')
        assert np.array_equal(np.load(source/'probe.npz')['idx'],np.load(root/'probe.npz')['idx'])
        shutil.copy2(source/'probe.npz',root/'probe.npz')
        manifest=json.loads(Path(f'/out/runs/{RUN}/manifest.json').read_text())
        (root/'curves.json').write_text(json.dumps(manifest,allow_nan=False));vol.commit()
    if stage in ('all','spectra'):
        say('layer and time spectra')
        jobs=[('layer',i,TAG) for i in range(8)]+[('time',i,TAG) for i in range(12)]
        list(spectra.starmap(jobs));guard()
    if stage in ('all','comparisons'):
        say('average/product and normalization comparisons')
        normcall=normalization.spawn(TAG)
        list(products.map(STEPS));normcall.get();guard();vol.reload()
        rows=sum([json.loads((work/f'avgprod_{step}.json').read_text()) for step in STEPS],[])
        (root/'avgprod.json').write_text(json.dumps(dict(modes=[0,1],rows=rows),allow_nan=False));vol.commit()
    if stage in ('all','decod'):
        say('decodability validation fits');list(decode.map(DECODE_STEPS));guard();vol.reload()
        results=[json.loads((work/f'decod_{step}.json').read_text()) for step in DECODE_STEPS]
        (root/'decod.json').write_text(json.dumps(dict(steps=DECODE_STEPS,sites=SITES,views=['flat','gap'],results=results,
            protocol=results[0]['protocol'],validation_only=True),allow_nan=False));vol.commit()
    if stage=='derivatives':
        say('per-sample derivatives and gradient cosine Grams');list(derivatives.map(STEPS));guard()
    if stage=='layouts':
        tag='r20s1b';rr,ww=paths(tag)
        say('seed1 extraction and layouts')
        list(extract.starmap([(s,'resnet20-relu-seed1',tag,True) for s in STEPS]))
        list(embed.map(STEPS));vol.reload()
        # Match build_viewer.py: centre and unit-RMS each site, orthogonal
        # Procrustes along checkpoints (reflections allowed).
        previous=None
        for step in STEPS:
            e=np.load(ww/f'umap_raw_{step}.npy');e=e-e.mean(1,keepdims=True)
            e/=np.sqrt((e**2).sum(2).mean(1))[:,None,None]
            if previous is not None:
                for si in range(12):
                    u,_,vt=np.linalg.svd(e[si].T@previous[si]);e[si]=e[si]@(u@vt)
            np.save(rr/f'umap_{step}.npy',e.astype('f4'));previous=e
        man=json.loads(Path('/out/runs/resnet20-relu-seed1/manifest.json').read_text())
        cp={r['step']:r for r in man['checkpoints']}
        (rr/'meta.json').write_text(json.dumps(dict(steps=STEPS,sites=SITES,n=2000,seed=1,
            val_acc=[cp[s].get('val_acc') for s in STEPS],layout='AlignedUMAP across sites, centred unit-RMS and orthogonal Procrustes across checkpoints',layout_only=True)))
        vol.commit()
    if stage=='sweep':
        say('19-rule sweep on 15 checkpoint/site cells')
        jobs=[(step,si) for step in (256,6400,64000) for si in (2,5,8,10,11)]
        list(sweep_cell.starmap(jobs));guard();vol.reload()
        rows=sum([json.loads((work/f'sweep_{step}_{si}.json').read_text()) for step,si in jobs],[])
        (root/'sweep.json').write_text(json.dumps(dict(rows=rows,n=2000,construction='zero diagonal symmetric scaling, M=(I+P)/2',
            validation='Hungarian-matched set/class Jaccard; labels are used only after all spectral construction',
            sinkhorn='10 damped log-domain warmups followed by line-searched Newton; strict row residual <2e-12'),allow_nan=False));vol.commit()
    if stage=='audit':audit_provenance.remote()
    if stage=='validate':validation.remote()
    say(f'{stage} complete; completed seed0 estimate ${spent():.3f}')
    return dict(stage=stage,estimated_seed0_usd=spent())


if __name__=='__main__':
    args=sys.argv[1:];recdir=HERE/'runs';recdir.mkdir(exist_ok=True)
    command=args[0] if args else 'deploy'
    if command=='deploy':subprocess.run([sys.executable,'-m','modal','deploy',__file__],check=True)
    elif command=='launch':
        stage=args[1] if len(args)>1 else 'all'
        call=modal.Function.from_name(APP,'coordinate').spawn(stage)
        rec=dict(stage=stage,call_id=call.object_id,launched=time.ctime())
        (recdir/f'v2-{stage}.json').write_text(json.dumps(rec,indent=1));print(json.dumps(rec))
    elif command=='status':
        stage=args[1] if len(args)>1 else 'all';rec=json.loads((recdir/f'v2-{stage}.json').read_text())
        try:print(json.dumps(modal.FunctionCall.from_id(rec['call_id']).get(timeout=0)))
        except TimeoutError:
            print('running')
            try:print(b''.join(vol.read_file(f'work/{TAG}/coordinator.log')).decode()[-3000:])
            except Exception as e:print(type(e).__name__)
    elif command=='fetch':
        tag=args[1] if len(args)>1 else TAG
        subprocess.run([sys.executable,'-m','modal','volume','get','--force','cm-cifar-coherent',f'analysis/{tag}',str(HERE/'data')],check=True)
    else:raise SystemExit(__doc__)
