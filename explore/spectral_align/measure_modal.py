"""Detached spectral alignment pipeline: deploy, launch pilot|all, status, fetch.

Scratch full eigenvectors are float32 (~34 GB for two rules at 1,056 cells),
centred-cosine Grams float32 (~17 GB). Full eigenvalues remain float64.
Only one time's reconstructed kernels is resident in an all-pairs worker.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import modal

HERE=Path(__file__).resolve().parent
APP='cm-spectral-align'
STEPS=[0,64,256,1024,2560,6400,25600,64000]
NETS=[f'resnet{d}-relu-seed{s}' for d in (20,32,56) for s in (0,1)]
ROOT=Path('/out/work/salign')
PUBLIC=Path('/out/analysis/salign')
# Checked https://modal.com/pricing 2026-11-18. Conservative CPU: count vCPU as core.
GPU_RATE=.000583
CPU_RATE=.0000131
MEM_RATE=.00000222
RATE=GPU_RATE+2*CPU_RATE+16*MEM_RATE
PAIR_RATE=GPU_RATE+2*CPU_RATE+65536/1024*MEM_RATE
TIMEOUT=int(os.environ.get('SALIGN_TIMEOUT','1800'))
image=(modal.Image.debian_slim(python_version='3.12').pip_install('torch','numpy','scipy','dtw-python')
       .env({'PYTHONPATH':'/root/src:/root/sa:/root/cc:/root/cm','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'})
       .add_local_dir(str(Path.home()/'code/ModelZoo/src'),remote_path='/root/src'))
for name in ('align.py','nets.py'):
    image=image.add_local_file(str(HERE/name),remote_path='/root/sa/'+name)
for name in ('coherent.py','seba.py','sweep_math.py','construction.py'):
    image=image.add_local_file(str(HERE.parent/'cifar_coherent'/name),remote_path='/root/cc/'+name)
image=image.add_local_file(str(HERE.parent/'cifar_more/cm3.py'),remote_path='/root/cm/cm3.py')
app=modal.App(APP)
vol=modal.Volume.from_name('cm-cifar-coherent')
zoo=modal.Volume.from_name('crw-modelzoo-train').read_only()
VOLS={'/out':vol,'/zoo':zoo}


def env():
    import torch
    assert torch.cuda.is_available()
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    vol.reload();ROOT.mkdir(parents=True,exist_ok=True);PUBLIC.mkdir(parents=True,exist_ok=True)


def two_rules(c):
    import torch
    from sweep_math import perplexity_logkernel
    d=1-c.double();d.diagonal().fill_(float('inf'))
    near=d.topk(15,largest=False,dim=1)
    mask=torch.zeros_like(c,dtype=torch.bool);mask.scatter_(1,near.indices,True);mask=mask|mask.T
    weights=torch.where(mask,torch.ones_like(c),torch.full_like(c,1e-6));weights.diagonal().fill_(0)
    yield torch.log(weights)
    yield perplexity_logkernel(c,100)[0]


def operators(c):
    from coherent import symmetric_balance
    for logk in two_rules(c):
        p=symmetric_balance(logk);m=p*.5;m.diagonal().add_(.5)
        yield m


def operator(c,rule):
    from coherent import symmetric_balance
    bank=two_rules(c)
    logk=next(bank)
    if rule==1:logk=next(bank)
    p=symmetric_balance(logk);m=p*.5;m.diagonal().add_(.5)
    return m


def load_acts(net,step):
    import numpy as np
    import torch
    from zoo.data import build_data
    from zoo.io import load_checkpoint
    from nets import build_model,activations,site_names
    ck=load_checkpoint(Path(f'/out/runs/{net}/checkpoints/step_{step}.pt'))
    model=build_model(ck['config']);model.load_state_dict(ck['model_state']);model.eval().cuda()
    data=build_data(ck['config'],Path('/zoo/data'),'cpu')
    y=data.test_y.numpy();rng=np.random.RandomState(0)
    idx=np.sort(np.concatenate([rng.choice(np.where(y==c)[0],200,replace=False) for c in range(10)]))
    storage=[[] for _ in site_names(depth(net))]
    with torch.inference_mode():
        for at in range(0,len(idx),128):
            aa=activations(model,data.normalize(data.test_x[idx[at:at+128]]).cuda())
            for dst,a in zip(storage,aa):dst.append(a.cpu())
    return [torch.cat(a) for a in storage],idx,y[idx]


def depth(net):return int(net.split('-')[0][6:])
def unitdir(net,step):return ROOT/net/str(step)
def record(name,start,rate=RATE,**extra):
    row=dict(unit=name,seconds=time.time()-start,started_at=start,estimated_usd=(time.time()-start)*rate,**extra)
    (ROOT/f'timing_{name}.json').write_text(json.dumps(row));vol.commit();return row


@app.function(image=image,gpu='A100-40GB',cpu=2,memory=16384,timeout=TIMEOUT,max_containers=8,retries=0,volumes=VOLS)
def extract(net:str,step:int):
    import numpy as np
    import torch
    from coherent import centred_cosine_gram,block_modes
    from cm3 import split,subset_grams,stability
    from align import spectrum
    env();start=time.time();dst=unitdir(net,step);dst.mkdir(parents=True,exist_ok=True)
    done=dst/'done.json'
    legacy_failures=list(dst.glob('subset_failure_*.json'))
    legacy_failures=[p for p in legacy_failures if len(p.stem.split('_'))==3]
    if done.exists() and not legacy_failures:return json.loads(done.read_text())
    acts,idx,labels=load_acts(net,step)
    np.savez(dst/'probe.npz',idx=idx,labels=labels)
    s1,s2,c1,c2=split(len(idx))
    for si,a in enumerate(acts):
        target=dst/f'cell_{si}.npz'
        legacy=dst/f'subset_failure_{si}.json'
        if target.exists() and not legacy.exists():continue
        grams=subset_grams(a.cuda(),s1,s2)
        np.save(dst/f'gram_{si}.npy',grams[0].cpu().numpy().astype('f4'))
        es=[];spans=[]
        for rule in range(2):
            try:
                m=operator(grams[0],rule)
                e,u=spectrum(m,full_vectors=True)
                es.append(e.cpu().numpy())
                np.save(dst/f'fullU_{rule}_{si}.npy',u.cpu().numpy().astype('f4'))
            except (RuntimeError,ValueError,AssertionError) as error:
                if rule==0:raise
                es.append(np.full(len(idx),np.nan));spans.append(np.full(80,np.nan,dtype='f4'))
                (dst/f'rule1_unavailable_{si}.json').write_text(json.dumps(dict(net=net,step=step,site_index=si,error=repr(error))))
                continue
            try:
                subs=[block_modes([operator(g,rule)],81,shared_stationary=True)[1][:,1:] for g in grams[1:]]
                spans.append(stability(subs[0],subs[1],c1,c2,0)[0])
            except (RuntimeError,ValueError,AssertionError) as error:
                spans.append(np.full(80,np.nan,dtype='f4'))
                (dst/f'subset_failure_{si}_{rule}.json').write_text(json.dumps(dict(net=net,step=step,site_index=si,rule=rule,error=repr(error))))
        np.savez(target,eig=np.stack(es),span=np.stack(spans))
        if legacy.exists():legacy.unlink()
        print(net,step,si,'seconds',time.time()-start,flush=True)
        del grams,a
    row=record(f'extract_{net}_{step}',start,sites=len(acts),net=net,step=step)
    done.write_text(json.dumps(row));vol.commit();return row


@app.function(image=image,gpu='A100-40GB',cpu=2,memory=16384,timeout=900,max_containers=1,retries=0,volumes=VOLS)
def control6():
    import numpy as np
    import torch
    from coherent import centred_cosine_gram
    env();start=time.time();acts,idx,labels=load_acts(NETS[0],64000)
    si=0;c=centred_cosine_gram(acts[si].cuda())
    stored=np.load(f'/out/work/r20s0b/operators_64000_{si}.npy')
    errors=[float(np.max(np.abs(m.cpu().numpy()-stored[r]))) for r,m in enumerate(operators(c))]
    assert max(errors)<1e-9,errors
    old=np.load('/out/analysis/r20s0b/probe.npz')
    assert np.array_equal(old['idx'],idx)
    np.savez(PUBLIC/'probe.npz',idx=idx,labels=labels)
    return record('control6',start,max_errors=errors)


def manifests():
    return {net:json.loads(Path(f'/out/runs/{net}/manifest.json').read_text()) for net in NETS}


def selection(ms):
    result={}
    for net in NETS[2:]:
        seed=int(net[-1]);target=next(c['val_acc'] for c in ms[NETS[seed]]['checkpoints'] if c['step']==64000)
        candidates=sorted((c for c in ms[net]['checkpoints'] if c['val_acc']>=target),key=lambda c:c['step'])
        c=candidates[0] if candidates else None
        available=c is not None and c['val_acc']-target<=.005
        result[net]=dict(target=target,available=available,status='available' if available else 'unavailable',
                         step=c['step'] if c else None,accuracy=c['val_acc'] if c else None,
                         overshoot=c['val_acc']-target if c else None)
    return result


@app.function(image=image,cpu=2,memory=4096,timeout=1800,max_containers=1,retries=0,volumes=VOLS)
def assemble():
    import numpy as np
    from nets import site_names,site_stage
    vol.reload();ms=manifests();iso=selection(ms);cells=[];es=[[],[]];sp=[[],[]]
    for net in NETS:
        extra=iso.get(net,{})
        steps=sorted(set(STEPS+([extra['step']] if extra.get('available') else [])))
        for step in steps:
            dst=unitdir(net,step);export=[]
            for si,name in enumerate(site_names(depth(net))):
                rec=np.load(dst/f'cell_{si}.npz')
                for r in range(2):es[r].append(rec['eig'][r]);sp[r].append(rec['span'][r])
                cells.append(dict(index=len(cells),net=net,depth=depth(net),seed=int(net[-1]),step=step,
                                  is_iso_accuracy=bool(extra.get('available') and step==extra['step']),
                                  site_index=si,site_name=name,stage=site_stage(depth(net))[si],
                                  val_acc=next(c['val_acc'] for c in ms[net]['checkpoints'] if c['step']==step)))
                if step==64000 or net==NETS[0]:export.append(np.load(dst/f'fullU_0_{si}.npy',mmap_mode='r')[:,:64].copy())
            if export:np.save(PUBLIC/f'U_{net}_{step}.npy',np.stack(export))
    for r in range(2):
        np.save(PUBLIC/f'eig_{r}.npy',np.stack(es[r]));np.save(PUBLIC/f'span_{r}.npy',np.stack(sp[r]))
    unavailable=[]
    for c in cells:
        p=unitdir(c['net'],c['step'])/f'rule1_unavailable_{c["site_index"]}.json'
        if p.exists():unavailable.append(dict(**json.loads(p.read_text()),index=c['index'],site=c['site_name']))
    (PUBLIC/'rule1_unavailable.json').write_text(json.dumps(unavailable,indent=2))
    (PUBLIC/'subset_failures.json').write_text(json.dumps([json.loads(p.read_text()) for p in ROOT.glob('*/*/subset_failure_*.json')]))
    (PUBLIC/'cells.json').write_text(json.dumps(cells));(PUBLIC/'iso_accuracy.json').write_text(json.dumps(iso,indent=2))
    (PUBLIC/'accuracies.json').write_text(json.dumps({n:next(c['val_acc'] for c in m['checkpoints'] if c['step']==64000) for n,m in ms.items()}))
    vol.commit();return len(cells)


@app.function(image=image,gpu='A100-40GB',cpu=2,memory=65536,timeout=3600,max_containers=2,retries=0,volumes=VOLS)
def allpairs(rule:int):
    import numpy as np
    import torch
    from align import pairwise_spectral,heat_kernel,normalized_heat_kernel,cosine_distance,center_gram,T_GRID,streamed_kernel_gram
    env();start=time.time();allcells=json.loads((PUBLIC/'cells.json').read_text());total=len(allcells);n=2000
    ep=np.load(PUBLIC/f'eig_{rule}.npy');valid=np.flatnonzero(np.isfinite(ep).all(1))
    if rule==0:assert len(valid)==total
    cells=[allcells[i] for i in valid];nc=len(cells);e=torch.tensor(ep[valid],device='cuda')
    def filename(c,kind):return unitdir(c['net'],c['step'])/f'{kind}_{c["site_index"]}.npy'
    # Host arrays contain one rule's complete eigenbasis, then one t's kernels.
    full=np.empty((nc,n,n-1),dtype='f4')
    for i,c in enumerate(cells):full[i]=np.load(filename(c,f'fullU_{rule}'))
    top=torch.tensor(full[:,:,:64].copy(),device='cuda')
    proj,eigd=pairwise_spectral(e,top,chunk=8)
    proj=proj.cpu().numpy();eigd=eigd.cpu().numpy();del top
    heat=np.empty((nc,nc,len(T_GRID)),dtype='f4');kernels=np.empty((nc,n,n),dtype='f4');worst=0.
    rng=np.random.default_rng(18);pairs=[tuple(rng.integers(nc,size=2)) for _ in range(6)]
    def accumulate():
        return streamed_kernel_gram(nc,n,None,device='cuda',row_chunk=32,
                                    load_rows=lambda a,b:torch.from_numpy(kernels[:,a:b].copy())).cpu().numpy()
    for ti,t in enumerate(T_GRID):
        for i in range(nc):
            u=torch.tensor(full[i],device='cuda');kernels[i]=normalized_heat_kernel(e[i],u,t).cpu().numpy()
        heat[:,:,ti]=accumulate()
        for a,b in pairs:
            ua=torch.tensor(full[a],device='cuda',dtype=torch.float64);ub=torch.tensor(full[b],device='cuda',dtype=torch.float64)
            # Dense matrix powers of reconstructed M, independent of lambda**t.
            ma=(ua*e[a,1:])@ua.T;mb=(ub*e[b,1:])@ub.T
            dense=cosine_distance(torch.linalg.matrix_power(ma,t),torch.linalg.matrix_power(mb,t))
            worst=max(worst,abs(float(dense)-float(heat[a,b,ti])))
        assert worst<1e-4,worst
        print('pairs',rule,'t',t,'seconds',time.time()-start,'dense_error',worst,flush=True)
    def expand(a):
        out=np.full((total,total,*a.shape[2:]),np.nan,dtype='f4')
        out[valid[:,None],valid[None,:]]=a
        return out
    np.savez(PUBLIC/f'pairs_{rule}.npz',proj=expand(proj),eigd=expand(eigd),heat=expand(heat))
    if rule==0:
        for i,c in enumerate(cells):
            g=torch.tensor(np.load(filename(c,'gram')),device='cuda');kernels[i]=center_gram(g).cpu().numpy()
        np.save(PUBLIC/'cka.npy',accumulate())
    return record(f'pairs_{rule}',start,rate=PAIR_RATE,dense_check_max_error=worst,ncells=nc)


@app.function(image=image,cpu=.25,memory=512,timeout=14400,max_containers=1,retries=0,volumes=VOLS)
def coordinator(stage:str):
    vol.reload();ROOT.mkdir(parents=True,exist_ok=True);PUBLIC.mkdir(parents=True,exist_ok=True)
    start=time.time()
    if stage=='debug':
        return dict(control6=control6.remote(),extract=extract.remote(NETS[0],64000))
    if stage=='pilot':
        check=control6.remote()
        pilot=extract.remote('resnet56-relu-seed0',64000)
        # Exact planned site count, conservative 2*30min all-pairs allowance.
        ms=manifests();iso=selection(ms)
        sites=8*(2*12+2*18+2*30)+sum((18 if depth(n)==32 else 30) for n,x in iso.items() if x['available'] and x['step'] not in STEPS)
        projected=pilot['seconds']/30*sites*RATE+2*3600*PAIR_RATE+check['estimated_usd']+.5
        result=dict(pilot=pilot,control6=check,projected_usd=projected,sites=sites,approved=projected<30,
                    suggested_timeout=int(pilot['seconds']*3+30),prices_checked='2026-11-18',price_source='https://modal.com/pricing')
        (ROOT/'gate.json').write_text(json.dumps(result,indent=2));vol.commit();return result
    if stage!='all':raise ValueError(stage)
    gate=json.loads((ROOT/'gate.json').read_text());assert gate['approved'] and gate['projected_usd']<30
    ms=manifests();iso=selection(ms);units=[]
    for net in NETS:
        extra=iso.get(net,{})
        for step in sorted(set(STEPS+([extra['step']] if extra.get('available') else []))):units.append((net,step))
    outcomes=list(extract.starmap(units,return_exceptions=True))
    failures=[dict(net=unit[0],step=unit[1],error=repr(result)) for unit,result in zip(units,outcomes) if isinstance(result,BaseException)]
    vol.reload();(ROOT/'failures.json').write_text(json.dumps(failures));vol.commit()
    if failures:raise RuntimeError(f'Extraction failed; successful units retained. See failures.json: {failures}')
    assemble.remote()
    list(allpairs.map([0,1]))
    vol.reload()
    rows=[json.loads(p.read_text()) for p in ROOT.glob('timing_*.json')]
    report=dict(units=rows,gate=gate,total_estimated_usd=sum(r['estimated_usd'] for r in rows),coordinator_wall_seconds=time.time()-start)
    (PUBLIC/'timings.json').write_text(json.dumps(report,indent=2));vol.commit();return report


if __name__=='__main__':
    args=sys.argv[1:];cmd=args[0] if args else 'status';runs=HERE/'runs';runs.mkdir(exist_ok=True)
    if cmd=='deploy':subprocess.run([sys.executable,'-m','modal','deploy',__file__],check=True)
    elif cmd=='launch':
        stage=args[1];call=modal.Function.from_name(APP,'coordinator').spawn(stage)
        rec=dict(stage=stage,call_id=call.object_id,launched=time.time());(runs/f'measure-{stage}.json').write_text(json.dumps(rec));print(json.dumps(rec))
    elif cmd=='status':
        for p in sorted(runs.glob('measure-*.json')):
            rec=json.loads(p.read_text())
            try:print(rec['stage'],json.dumps(modal.FunctionCall.from_id(rec['call_id']).get(timeout=0)))
            except TimeoutError:print(rec['stage'],'running',rec['call_id'])
    elif cmd=='fetch':
        dst=HERE/'data';dst.mkdir(exist_ok=True)
        for entry in vol.iterdir('analysis/salign'):
            path=entry.path;target=dst/Path(path).name
            with target.open('wb') as f:
                for chunk in vol.read_file(path):f.write(chunk)
            print(target.name,target.stat().st_size)
    else:raise ValueError(cmd)
