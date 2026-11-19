"""Post-fetch checks; NaNs only in explicitly documented unavailable measurements."""
import json
from pathlib import Path
import numpy as np

root=Path(__file__).resolve().parent/'data'
cells=json.loads((root/'cells.json').read_text());n=len(cells)
unavailable=json.loads((root/'rule1_unavailable.json').read_text())
missing={x['index'] for x in unavailable}
subset=json.loads((root/'subset_failures.json').read_text())
subset_keys={(x['net'],x['step'],x['site_index'],x['rule']) for x in subset}
for r in range(2):
    valid=np.array([i for i in range(n) if r==0 or i not in missing])
    bad=np.array(sorted(missing),dtype=int) if r==1 else np.array([],dtype=int)
    for name,shape in [('eig',(n,2000)),('span',(n,80))]:
        a=np.load(root/f'{name}_{r}.npy',mmap_mode='r')
        assert a.shape==shape,(name,a.shape)
        assert not np.isinf(a).any(),(name,r)
        if len(bad):assert np.isnan(a[bad]).all(),(name,r)
        if name=='span':
            nanrows=np.flatnonzero(np.isnan(a).any(1))
            for i in nanrows:
                c=cells[i]
                assert (r==1 and i in missing) or (c['net'],c['step'],c['site_index'],r) in subset_keys
                assert np.isnan(a[i]).all()
        else:assert np.isfinite(a[valid]).all(),(name,r)
    with np.load(root/f'pairs_{r}.npz') as bank:
        for name,k in [('proj',19),('heat',5),('eigd',19)]:
            a=bank[name];assert a.shape==(n,n,k);assert a.dtype==np.float32
            if len(bad):assert np.isnan(a[bad]).all() and np.isnan(a[:,bad]).all()
            block=a[valid[:,None],valid[None,:]]
            assert np.isfinite(block).all(),(name,r)
            assert np.max(np.abs(block-block.transpose(1,0,2)))<1e-6,(name,r)
            assert np.max(np.abs(a[valid,valid]))==0,(name,r)
a=np.load(root/'cka.npy',mmap_mode='r')
assert a.shape==(n,n) and np.isfinite(a).all()
assert np.max(np.abs(a-a.T))<1e-6 and np.max(np.abs(a.diagonal()))==0
for p in root.glob('U_*.npy'):
    a=np.load(p,mmap_mode='r');assert a.ndim==3 and a.shape[1:]==(2000,64)
    assert a.dtype==np.float32 and np.isfinite(a).all()
probe=np.load(root/'probe.npz')
old=np.load(root.parents[1]/'cifar_coherent/data/r20s0b/probe.npz')
assert np.array_equal(probe['idx'],old['idx'])
assert np.array_equal(probe['labels'],old['labels'])
print(f'all fetched-data checks passed: {n} cells, {len(missing)} documented rule-1-unavailable cells, {len(subset)} subset failures, {len(list(root.glob("U_*.npy")))} eigenvector banks')
