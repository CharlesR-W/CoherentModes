"""Build the WordNet superclass map for ImageNet-100 (display/validation only; never used by the method).

Rule (fixed before looking at any model output): every class follows its first WordNet hypernym path
(nltk `hypernym_paths()[0]`). Start with one group at the root; repeatedly split the largest group
into the next node along its members' paths, until no group has more than 15 classes. Groups with
fewer than 3 classes are pooled into "other <node they were split from>".

    uv run --with nltk python -I explore/vit_coherent/superclasses.py <classes.py> <nltk_data_dir> <out.json>
"""
import json
import re
import sys
from collections import defaultdict

src, nltk_dir, out = sys.argv[1:4]
import nltk  # noqa: E402

nltk.data.path.insert(0, nltk_dir)
from nltk.corpus import wordnet as wn  # noqa: E402

pairs = re.findall(r'"(n\d{8})": "([^"]+)"', open(src).read())
wnids = sorted(w for w, _ in pairs)                # class id = rank of wnid (sorted)
names = dict(pairs)
paths = {w: [s.name() for s in wn.synset_from_pos_and_offset("n", int(w[1:])).hypernym_paths()[0]] for w in wnids}
# group = (depth, node, parent node); members share path[:depth+1]
groups = {(0, paths[wnids[0]][0], None): list(wnids)}
MAX, MIN = 15, 3
while True:
    key = max(groups, key=lambda k: (len(groups[k]), k[1]))
    if len(groups[key]) <= MAX:
        break
    d, node, _ = key
    mem = groups.pop(key)
    kids = defaultdict(list)
    for w in mem:
        p = paths[w]
        kids[p[min(d + 1, len(p) - 1)]].append(w)
    for k, v in kids.items():
        groups[(d + 1, k, node)] = v
final = defaultdict(list)
for (d, node, parent), mem in groups.items():
    final[node if len(mem) >= MIN or parent is None else f"other {parent}"].extend(mem)
sup = sorted(final)
cls = [{"id": i, "wnid": w, "name": names[w], "superclass": next(j for j, s in enumerate(sup) if w in final[s])}
       for i, w in enumerate(wnids)]
json.dump({"rule": __doc__.split("\n\n")[1].replace("\n", " "),
           "superclasses": [{"id": j, "name": s, "n_classes": len(final[s])} for j, s in enumerate(sup)],
           "classes": cls}, open(out, "w"), indent=1)
for j, s in enumerate(sup):
    print(j, s, len(final[s]), [names[w].split(",")[0] for w in final[s]][:6])
