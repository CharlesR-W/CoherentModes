"""Probe A and the order-control probe S (SPEC section 3). Runs as unit U0 (CPU) on Modal.

Labels (sources) are used here only to assemble a balanced, varied probe with per-source filters. De-duplication
is label-free and greedy in draw order across all sources. The draw order interleaves the sources round-robin
(maths, code, wiki, forum, poetry, maths, ...), each source taking its next shuffled document, until each has
PER_SOURCE accepted snippets.

`build_probe` is pure (documents and a tokenizer in, probe dict out) so it is tested on CPU with a toy tokenizer.
`main` reads the real corpora on Modal: the Pile test split (`monology/pile-uncopyrighted`, test.jsonl.zst) and the
Gutenberg Poetry Corpus (`biglam/gutenberg-poetry-corpus`, one parquet of {line, gutenberg_id}).
"""
from __future__ import annotations

import json
from collections import Counter

import numpy as np

import lm_construction as LC

PILE_SETS = {'maths': 'ArXiv', 'code': 'Github', 'wiki': 'Wikipedia (en)', 'forum': 'StackExchange'}
MAX_ATTEMPTS = 5
WINDOW_CHARS = 8000
FILTERS = dict(
    all=dict(min_ascii_fraction=0.95, max_whitespace_token_fraction=0.30, forbid=['<|endoftext|>', '\x00'],
             forbid_token_ids=[LC.EOT]),
    maths=dict(min_dollars=4, any_of=['\\begin{equation', '\\begin{align', '\\[', '\\frac']),
    code=dict(max_digit_fraction=0.30, max_line_chars=300),
    dedup=dict(distinct_multiset_at=LC.POSITIONS, max_8gram_jaccard=0.2),
)


# ------------------------------------------------------------------------------------- candidate starts
def pile_starts(text):
    """Offset 0, and every offset right after '\\n\\n' whose next character is not whitespace."""
    starts, at = [0], text.find('\n\n')
    while at >= 0:
        k = at + 2
        if k < len(text) and not text[k].isspace():
            starts.append(k)
        at = text.find('\n\n', at + 1)
    return sorted(set(starts))


def line_starts(text):
    """Poetry: every line start of the joined document."""
    return [0] + [i + 1 for i, ch in enumerate(text) if ch == '\n' and i + 1 < len(text)]


def snippet(tokenizer, text, start, seq=LC.SEQ):
    """First `seq` tokens of text[start:], or None if fewer remain. Tokenises a window and widens it to the
    whole remainder whenever the window could have cut the seq-th token."""
    window = WINDOW_CHARS
    while True:
        part = text[start:start + window]
        enc = tokenizer(part, add_special_tokens=False, return_offsets_mapping=True)
        ids, offs = enc['input_ids'], enc['offset_mapping']
        full = start + window >= len(text)
        if len(ids) > seq and offs[seq][1] < len(part) - 64 or full:
            break
        window *= 4
    if len(ids) < seq:
        return None
    end = offs[seq - 1][1]
    return ids[:seq], part[:end]


# ------------------------------------------------------------------------------------------------ filters
def reject_reason(source, ids, text, tokenizer):
    tok_str = [tokenizer.decode([t]) for t in ids]
    if sum(ord(ch) < 128 for ch in text) < FILTERS['all']['min_ascii_fraction'] * len(text):
        return 'ascii'
    if sum(s.strip() == '' for s in tok_str) > FILTERS['all']['max_whitespace_token_fraction'] * len(ids):
        return 'whitespace'
    if any(f in text for f in FILTERS['all']['forbid']) or LC.EOT in ids:
        return 'eot_or_nul'
    if source == 'maths':
        f = FILTERS['maths']
        if text.count('$') < f['min_dollars'] and not any(s in text for s in f['any_of']):
            return 'maths_markup'
    if source == 'code':
        f = FILTERS['code']
        if sum(ch.isdigit() for ch in text) > f['max_digit_fraction'] * len(text):
            return 'code_digits'
        if any(len(line) > f['max_line_chars'] for line in text.split('\n')):
            return 'code_long_line'
    return None


class Dedup:
    """(i) distinct multiset of tokens 1..t for every sampled t; (ii) 8-gram Jaccard <= 0.2 with every accepted."""

    def __init__(self):
        self.multisets = {t: set() for t in LC.POSITIONS}
        self.grams = []

    @staticmethod
    def _grams(ids):
        return {tuple(ids[i:i + 8]) for i in range(len(ids) - 7)}

    def reason(self, ids):
        for t in LC.POSITIONS:
            if tuple(sorted(ids[:t])) in self.multisets[t]:
                return f'dup_multiset_t{t}'
        g = self._grams(ids)
        for h in self.grams:
            inter = len(g & h)
            if inter and inter / len(g | h) > FILTERS['dedup']['max_8gram_jaccard']:
                return 'dup_8gram'
        return None

    def add(self, ids):
        for t in LC.POSITIONS:
            self.multisets[t].add(tuple(sorted(ids[:t])))
        self.grams.append(self._grams(ids))


# -------------------------------------------------------------------------------------------------- build
def build_probe(documents, tokenizer, per_source=LC.PER_SOURCE):
    """documents: {source: [(doc_id, text), ...] in file order}. Returns the probe.json dict (SPEC 10.1)."""
    queues = {}
    for s in LC.SOURCES:
        docs = list(documents[s])
        order = np.random.RandomState(0).permutation(len(docs))
        queues[s] = [docs[i] for i in order]
    accepted = {s: [] for s in LC.SOURCES}
    rejections = {s: Counter() for s in LC.SOURCES}
    cursor = {s: 0 for s in LC.SOURCES}
    dedup = Dedup()
    draw = []
    while any(len(accepted[s]) < per_source for s in LC.SOURCES):
        progressed = False
        for s in LC.SOURCES:
            if len(accepted[s]) >= per_source or cursor[s] >= len(queues[s]):
                continue
            progressed = True
            rank = cursor[s]
            cursor[s] += 1
            doc_id, text = queues[s][rank]
            starts = line_starts(text) if s == 'poetry' else pile_starts(text)
            rng = np.random.RandomState(1000 + rank)
            for _ in range(MAX_ATTEMPTS):
                start = int(starts[rng.randint(len(starts))])
                got = snippet(tokenizer, text, start)
                if got is None:
                    rejections[s]['too_short'] += 1
                    continue
                ids, chunk = got
                why = reject_reason(s, ids, chunk, tokenizer) or dedup.reason(ids)
                if why:
                    rejections[s][why] += 1
                    continue
                dedup.add(ids)
                accepted[s].append(dict(source=s, doc=doc_id, doc_rank=rank, char_start=start, tokens=list(ids)))
                draw.append((s, len(accepted[s]) - 1))
                break
        if not progressed:
            short = {s: len(accepted[s]) for s in LC.SOURCES}
            raise RuntimeError(f'ran out of documents: accepted {short}')
    flat = [accepted[s][k] for s, k in draw]
    perm = np.random.RandomState(0).permutation(len(flat))
    items = []
    for new_id, old in enumerate(perm):
        it = dict(flat[old])
        it['id'] = new_id
        it['token_str'] = [tokenizer.decode([t]) for t in it['tokens']]
        it['text'] = tokenizer.decode(it['tokens'])
        items.append(it)
    s_perm = [np.random.RandomState(it['id']).permutation(LC.SEQ - 1).tolist() for it in items]
    return dict(items=items,
                S=dict(rule='tokens 1..127 permuted with RandomState(item_id); token 128 and EOT kept',
                       permutation=s_perm),
                filters=FILTERS, seeds=dict(document_shuffle='RandomState(0) per source',
                                            start='RandomState(1000 + doc_rank)', item_order='RandomState(0)',
                                            S='RandomState(item_id)'),
                draw_order='round-robin over sources ' + ', '.join(LC.SOURCES),
                rejections={s: dict(rejections[s]) for s in LC.SOURCES},
                documents_drawn={s: cursor[s] for s in LC.SOURCES})


def shuffled(tokens, perm):
    t = np.asarray(tokens)
    return np.concatenate([t[:-1][np.asarray(perm)], t[-1:]])


def check_probe(probe):
    """Invariants of SPEC 3: balanced sources, distinct documents, distinct multisets at every sampled t, S keeps
    token 128 and the multiset."""
    items = probe['items']
    assert Counter(it['source'] for it in items) == {s: LC.PER_SOURCE for s in LC.SOURCES}
    assert len({(it['source'], it['doc']) for it in items}) == len(items)
    for t in LC.POSITIONS:
        assert len({tuple(sorted(it['tokens'][:t])) for it in items}) == len(items), t
    for it, perm in zip(items, probe['S']['permutation']):
        s = shuffled(it['tokens'], perm)
        assert s[-1] == it['tokens'][-1] and sorted(s) == sorted(it['tokens'])
    return True


# ------------------------------------------------------------------------------------------------- Modal
def read_pile(path, wanted=PILE_SETS):
    import io
    import zstandard
    names = {v: k for k, v in wanted.items()}
    docs = {k: [] for k in wanted}
    with open(path, 'rb') as fh:
        reader = io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(fh), encoding='utf-8')
        for line_no, line in enumerate(reader):
            row = json.loads(line)
            src = names.get(row['meta']['pile_set_name'])
            if src:
                docs[src].append((f'pile-test:{line_no}', row['text']))
    return docs


def read_poetry(path):
    import pyarrow.parquet as pq
    table = pq.read_table(path, columns=['line', 'gutenberg_id'])
    lines, ids = table.column('line').to_pylist(), table.column('gutenberg_id').to_pylist()
    docs, current, buf = [], None, []
    for line, gid in zip(lines, ids):
        if gid != current and buf:
            docs.append((f'gutenberg:{current}', '\n'.join(buf)))
            buf = []
        current = gid
        buf.append(line)
    if buf:
        docs.append((f'gutenberg:{current}', '\n'.join(buf)))
    seen, out = set(), []
    for doc_id, text in docs:                 # a document is one gutenberg_id: merge non-contiguous runs
        if doc_id in seen:
            k = next(i for i, (d, _) in enumerate(out) if d == doc_id)
            out[k] = (doc_id, out[k][1] + '\n' + text)
        else:
            seen.add(doc_id)
            out.append((doc_id, text))
    return out


def main(out_path, cache_dir):
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(LC.MODEL, revision=f'step{LC.FINAL}', cache_dir=cache_dir)
    pile = hf_hub_download('monology/pile-uncopyrighted', 'test.jsonl.zst', repo_type='dataset', cache_dir=cache_dir)
    poetry = hf_hub_download('biglam/gutenberg-poetry-corpus', 'data/train-00000-of-00001-fa9fb9e1f16eed7e.parquet',
                             repo_type='dataset', cache_dir=cache_dir)
    docs = read_pile(pile)
    docs['poetry'] = read_poetry(poetry)
    probe = build_probe(docs, tok)
    check_probe(probe)
    probe['tokenizer'] = dict(model=LC.MODEL, revision=f'step{LC.FINAL}')
    with open(out_path, 'w') as fh:
        json.dump(probe, fh)
    return dict(n=len(probe['items']), rejections=probe['rejections'], documents_drawn=probe['documents_drawn'])
