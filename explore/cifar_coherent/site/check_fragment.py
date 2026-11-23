"""Validate one experiment fragment without building the site (stdlib only; writes nothing).

    python3 site/check_fragment.py <id>            # e.g. markov; run from explore/cifar_coherent/ (or anywhere)
    python3 site/check_fragment.py <id> --strict   # warnings also fail (exit status 1)
    python3 site/check_fragment.py --appendix NAME [--strict]   # an appendix: choices, baselines, method-details,
                                                                # measurements or reproduce (content/<name>.html)

Checks site/content/experiments/<id>.html against the template (_template.html) and the plan's rules:
  errors (exit 1):
    - the fragment is one <section class="experiment" id="x-<id>"> with one <h2> whose text is the manifest title
      (content/experiments/index.json), and the template's <h3> headings in order; every id starts with "x-<id>"
    - every {{fact:NAME}} resolves against build_inputs/facts_base.json (a snapshot of a full build) plus
      assets/x/<id>.facts.json (as x.<id>.<name>); a fact of another experiment (x.<other>.*) is an error
    - every data-src / data-ceiling-src file and JSON path exists (assets/... under site/, d/... under the last
      build viewer/site/ if present); every data-live kind is known; every <img src> exists under site/
    - every <figure> has a <figcaption> that starts with <b>Plotted:</b>
    - no links to #findings or #alignment (retired tabs); no <script>, no inline style colours
  warnings (for the writer to clear or justify in the handback):
    - digits in prose outside {{fact:...}}, <code>, maths ($...$, \\(...\\)), tag attributes, and site names
      (stem, b1..b9, pool, logits)
    - set names built from class names ("the vehicle set", "animal group", ...)
    - qualitative-reading phrases from the retired Findings style ("How I read it", "reading", "speculation")
  both modes also fail on an internal link #<tab>/<id> whose <id> is not in that tab's content fragment(s).

--appendix NAME applies the same fact, data-src, live-kind, image, caption, link and prose rules to
content/<name>.html, with these differences: facts resolve against facts_base.json plus every experiment's
assets/x/*.facts.json (appendices may quote any x.<id>.* fact); ids are free-form but must be unique on the page
(checked against every other content fragment); no template headings. For measurements (Appendix D), every
assets/x/<id>.json "app4" panel of every experiment in the manifest must be shown.

Facts: build_inputs/facts_base.json (the non-x facts of a full build; gitignored: a local, frozen snapshot so that
agents can validate while a build rewrites viewer/site/); if it is missing, the last build's viewer/site/d/facts.json
is used. Delete the snapshot after a build to validate against the new facts.
"""
from __future__ import annotations

import json
import re
import sys
from html.parser import HTMLParser
from pathlib import Path

SITE = Path(__file__).resolve().parent
ROOT = SITE.parent
OUT = ROOT / "viewer" / "site"
XDIR = SITE / "content" / "experiments"
TEMPLATE_H3 = ["Why it is interesting", "How I measure it", "Controls and nulls", "Answer", "Further evidence",
               "Caveats", "Open checks", "In the explorer"]
LIVE_KINDS = {"decod", "curves", "lines", "heatmap", "kernel", "cells", "images"}   # + bn-*, al-* (renderer modules)
SITE_NAMES = r"\b(?:stem|b[1-9]|pool|logits)\b"
CLASS_WORDS = r"(?:airplane|automobile|car|bird|cat|deer|dog|frog|horse|ship|truck|vehicle|animal|plane)s?"
SET_NOUNS = r"(?:set|sets|group|groups|cluster|clusters|block|blocks|mode|modes)"


class Parse(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.stack, self.tags, self.text_runs = [], [], []
        self.h2, self.h3 = [], []
        self.figures, self.cur_fig = [], None
        self.cur_head, self.cur_cap = None, None
        self.skip = 0          # inside <code>, <pre>, <script>, <style>

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        self.tags.append((tag, a, self.getpos()))
        if tag in ("code", "pre", "script", "style", "kbd"):
            self.skip += 1
        if tag in ("h2", "h3"):
            self.cur_head = [tag, a.get("id"), ""]
        if tag == "figure":
            self.cur_fig = {"attrs": a, "caption": None, "pos": self.getpos()}
        if tag == "figcaption" and self.cur_fig is not None:
            self.cur_cap = ""
        if tag not in ("br", "img", "hr", "input", "source", "meta", "link", "wbr"):
            self.stack.append(tag)

    def handle_startendtag(self, tag, attrs):
        self.tags.append((tag, dict(attrs), self.getpos()))

    def handle_endtag(self, tag):
        if tag in ("code", "pre", "script", "style", "kbd"):
            self.skip = max(0, self.skip - 1)
        if tag in ("h2", "h3") and self.cur_head:
            (self.h2 if tag == "h2" else self.h3).append((self.cur_head[1], self.cur_head[2].strip()))
            self.cur_head = None
        if tag == "figcaption" and self.cur_fig is not None:
            self.cur_fig["caption"] = self.cur_cap
            self.cur_cap = None
        if tag == "figure" and self.cur_fig is not None:
            self.figures.append(self.cur_fig)
            self.cur_fig = None
        if self.stack and self.stack[-1] == tag:
            self.stack.pop()

    def handle_data(self, data):
        if self.cur_head is not None:
            self.cur_head[2] += data
        if self.cur_cap is not None:
            self.cur_cap += data
        if not self.skip:
            self.text_runs.append((data, self.getpos()[0]))


def get_path(obj, path):
    for k in path.split(".") if path else []:
        if isinstance(obj, dict):
            obj = obj.get(k)
        elif isinstance(obj, list) and k.isdigit() and int(k) < len(obj):
            obj = obj[int(k)]
        else:
            return None
        if obj is None:
            return None
    return obj


APPENDICES = {"choices": "choices", "baselines": "baselines", "method-details": "method_details",
              "method_details": "method_details", "measurements": "measurements", "reproduce": "reproduce"}
TAB_FILES = {"overview": ["overview.html"], "primer": ["primer.html"], "method": ["method.html"],
             "choices": ["choices.html"], "baselines": ["baselines.html"], "method-details": ["method_details.html"],
             "measurements": ["measurements.html"], "reproduce": ["reproduce.html"]}
TABS = {"overview", "primer", "explorer", "experiments", "method", "appendices", "choices", "baselines",
        "method-details", "measurements", "reproduce"}


def load_base_facts() -> tuple[dict, str | None]:
    """Non-x facts: build_inputs/facts_base.json, else the last build's d/facts.json."""
    for fp in (SITE / "build_inputs" / "facts_base.json", OUT / "d" / "facts.json"):
        if fp.exists():
            return {k: v for k, v in json.loads(fp.read_text()).items() if not k.startswith("x.")}, None
    return {}, "no facts snapshot: neither build_inputs/facts_base.json nor viewer/site/d/facts.json exists (build first)"


def tab_ids(tab: str) -> set[str]:
    """Every id in a tab's content fragment(s) (experiments: the manifest's fragments)."""
    if tab == "experiments":
        files = [XDIR / f"{e['id']}.html" for e in json.loads((XDIR / "index.json").read_text())]
        files.append(XDIR / "_lede.html")
    else:
        files = [SITE / "content" / f for f in TAB_FILES.get(tab, [])]
    ids = set()
    for f in files:
        if f.exists():
            ids.update(re.findall(r'\bid="([^"]+)"', f.read_text()))
    return ids


def check_links(tags, errs, warns):
    """#<tab>/<id> links must land on an id in that tab; retired tabs are errors."""
    cache = {}
    for t, a, (ln, _) in tags:
        href = a.get("href") or ""
        if re.match(r"#(findings|alignment)\b", href):
            errs.append(f"line {ln}: link to a retired tab: {href}")
            continue
        if t != "a" or not href.startswith("#") or href.startswith("#explorer"):
            continue
        tab, _, rest = href[1:].partition("/")
        tab = tab.split("?")[0]
        if tab not in TABS:
            warns.append(f"line {ln}: link {href}: '{tab}' is not a tab (bare #id links work but are fragile)")
            continue
        anchor = rest.split("?")[0]
        if anchor:
            if tab not in cache:
                cache[tab] = tab_ids(tab)
            if anchor not in cache[tab]:
                errs.append(f"line {ln}: link {href}: no id '{anchor}' in the {tab} tab")


def prose_warnings(p, warns, extra_style=()):
    for data, ln in p.text_runs:
        s = re.sub(r"\$\$.*?\$\$|\$[^$]*\$|\\\(.*?\\\)|\\\[.*?\\\]", " ", data, flags=re.S)
        s = s.replace("FACT", " ")
        s = re.sub(SITE_NAMES, " ", s)
        s = re.sub(r"\b(?:ResNet|CIFAR)-[0-9]+\b|\bseed [0-9]\b", " ", s)     # model, dataset and seed names
        s = re.sub(r"\b[A-Z]{1,6}[0-9]+\b", " ", s)          # identifiers such as X7, App4-style labels
        for m in re.finditer(r"[0-9][0-9.,]*", s):
            ctx = s[max(0, m.start() - 25): m.end() + 25].replace("\n", " ").strip()
            warns.append(f"line {ln}: digits in prose '{m.group()}' (use a fact, maths or <code>): ...{ctx}...")
    text = "\n".join(d for d, _ in p.text_runs)
    for m in re.finditer(rf"\b{CLASS_WORDS}[- ]{SET_NOUNS}\b|\b{SET_NOUNS} of {CLASS_WORDS}\b", text, flags=re.I):
        warns.append(f"set named from class make-up? '{m.group()}' (classes only validate; name sets by number)")
    for m in re.finditer(r"How I read it|\bspeculation\b|\(reading\)|\(observed\)", text):
        warns.append(f"retired Findings style: '{m.group()}' (state measured answers only)")


def check_figures(p, errs, warns, xid=None):
    """Captions, live kinds, data-src targets, images (shared by both modes)."""
    xjson = {}
    for f in p.figures:
        ln = f["pos"][0]
        cap = (f["caption"] or "").strip()
        if f["caption"] is None:
            errs.append(f"line {ln}: <figure> without <figcaption>")
        elif not cap.startswith("Plotted:"):
            errs.append(f"line {ln}: figcaption must start with <b>Plotted:</b> (found: {cap[:40]!r})")
    srcs = set()
    for t, a, (ln, _) in p.tags:
        if t == "script":
            errs.append(f"line {ln}: no <script> in fragments")
        if "style" in a and re.search(r"#[0-9a-fA-F]{3,6}\b|rgb\(", a["style"]):
            errs.append(f"line {ln}: hard-coded colour in style=; use palette tokens (var(--c-set-0), ...)")
        if "data-live" in a:
            k = a["data-live"]
            if k not in LIVE_KINDS and not k.startswith(("bn-", "al-")):
                errs.append(f"line {ln}: unknown data-live kind '{k}'")
            if k in ("lines", "heatmap") and "data-src" not in a:
                errs.append(f"line {ln}: data-live=\"{k}\" needs data-src")
        for attr in ("data-src", "data-ceiling-src"):
            if attr not in a:
                continue
            srcs.add(a[attr])
            url, _, path = a[attr].partition("#")
            if url.startswith("assets/"):
                fp = SITE / url
            elif url.startswith("d/"):
                fp = OUT / url
                if not fp.exists():
                    warns.append(f"line {ln}: {attr}={a[attr]}: {url} not in the last build (viewer/site/); "
                                 "integration's --release build will check it")
                    continue
            else:
                errs.append(f"line {ln}: {attr}={a[attr]}: must start with assets/ or d/")
                continue
            if not fp.exists():
                errs.append(f"line {ln}: {attr}: {fp} does not exist (run your generator)")
                continue
            if fp not in xjson:
                xjson[fp] = json.loads(fp.read_text())
            o = get_path(xjson[fp], path)
            if o is None:
                errs.append(f"line {ln}: {attr}: JSON path '{path}' not found in {url}")
            elif a.get("data-live") == "lines" and not (isinstance(o, dict) and "x" in o and "series" in o):
                errs.append(f"line {ln}: {a[attr]} is not a lines object (needs x, series)")
            elif a.get("data-live") == "heatmap" and not (isinstance(o, dict) and "z" in o and "rows" in o):
                errs.append(f"line {ln}: {a[attr]} is not a heatmap object (needs rows, cols, z)")
        if t == "img":
            src = a.get("src", "")
            if xid and src == f"assets/fig/x_{xid}_explorer.png" and not (SITE / src).exists():
                warns.append(f"line {ln}: {src} not captured yet: after a build, run "
                             f"python3 site/check_browser.py --capture-explorer {xid} (integration does this)")
            elif not src.startswith("data:") and not (SITE / src).exists():
                errs.append(f"line {ln}: <img src={src}> not found under site/")
            if not a.get("alt"):
                errs.append(f"line {ln}: <img> without alt text")
    return srcs


def main_appendix(name: str, strict: bool):
    if name not in APPENDICES:
        sys.exit(f"--appendix: '{name}' is not one of {', '.join(sorted(set(APPENDICES) - {'method_details'}))}")
    stem = APPENDICES[name]
    frag = SITE / "content" / f"{stem}.html"
    if not frag.exists():
        sys.exit(f"error: {frag} not found")
    raw = frag.read_text()
    errs, warns = [], []
    base, err = load_base_facts()
    if err:
        errs.append(err)
    facts = dict(base)
    for fp in (SITE / "assets" / "x").glob("*.facts.json"):
        xid = fp.name[: -len(".facts.json")]
        for k, v in json.loads(fp.read_text()).items():
            facts[f"x.{xid}.{k}"] = v
    used = [u.strip() for u in re.findall(r"\{\{\s*fact:([^}]+)\}\}", raw)]
    for u in used:
        if u not in facts:
            near = [k for k in facts if k.split(".")[-1] == u.split(".")[-1]][:5]
            errs.append(f"unknown fact '{u}'" + (f" (same last part: {', '.join(near)})" if near else ""))
    p = Parse()
    p.feed(re.sub(r"\{\{\s*fact:[^}]+\}\}", "FACT", raw))
    ids = [a.get("id") for t, a, _ in p.tags if a.get("id")]
    dup = sorted({i for i in ids if ids.count(i) > 1})
    if dup:
        errs.append(f"duplicate ids: {', '.join(dup)}")
    others = {}
    shell = [SITE / "index.html", SITE / "explorer.html"]
    for f in [*(SITE / "content").rglob("*.html"), *shell]:
        if f == frag or f.name == "_template.html" or (f.parent == SITE / "content" and f.name.startswith("primer_")):
            continue
        for i in re.findall(r'\bid="([^"{}]+)"', f.read_text()):
            others.setdefault(i, str(f.relative_to(SITE)))
    for i in sorted(set(ids)):
        if i in others:
            errs.append(f"id '{i}' is also in {others[i]} (ids are page-global)")
    check_links(p.tags, errs, warns)
    srcs = check_figures(p, errs, warns)
    if stem == "measurements":     # every panel the editorial cut moved out of an experiment is shown
        for e in json.loads((XDIR / "index.json").read_text()):
            xp = SITE / "assets" / "x" / f"{e['id']}.json"
            if not xp.exists():
                continue
            for k in json.loads(xp.read_text()).get("app4", {}):
                if f"assets/x/{e['id']}.json#app4.{k}" not in srcs:
                    errs.append(f"Appendix D panel not shown: assets/x/{e['id']}.json#app4.{k}")
    prose_warnings(p, warns)
    for e in errs:
        print("error:", e)
    for w in warns:
        print("warning:", w)
    print(f"{frag.relative_to(ROOT)}: {len(errs)} error(s), {len(warns)} warning(s); "
          f"{len(used)} fact uses, {len(p.figures)} figures, {len(ids)} ids")
    if errs or (strict and warns):
        sys.exit(1)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    strict = "--strict" in sys.argv
    if len(args) != 1:
        sys.exit(__doc__)
    if "--appendix" in sys.argv:
        return main_appendix(args[0], strict)
    xid = args[0]
    errs, warns = [], []
    frag = XDIR / f"{xid}.html"
    if not frag.exists():
        sys.exit(f"error: {frag} not found (copy content/experiments/_template.html)")
    raw = frag.read_text()

    # ---- manifest title
    titles = {e["id"]: e["title"] for e in json.loads((XDIR / "index.json").read_text())}
    if xid not in titles:
        errs.append(f"'{xid}' is not in content/experiments/index.json")

    # ---- facts
    base, err = load_base_facts()
    if err:
        errs.append(err)
    own_p = SITE / "assets" / "x" / f"{xid}.facts.json"
    own = {}
    if own_p.exists():
        for k, v in json.loads(own_p.read_text()).items():
            if isinstance(v, bool) or not isinstance(v, (str, int, float)):
                errs.append(f"{own_p.name}: fact '{k}' is not a string or number")
            own[f"x.{xid}.{k}"] = v
    facts = {**base, **own}
    used = re.findall(r"\{\{\s*fact:([^}]+)\}\}", raw)
    for name in used:
        name = name.strip()
        if name.startswith("x.") and not name.startswith(f"x.{xid}."):
            errs.append(f"fact '{name}' belongs to another experiment; compute what you need in your own generator")
        elif name not in facts:
            near = [k for k in facts if k.split(".")[-1] == name.split(".")[-1]][:5]
            errs.append(f"unknown fact '{name}'" + (f" (same last part: {', '.join(near)})" if near else ""))
    unused = sorted(set(own) - {u.strip() for u in used})
    if unused:
        warns.append(f"{len(unused)} fact(s) generated but not quoted (fine if deliberate): " + ", ".join(unused[:12])
                     + (" ..." if len(unused) > 12 else ""))

    # ---- structure
    p = Parse()
    p.feed(re.sub(r"\{\{\s*fact:[^}]+\}\}", "FACT", raw))
    secs = [(t, a) for t, a, _ in p.tags if t == "section"]
    if not secs or secs[0][1].get("class") != "experiment" or secs[0][1].get("id") != f"x-{xid}":
        errs.append(f'the fragment must start with <section class="experiment" id="x-{xid}">')
    if len(secs) != 1:
        errs.append(f"expected exactly one <section>, found {len(secs)}")
    if not raw.strip().startswith(f'<section class="experiment" id="x-{xid}">') and not raw.lstrip().startswith("<!--"):
        errs.append("nothing may precede the <section> except one HTML comment")
    if len(p.h2) != 1:
        errs.append(f"expected exactly one <h2> (the question), found {len(p.h2)}")
    elif xid in titles and p.h2[0][1] != titles[xid]:
        errs.append(f"<h2> text must equal the manifest title:\n      h2:       {p.h2[0][1]}\n      manifest: {titles[xid]}")
    got = [t for _, t in p.h3]
    pos = [got.index(h) if h in got else -1 for h in TEMPLATE_H3]
    for h, i in zip(TEMPLATE_H3, pos):
        if i < 0:
            errs.append(f"missing <h3>{h}</h3> (template heading)")
    present = [i for i in pos if i >= 0]
    if present != sorted(present):
        errs.append("template <h3> headings are out of order: " + " / ".join(TEMPLATE_H3))
    for t, a, (ln, _) in p.tags:
        i = a.get("id")
        if i and not (i == f"x-{xid}" or i.startswith(f"x-{xid}-")):
            errs.append(f"line {ln}: id '{i}' must start with 'x-{xid}-' (ids are page-global)")
        if t in ("h2", "h3") and t == "h3" and not i:
            errs.append(f"line {ln}: <h3> without an id")

    check_links(p.tags, errs, warns)
    check_figures(p, errs, warns, xid)

    # ---- prose warnings
    prose_warnings(p, warns)

    # ---- report
    for e in errs:
        print("error:", e)
    for w in warns:
        print("warning:", w)
    print(f"{frag.relative_to(ROOT)}: {len(errs)} error(s), {len(warns)} warning(s); "
          f"{len(used)} fact uses, {len(p.figures)} figures")
    if errs or (strict and warns):
        sys.exit(1)


if __name__ == "__main__":
    main()
