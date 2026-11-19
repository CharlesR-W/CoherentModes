"""Coarse token-context classes for a decoded token string (validation labels only).

`token_class(s)` maps one decoded token string (as stored in `probe.json` `token_str`) to one of `CLASSES`.
The rules are applied in order and the first match wins, with `core = s.strip()`. Nothing in any construction may
use these classes: they exist to score sets after the fact, as source labels do.

Pure, dependency-free and deterministic (standard library only). "Punctuation" here means a character that is
neither alphanumeric nor whitespace (`not ch.isalnum() and not ch.isspace()`), so quotes, dashes, symbols and the
replacement character U+FFFD all count.
"""
from __future__ import annotations

CLASSES = ['newline', 'indent', 'number', 'latex', 'open', 'close', 'sentence_punct', 'symbol',
           'function_word', 'word_start', 'word_cont', 'other']

# Common English function words: articles, determiners, prepositions, pronouns, auxiliaries and modals,
# conjunctions, and a few common adverbial particles. Fixed list; compared against core.lower().
FUNCTION_WORDS = frozenset('''
a an the
this that these those some any no every each either neither all both few many much more most other another such
what which who whom whose
i me my myself you your yours yourself he him his himself she her herself it its itself
we us our ours they them their one
of in on at to for from by with without about above below under over into upon out off up down
through between among against during before after since until toward towards across along around behind
near within than like as
and or but nor so yet if then else because although though while unless whether
when where why how there here
be am is are was were been being have has had having do does did
can could may might must shall should will would
not no also only just very too
'''.split())

_OPEN = set('([{<')
_CLOSE = set(')]}>')
_SENT = set('.,;:!?')


def _punct(ch: str) -> bool:
    return not ch.isalnum() and not ch.isspace()


def _alphabetic(core: str) -> bool:
    """Letters only, allowing internal apostrophes and hyphens (not leading or trailing)."""
    if not core or not core[0].isalpha() or not core[-1].isalpha():
        return False
    return core.replace("'", '').replace('-', '').isalpha()


def token_class(s: str) -> str:
    """The class of one decoded token string; first matching rule wins (see `CLASSES` for the order)."""
    core = s.strip()
    if core == '':
        return 'newline' if '\n' in s else 'indent'
    if any(ch.isdigit() for ch in core) and all(ch.isdigit() or ch in '.,' for ch in core):
        return 'number'
    if '\\' in s or '$' in s:
        return 'latex'
    if all(_punct(ch) for ch in core):
        if core[-1] in _OPEN:
            return 'open'
        if core[0] in _CLOSE:
            return 'close'
        if all(ch in _SENT for ch in core):
            return 'sentence_punct'
        return 'symbol'
    if core.lower() in FUNCTION_WORDS:
        return 'function_word'
    if _alphabetic(core):
        return 'word_start' if s[0].isspace() else 'word_cont'
    return 'other'


if __name__ == '__main__':
    cases = {
        '\n': 'newline', '\n\n': 'newline', ' \n': 'newline', '    ': 'indent', '\t': 'indent', '\xa0': 'indent',
        '1': 'number', ' 2019': 'number', '3.14': 'number', '1,000': 'number',
        ' $\\': 'latex', '\\': 'latex', '$': 'latex', 'frac': 'word_cont', '_{': 'open', ' (': 'open',
        '([': 'open', ')': 'close', '),': 'close', ');': 'close', ']': 'close', '.': 'sentence_punct',
        ',': 'sentence_punct', '?!': 'sentence_punct', '"': 'symbol', '-': 'symbol', ' =': 'symbol', '//': 'symbol',
        '’': 'symbol', '"—': 'symbol', ' the': 'function_word', 'The': 'function_word', ' of': 'function_word',
        ' I': 'function_word', ' cat': 'word_start', ' well-known': 'word_start', " don't": 'word_start',
        'ing': 'word_cont', "'s": 'other', 'x2': 'other', ' 2nd': 'other', '-ish': 'other',
    }
    bad = {s: (token_class(s), want) for s, want in cases.items() if token_class(s) != want}
    assert not bad, bad
    assert set(token_class(s) for s in cases) <= set(CLASSES)
    assert 100 <= len(FUNCTION_WORDS) <= 150, len(FUNCTION_WORDS)
    print(f'token_classes: {len(cases)} cases ok, {len(FUNCTION_WORDS)} function words, {len(CLASSES)} classes')
