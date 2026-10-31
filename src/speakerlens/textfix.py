"""Small text repairs for speech-model output.

- Repetition loops: a phrase repeated many times in a row is cut to one copy.
- Spelled-out acronyms: "a i", "U S T", "p h d" become AI, UST, PHD.
- All-capital English: a turn written in capitals is lowered, keeping common acronyms.
- Wrong script: Japanese kana in a Mandarin/English turn is detected.
"""

from __future__ import annotations

import re

# A unit of 2-80 characters immediately repeated at least 3 more times.
_LOOP = re.compile(r"(.{2,80}?)\1{3,}", re.DOTALL)
# Two or more single ASCII letters separated by single spaces, not touching other letters.
_SPELLED = re.compile(r"(?<![A-Za-z'])[A-Za-z](?: [A-Za-z])+(?![A-Za-z'])")


_WORD = re.compile(r"[A-Za-z][A-Za-z']*")
_KANA = re.compile(r"[\u3040-\u30ff]")
# All-capital words that stay capitals when a shouting-case turn is lowered.
ACRONYMS = {
    "AI", "API", "APP", "CEO", "CFO", "COO", "CPU", "CTO", "GPU", "HR", "ID", "KPI", "MBA",
    "OK", "PDF", "PM", "PPT", "PR", "QA", "TV", "UI", "UK", "USA", "UX", "VPN",
}  # fmt: skip


def has_kana(text: str) -> bool:
    """Japanese hiragana or katakana: a sign the model drifted into Japanese."""
    return bool(_KANA.search(text))


def tame_caps(text: str, min_words: int = 2) -> str:
    """Lower-case the English of a turn written in capitals ("SO WE NEED A PLAN TODAY" ->
    "so we need a plan today"), keeping "I" and common acronyms (HR, OK, AI ...). Only
    when at least `min_words` words of 3+ letters are all capitals, so a lone acronym in
    normal text ("the GPU is busy") is left alone."""
    words = _WORD.findall(text)
    if sum(1 for w in words if len(w) >= 3 and w.isupper()) < min_words:
        return text

    def fix(m: re.Match[str]) -> str:
        w = m.group(0)
        if not w.isupper() or w in ACRONYMS:
            return w
        if w == "I" or w.startswith("I'"):
            return "I" + w[1:].lower()
        return w.lower()

    return _WORD.sub(fix, text)


def collapse_loops(text: str, min_span: int = 24) -> tuple[str, bool]:
    """Keep one copy of any phrase repeated 4+ times in a row over at least `min_span`
    characters. Spaces are ignored when matching, since loops often drift in spacing
    ("if I was just", "ifI was just"). Returns the text and whether a loop was found."""
    kept = [(i, c) for i, c in enumerate(text) if not c.isspace()]
    squeezed = "".join(c for _, c in kept)
    cuts = []
    for m in _LOOP.finditer(squeezed):
        if m.end() - m.start() < min_span:
            continue
        unit = len(m.group(1))
        cuts.append((kept[m.start() + unit][0], kept[m.end() - 1][0] + 1))
    if not cuts:
        return text, False
    for start, end in reversed(cuts):
        text = text[:start] + text[end:]
    return re.sub(r"\s+", " ", text).strip(), True


def join_spelled_letters(text: str) -> str:
    """'a i' -> 'AI', 'U S T' -> 'UST'. Left alone: mixed-case runs such as 'I a', and
    repeats of a one-letter word such as a stuttered 'I I' or 'a a', since those are real
    words."""

    def fix(m: re.Match[str]) -> str:
        letters = m.group(0).replace(" ", "")
        if not (letters.isupper() or letters.islower()):
            return m.group(0)
        if set(letters.lower()) in ({"i"}, {"a"}):
            return m.group(0)
        return letters.upper()

    return _SPELLED.sub(fix, text)
