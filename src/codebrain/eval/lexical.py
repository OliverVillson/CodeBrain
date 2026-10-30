"""Identifier splitting and a small in-memory BM25 (the sketch §4 lexical view)."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from collections.abc import Sequence

_WORD = re.compile(r"[A-Za-z][A-Za-z0-9]*")
_CAMEL = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|[0-9]+")
_STOP = """a an and are as at be by def else for from if in is it none not of on or return
self cls the this to true false with class import pass raise try except"""
STOPWORDS = frozenset(_STOP.split())


def split_identifiers(text: str) -> list[str]:
    """``getUserById`` and ``get_user_by_id`` both become get, user, by, id (plus the whole)."""
    out: list[str] = []
    for word in _WORD.findall(text):
        low = word.lower()
        parts = [p.lower() for p in _CAMEL.findall(word)]
        if low not in STOPWORDS:
            out.append(low)
        if len(parts) > 1:
            out.extend(p for p in parts if p not in STOPWORDS and len(p) > 1)
    return out


class BM25:
    def __init__(self, docs: Sequence[Sequence[str]], k1: float = 1.2, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.n = len(docs)
        self.lengths = [len(d) for d in docs]
        self.avgdl = (sum(self.lengths) / self.n) if self.n else 0.0
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for i, d in enumerate(docs):
            for term, tf in Counter(d).items():
                self.postings[term].append((i, tf))

    def scores(self, query: Sequence[str]) -> dict[int, float]:
        out: dict[int, float] = defaultdict(float)
        for term in set(query):
            plist = self.postings.get(term)
            if not plist:
                continue
            idf = math.log(1 + (self.n - len(plist) + 0.5) / (len(plist) + 0.5))
            for i, tf in plist:
                norm = self.k1 * (1 - self.b + self.b * self.lengths[i] / (self.avgdl or 1.0))
                out[i] += idf * tf * (self.k1 + 1) / (tf + norm)
        return out

    def top(self, query: Sequence[str], k: int) -> list[int]:
        s = self.scores(query)
        return sorted(s, key=lambda i: -s[i])[:k]
