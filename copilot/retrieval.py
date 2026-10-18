"""Stdlib BM25 over document sections, with product-area and latest-version filters."""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, asdict
from functools import lru_cache
from pathlib import Path

from .config import Config, DEFAULT_CORPUS_DIR

STOPWORDS = frozenset(
    "a an and are as at be but by can do does for from has have how i if in is it its me my of on or "
    "so that the their there this to was what when where which who will with you your we our not no "
    "give please want many long much tell need get could would should know about".split()
)


def stem(tok: str) -> str:
    if len(tok) > 4 and tok.endswith("ies"):
        return tok[:-3] + "y"
    if len(tok) > 3 and tok.endswith("s") and not tok.endswith("ss"):
        return tok[:-1]
    return tok


def tokenize(text: str) -> list[str]:
    return [stem(t) for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOPWORDS]


@dataclass(frozen=True)
class Section:
    doc_id: str
    version: int
    area: str
    title: str
    heading: str
    text: str


@dataclass(frozen=True)
class Passage:
    doc_id: str
    version: int
    section: str
    area: str
    title: str
    text: str
    score: float

    def to_dict(self) -> dict:
        d = asdict(self)
        d["score"] = round(self.score, 4)
        return d


@dataclass(frozen=True)
class Doc:
    doc_id: str
    version: int
    area: str
    title: str
    sections: tuple


def parse_doc(raw: str) -> Doc:
    m = re.match(r"---\n(.*?)\n---\n(.*)", raw, re.S)
    if not m:
        raise ValueError("document is missing front matter")
    meta = dict(line.split(": ", 1) for line in m.group(1).splitlines() if ": " in line)
    secs, heading, buf = [], None, []
    for line in m.group(2).splitlines():
        if line.startswith("## "):
            if heading:
                secs.append((heading, " ".join(buf).strip()))
            heading, buf = line[3:].strip(), []
        elif line.strip():
            buf.append(line.strip())
    if heading:
        secs.append((heading, " ".join(buf).strip()))
    return Doc(meta["id"], int(meta["version"]), meta["area"], meta["title"], tuple(secs))


@lru_cache(maxsize=8)
def load_corpus(corpus_dir: str = str(DEFAULT_CORPUS_DIR)) -> tuple:
    docs = [parse_doc(p.read_text(encoding="utf-8")) for p in sorted(Path(corpus_dir).glob("*.md"))]
    return tuple(sorted(docs, key=lambda d: (d.doc_id, d.version)))


class Retriever:
    def __init__(self, docs=None, k1: float = 1.5, b: float = 0.75, latest_only: bool = True):
        self.docs = tuple(docs if docs is not None else load_corpus())
        self.k1, self.b, self.latest_only = k1, b, latest_only
        self.sections: list[Section] = []
        self.latest: dict[str, int] = {}
        for d in self.docs:
            self.latest[d.doc_id] = max(self.latest.get(d.doc_id, 0), d.version)
            for heading, text in d.sections:
                self.sections.append(Section(d.doc_id, d.version, d.area, d.title, heading, text))
        self._tf = [Counter(tokenize(f"{s.title} {s.heading} {s.heading} {s.text}")) for s in self.sections]
        self._len = [sum(c.values()) for c in self._tf]
        self._avg = sum(self._len) / max(1, len(self._len))
        df: Counter = Counter()
        for c in self._tf:
            df.update(c.keys())
        n = len(self.sections)
        self._idf = {t: math.log(1 + (n - v + 0.5) / (v + 0.5)) for t, v in df.items()}

    @classmethod
    def from_config(cls, cfg: Config) -> "Retriever":
        return cls(load_corpus(cfg.corpus_dir), cfg.bm25_k1, cfg.bm25_b, cfg.latest_only)

    def doc_ids(self) -> set:
        return set(self.latest)

    def has_doc(self, doc_id: str) -> bool:
        return doc_id in self.latest

    def doc_text(self, doc_id: str) -> str:
        """Full text of the latest version of a document (used to verify citations)."""
        v = self.latest.get(doc_id)
        return " ".join(f"{s.heading} {s.text}" for s in self.sections if s.doc_id == doc_id and s.version == v)

    def search(self, query: str, k: int = 4, area: str | None = None, latest_only: bool | None = None) -> list[Passage]:
        lo = self.latest_only if latest_only is None else latest_only
        q = tokenize(query)
        out = []
        for i, s in enumerate(self.sections):
            if area and s.area != area:
                continue
            if lo and s.version != self.latest[s.doc_id]:
                continue
            tf, dl, score = self._tf[i], self._len[i], 0.0
            for t in set(q):
                f = tf.get(t, 0)
                if f:
                    score += self._idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self._avg))
            if score > 0:
                out.append(Passage(s.doc_id, s.version, s.heading, s.area, s.title, s.text, score))
        out.sort(key=lambda p: (-p.score, p.doc_id, -p.version, p.section))
        return out[:k]
