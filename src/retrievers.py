"""Retrieval methods for Day 7 over the Day 6 corpus (index B, 92 chunks).

- DenseRetriever   : bge-small query embedding -> pgvector cosine search (Day 6 store), SQL filter.
- BM25Retriever    : lexical BM25 (rank_bm25) over the same chunks, same filter.
- HybridRRF        : reciprocal rank fusion of dense + BM25 candidate lists.
- RerankRetriever  : candidates from another retriever, re-scored by a cross-encoder.
- expand_neighbors : context strategy that adds the previous/next chunk of the same section.

Every result is a chunk dict (id, text, source_id, source, page, metadata) plus `score`,
`rank` and `strategy`. Filters only come from the user-provided `filter` field.
"""
from __future__ import annotations

import os
import re

import psycopg

from common import QueryEmbedder, load_chunks

STOPWORDS = set('a an the of in on to for is are was were be by with as at or and it this that these those '
                'what which how when does do can you your from into its if than then there their'.split())


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r'[a-z0-9_]+', text.lower()) if t not in STOPWORDS]


def _apply_filter(chunks, filters):
    if not filters:
        return chunks
    return [c for c in chunks if all(c['metadata'].get(k) == v for k, v in filters.items())]


class DenseRetriever:
    name = 'dense'

    def __init__(self, chunks=None, embedder=None, database_url=None, index='B', model=None):
        self.by_id = {c['id']: c for c in (chunks or load_chunks())}
        self.embedder = embedder or QueryEmbedder()
        self.url = database_url or os.environ['DATABASE_URL']
        self.index = index
        self.model = model or self.embedder.model_name
        self.conn = psycopg.connect(self.url, connect_timeout=5)

    def search(self, query, *, filters=None, top_k=3):
        vec = '[' + ','.join(f'{x:.7f}' for x in self.embedder.embed(query)) + ']'
        where, params = ['index_name = %s', 'model = %s'], [self.index, self.model]
        for k, v in (filters or {}).items():
            if k not in ('chapter', 'section'):
                raise ValueError(f'filter {k} not supported')
            where.append(f'{k} = %s')
            params.append(v)
        sql = (f"SELECT chunk_id, 1 - (embedding <=> %s::vector) FROM day6_chunks WHERE {' AND '.join(where)} "
               f"ORDER BY embedding <=> %s::vector, chunk_id LIMIT %s")
        rows = self.conn.execute(sql, [vec, *params, vec, top_k]).fetchall()
        if not rows and not filters:
            raise RuntimeError('dense search returned nothing: wrong index/model in day6_chunks?')
        return [{**self.by_id[cid], 'score': float(s), 'rank': i + 1, 'strategy': self.name}
                for i, (cid, s) in enumerate(rows)]


class BM25Retriever:
    name = 'bm25'

    def __init__(self, chunks=None, k1=1.5, b=0.75):
        from rank_bm25 import BM25Okapi
        self.chunks = chunks or load_chunks()
        self.k1, self.b = k1, b
        self._bm25 = BM25Okapi
        self._cache = {}

    def _index(self, key, chunks):
        if key not in self._cache:
            self._cache[key] = self._bm25([tokenize(c['text']) for c in chunks], k1=self.k1, b=self.b)
        return self._cache[key]

    def search(self, query, *, filters=None, top_k=3):
        pool = _apply_filter(self.chunks, filters)
        if not pool:
            return []
        bm25 = self._index(repr(sorted((filters or {}).items())), pool)
        scores = bm25.get_scores(tokenize(query))
        order = sorted(range(len(pool)), key=lambda i: (-scores[i], pool[i]['id']))
        out = [i for i in order if scores[i] > 0][:top_k]
        return [{**pool[i], 'score': float(scores[i]), 'rank': r + 1, 'strategy': self.name}
                for r, i in enumerate(out)]


class HybridRRF:
    """score(d) = sum over retrievers of 1 / (k + rank_d); ranks only, so no score normalisation needed."""
    name = 'hybrid_rrf'

    def __init__(self, retrievers, k=60, candidates=20):
        self.retrievers, self.k, self.candidates = retrievers, k, candidates

    def search(self, query, *, filters=None, top_k=3):
        fused, chunk_by_id, parts = {}, {}, {}
        for r in self.retrievers:
            for res in r.search(query, filters=filters, top_k=self.candidates):
                fused[res['id']] = fused.get(res['id'], 0.0) + 1.0 / (self.k + res['rank'])
                chunk_by_id[res['id']] = res
                parts.setdefault(res['id'], {})[r.name] = res['rank']
        order = sorted(fused, key=lambda i: (-fused[i], i))[:top_k]
        out = []
        for n, cid in enumerate(order):
            c = {k: v for k, v in chunk_by_id[cid].items() if k not in ('score', 'rank', 'strategy')}
            out.append({**c, 'score': fused[cid], 'rank': n + 1, 'strategy': self.name,
                        'component_ranks': parts[cid]})
        return out


class RerankRetriever:
    """Take `candidates` from a base retriever, re-score (query, chunk) pairs with a cross-encoder.

    segment_chars=None : score the whole chunk text (one pair per candidate).
    segment_chars=N    : also split the chunk into consecutive line groups of <= N chars and use the
                         max score over the whole chunk and its segments (a short relevant passage at
                         the end of a long, mostly unrelated chunk is no longer diluted).
    """
    name = 'rerank'

    def __init__(self, base, model_name='cross-encoder/ms-marco-MiniLM-L-6-v2', candidates=20, scorer=None,
                 segment_chars=None):
        self.base, self.candidates, self.model_name, self.segment_chars = base, candidates, model_name, segment_chars
        if scorer is None:
            from sentence_transformers import CrossEncoder
            model = CrossEncoder(model_name)
            scorer = lambda q, texts: [float(s) for s in model.predict([(q, t) for t in texts])]
        self.scorer = scorer

    @staticmethod
    def segments(text, max_chars):
        segs, cur = [], ''
        for line in text.split('\n'):
            if cur and len(cur) + len(line) + 1 > max_chars:
                segs.append(cur)
                cur = line
            else:
                cur = f'{cur}\n{line}' if cur else line
        if cur:
            segs.append(cur)
        return segs

    def search(self, query, *, filters=None, top_k=3):
        cands = self.base.search(query, filters=filters, top_k=self.candidates)
        if not cands:
            return []
        if not self.segment_chars:
            scores, best_seg = self.scorer(query, [c['text'] for c in cands]), [None] * len(cands)
        else:
            texts, owner = [], []
            for i, c in enumerate(cands):
                parts = [c['text']] + self.segments(c['text'], self.segment_chars)
                texts += parts
                owner += [i] * len(parts)
            flat = self.scorer(query, texts)
            scores, best_seg = [float('-inf')] * len(cands), [None] * len(cands)
            for t, i, sc in zip(texts, owner, flat):
                if sc > scores[i]:
                    scores[i], best_seg[i] = sc, t
        order = sorted(range(len(cands)), key=lambda i: (-scores[i], cands[i]['id']))[:top_k]
        tag = f'{self.base.name}+rerank' + ('_maxseg' if self.segment_chars else '')
        return [{**{k: v for k, v in cands[i].items() if k not in ('score', 'rank', 'strategy')},
                 'score': scores[i], 'rank': n + 1, 'strategy': tag, 'pre_rerank_rank': cands[i]['rank'],
                 **({'best_segment': best_seg[i][:160]} if best_seg[i] else {})} for n, i in enumerate(order)]


def expand_neighbors(results, all_chunks, window=1):
    """Context strategy: after each retrieved chunk, insert its neighbours in the same section.

    Chunk IDs are sequential (B-...-0042); neighbours are marked `neighbor_of` so traces show why
    they entered the context. Retrieved chunks keep priority, ranking metrics are unchanged.
    """
    order = {c['id']: i for i, c in enumerate(all_chunks)}
    out, seen = [], set()
    for r in results:
        if r['id'] not in seen:
            out.append(r)
            seen.add(r['id'])
    for r in results:
        i = order[r['id']]
        for j in [i + d for d in range(-window, window + 1) if d]:
            if 0 <= j < len(all_chunks):
                n = all_chunks[j]
                if n['id'] not in seen and n['metadata'].get('section') == r['metadata'].get('section'):
                    out.append({**n, 'score': None, 'rank': None, 'strategy': 'neighbor', 'neighbor_of': r['id']})
                    seen.add(n['id'])
    return out
