"""Shared helpers for Day 7: paths, data loading, embedding, evidence matching, context packing."""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'day7'
RESULTS = ROOT / 'results'
CONFIGS = ROOT / 'configs'
load_dotenv(ROOT / '.env')

EMBED_MODEL = os.getenv('EMBEDDING_MODEL', 'BAAI/bge-small-en-v1.5')  # must match the Day 6 index
BGE_QUERY_PREFIX = 'Represent this sentence for searching relevant passages: '


def collapse(text: str) -> str:
    return re.sub(r'\s+', ' ', text).strip()


def load_json(path: Path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path: Path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def load_chunks() -> list[dict]:
    return load_json(DATA / 'chunks.json')


def load_questions(split: str | None = None) -> list[dict]:
    qs = load_json(DATA / 'questions.json')
    return [q for q in qs if split is None or q['split'] == split]


def public_question(q: dict) -> dict:
    """What a retriever/agent may see: never gold, expected answer or answerability."""
    return {'id': q['id'], 'question': q['question'], 'filter': q.get('filter')}


class QueryEmbedder:
    """Same model and query prefix as the Day 6 index (bge-small-en-v1.5, 384-d)."""

    def __init__(self, model_name: str = EMBED_MODEL):
        from sentence_transformers import SentenceTransformer
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.prefix = BGE_QUERY_PREFIX if 'bge' in model_name.lower() and '-en' in model_name.lower() else ''

    def embed(self, query: str) -> list[float]:
        return self.model.encode([self.prefix + query], normalize_embeddings=True)[0].tolist()


def chunk_hits(chunk: dict, gold: list[dict]) -> bool:
    """Hit = right source, a gold page among the chunk's pages, whole evidence inside the chunk."""
    text = collapse(chunk['text'])
    pages = chunk.get('metadata', {}).get('pages') or [chunk['page']]
    return any(chunk['source_id'] == g['source_id'] and g['page'] in pages and collapse(g['evidence']) in text
               for g in gold)


def pack_context(chunks: list[dict], budget: int) -> tuple[str, list[str]]:
    """Greedy, in rank order, de-duplicated; a chunk that does not fit is skipped (budget in characters)."""
    blocks, ids, seen = [], [], set()
    for c in chunks:
        if c['id'] in seen:
            continue
        seen.add(c['id'])
        m = c.get('metadata', {})
        block = (f"[{c['id']} | pdf page {','.join(map(str, m.get('pages', [c['page']])))} | "
                 f"section {m.get('section', '?')}]\n{c['text']}")
        if len('\n\n'.join(blocks + [block])) <= budget:
            blocks.append(block)
            ids.append(c['id'])
    return '\n\n'.join(blocks), ids
