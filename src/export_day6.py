"""Export Day 6 chunks (pgvector) and benchmark into the Day 7 JSON contract.

Reads chunks of one Day 6 index directly from PostgreSQL, converts the Day 6
questions (gold_page/evidence_quote) into Day 7 `gold` lists and adds the
unanswerable questions. Every gold evidence is checked to lie inside a chunk of
the right source whose pages include the gold page (whitespace-insensitive).

Usage (repo root, .env has DATABASE_URL):
  python3 src/export_day6.py --day6-questions ../K5-L3A-DAY06-.../benchmark/questions.jsonl
"""
import argparse
import json
import os
import re
from pathlib import Path

import psycopg
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data' / 'day7'


def collapse(text):
    return re.sub(r'\s+', ' ', text).strip()


def export_chunks(url, index, model):
    sql = ("SELECT chunk_id, document_id, source, page, pages, chapter, section, text, char_len, "
           "content_hash, metadata FROM day6_chunks WHERE index_name=%s AND model=%s ORDER BY chunk_id")
    with psycopg.connect(url, connect_timeout=5) as conn:
        rows = conn.execute(sql, (index, model)).fetchall()
    if not rows:
        raise SystemExit(f'No chunks for index={index} model={model}. Is Docker running and Day 6 ingested?')
    chunks = []
    for cid, doc, source, page, pages, chapter, section, text, clen, chash, meta in rows:
        chunks.append({'id': cid, 'text': text, 'source_id': doc, 'source': source, 'page': page,
                       'metadata': {'pages': list(pages), 'book_pages': [p - 10 for p in pages],
                                    'chapter': chapter, 'section': section, 'char_len': clen,
                                    'content_hash': chash, 'char_start': meta.get('char_start'),
                                    'char_end': meta.get('char_end'), 'day6_index': index,
                                    'day6_embedding_model': model}})
    return chunks


def convert_question(q):
    ev, pages = q['evidence_quote'], q['gold_page']
    if isinstance(ev, list):  # alternatives paired with pages
        gold = [{'source_id': q['gold_source_id'], 'page': p, 'evidence': e} for e, p in zip(ev, pages)]
    else:
        gold = [{'source_id': q['gold_source_id'], 'page': p, 'evidence': ev} for p in pages]
    out = {'id': q['id'], 'split': q['split'], 'type': q.get('type'), 'question': q['question'],
           'expected_answer': q['expected_answer'], 'answerable': True,
           'expected_behavior': 'search_and_answer', 'gold': gold}
    if q.get('filter'):
        out['filter'] = q['filter']
    return out


def gold_in_chunks(g, chunks):
    ev = collapse(g['evidence'])
    return [c['id'] for c in chunks if c['source_id'] == g['source_id']
            and g['page'] in c['metadata']['pages'] and ev in collapse(c['text'])]


def main():
    load_dotenv(ROOT / '.env')
    ap = argparse.ArgumentParser()
    ap.add_argument('--day6-questions', required=True, type=Path)
    ap.add_argument('--index', default='B', help='Day 6 index to reuse (B = structure-aware chunks)')
    ap.add_argument('--model', default='BAAI/bge-small-en-v1.5')
    args = ap.parse_args()
    chunks = export_chunks(os.environ['DATABASE_URL'], args.index, args.model)
    day6 = [json.loads(l) for l in args.day6_questions.read_text(encoding='utf-8').splitlines() if l.strip()]
    questions = [convert_question(q) for q in day6]
    questions += json.loads((OUT / 'unanswerable.json').read_text(encoding='utf-8'))
    problems = []
    for q in questions:
        for g in q.get('gold', []):
            g['chunk_ids'] = gold_in_chunks(g, chunks)  # informative only, never sent to agent/retriever
        if q['answerable'] and not any(g['chunk_ids'] for g in q['gold']):
            problems.append(q['id'])
    cases = [{'id': q['id'], 'split': q['split'], 'answerable': q['answerable']} for q in questions]
    OUT.mkdir(parents=True, exist_ok=True)
    for name, obj in (('chunks.json', chunks), ('questions.json', questions), ('cases.json', cases)):
        (OUT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    count = lambda s, a: sum(q['split'] == s and q['answerable'] == a for q in questions)
    print(f"chunks: {len(chunks)} (index {args.index}, {args.model})")
    print(f"dev: {count('dev', True)} answerable + {count('dev', False)} unanswerable | "
          f"test: {count('test', True)} answerable + {count('test', False)} unanswerable")
    split_ev = [q['id'] for q in questions if q['answerable'] and any(not g['chunk_ids'] for g in q['gold'])]
    print('alternatives not contained in a single chunk (ok if another alternative is):', split_ev)
    print('questions whose evidence is in NO chunk:', problems or 'none')


if __name__ == '__main__':
    main()
