"""Retrieval + context benchmark (no LLM). Usage from repo root:

  python3 src/eval_retrieval.py --split dev                      # all configs in configs/retrieval.json
  python3 src/eval_retrieval.py --split dev --configs M1_dense M3_hybrid
  python3 src/eval_retrieval.py --split test --configs M1_dense <SELECTED>   # only after locking

Hit@1 / Evidence Hit@3: answerable questions with a gold-evidence chunk at rank 1 / in top-3.
Context Evidence Rate: answerable questions whose packed context contains a gold-evidence chunk.
Unanswerable questions are run (traces kept) but excluded from these denominators.
Outputs: results/retrieval/<config>_<split>/{traces,summary}.json and results/retrieval/summary_<split>.md
"""
import argparse
import statistics
import time

from common import CONFIGS, RESULTS, chunk_hits, load_chunks, load_json, load_questions, pack_context, \
    public_question, write_json
from retrievers import BM25Retriever, DenseRetriever, HybridRRF, RerankRetriever, expand_neighbors


def build(cfg, cache, chunks):
    method = cfg['method']
    if 'dense' not in cache and method in ('dense', 'hybrid', 'hybrid+rerank', 'hybrid+rerank_maxseg'):
        cache['dense'] = DenseRetriever(chunks)
    if 'bm25' not in cache and method in ('bm25', 'hybrid', 'hybrid+rerank', 'hybrid+rerank_maxseg'):
        cache['bm25'] = BM25Retriever(chunks)
    if method == 'dense':
        return cache['dense']
    if method == 'bm25':
        return cache['bm25']
    hybrid = HybridRRF([cache['dense'], cache['bm25']], k=cfg.get('rrf_k', 60), candidates=cfg.get('candidates', 20))
    if method == 'hybrid':
        return hybrid
    if method in ('hybrid+rerank', 'hybrid+rerank_maxseg'):
        seg = cfg.get('segment_chars') if method.endswith('maxseg') else None
        key = f"rerank:{cfg['reranker']}:{seg}"
        if key not in cache:
            cache[key] = RerankRetriever(hybrid, cfg['reranker'], candidates=cfg.get('candidates', 20),
                                         segment_chars=seg)
        return cache[key]
    raise ValueError(f'unknown method {method}')


def run_config(name, cfg, questions, chunks, cache):
    t0 = time.perf_counter()
    retriever = build(cfg, cache, chunks)
    setup_s = time.perf_counter() - t0
    traces = []
    for q in questions:
        pq = public_question(q)  # gold never reaches the retriever
        t = time.perf_counter()
        results = retriever.search(pq['question'], filters=pq['filter'], top_k=cfg['top_k'])
        ranked = results[:3]
        ctx_chunks = expand_neighbors(results, chunks, cfg['neighbors']) if cfg.get('neighbors') else results
        context, ctx_ids = pack_context(ctx_chunks, cfg['context_budget'])
        ms = (time.perf_counter() - t) * 1000
        gold = q.get('gold', [])
        marks = [chunk_hits(c, gold) for c in ranked] if q['answerable'] else []
        ctx_hit = q['answerable'] and any(chunk_hits(c, gold) for c in ctx_chunks if c['id'] in ctx_ids)
        traces.append({
            'query_id': q['id'], 'split': q['split'], 'answerable': q['answerable'], 'question': q['question'],
            'query_sent': pq['question'], 'filter': pq['filter'], 'config': name, 'method': cfg['method'],
            'top3': [{'rank': c['rank'], 'id': c['id'], 'score': c['score'], 'pages': c['metadata']['pages'],
                      'section': c['metadata']['section'], 'hit': (marks[i] if marks else None),
                      **({'component_ranks': c['component_ranks']} if 'component_ranks' in c else {}),
                      **({'pre_rerank_rank': c['pre_rerank_rank']} if 'pre_rerank_rank' in c else {}),
                      **({'best_segment': c['best_segment']} if 'best_segment' in c else {})}
                     for i, c in enumerate(ranked)],
            'context_ids': ctx_ids, 'context_chars': len(context), 'context': context,
            'neighbors_added': [c['id'] for c in ctx_chunks if c.get('strategy') == 'neighbor' and c['id'] in ctx_ids],
            'hit1': bool(marks and marks[0]), 'hit3': any(marks), 'context_hit': bool(ctx_hit),
            'elapsed_ms': round(ms, 1)})
    ans = [t for t in traces if t['answerable']]
    n = len(ans)
    summary = {'config': name, 'method': cfg['method'], 'split': questions[0]['split'], 'N_answerable': n,
               'N_unanswerable': len(traces) - n,
               'hit1': sum(t['hit1'] for t in ans), 'hit3': sum(t['hit3'] for t in ans),
               'context_hit': sum(t['context_hit'] for t in ans),
               'avg_context_chars': round(statistics.mean(t['context_chars'] for t in traces)),
               'avg_context_chunks': round(statistics.mean(len(t['context_ids']) for t in traces), 2),
               'median_ms_per_query': round(statistics.median(t['elapsed_ms'] for t in traces), 1),
               'setup_seconds': round(setup_s, 2), 'cost_usd': 0, 'config_params': cfg}
    return traces, summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', required=True, choices=('dev', 'test'))
    ap.add_argument('--configs', nargs='*')
    ap.add_argument('--best', help='config name whose method I2_neighbors reuses (method BEST_OF_ABOVE)')
    args = ap.parse_args()
    cfgs = {k: v for k, v in load_json(CONFIGS / 'retrieval.json').items() if not k.startswith('_')}
    if args.split == 'test':
        lock = CONFIGS / 'selected.json'
        if not lock.exists():
            raise SystemExit('Lock the selected config in configs/selected.json before running test.')
        if not args.configs:
            raise SystemExit('On test, name the configs explicitly (baseline + selected).')
    names = args.configs or list(cfgs)
    chunks, questions, cache, rows = load_chunks(), load_questions(args.split), {}, []
    for name in names:
        cfg = dict(cfgs[name])
        if cfg['method'] == 'BEST_OF_ABOVE':
            if not args.best:
                print(f'skip {name}: pass --best <config> to choose which method it extends')
                continue
            cfg = {**cfgs[args.best], **{k: v for k, v in cfg.items() if k != 'method'}, 'based_on': args.best}
        traces, summary = run_config(name, cfg, questions, chunks, cache)
        out = RESULTS / 'retrieval' / f'{name}_{args.split}'
        write_json(out / 'traces.json', traces)
        write_json(out / 'summary.json', summary)
        rows.append(summary)
        print(f"{name:18} Hit@1 {summary['hit1']}/{summary['N_answerable']}  Hit@3 {summary['hit3']}/{summary['N_answerable']}  "
              f"Context {summary['context_hit']}/{summary['N_answerable']}  ctx {summary['avg_context_chars']} chars  "
              f"{summary['median_ms_per_query']} ms/q")
    pct = lambda a, b: f'{a}/{b} ({100 * a / b:.0f}%)' if b else 'N/A'
    lines = ['| Config | Method | top-k | Budget (chars) | Neighbors | Hit@1 | Evidence Hit@3 | Context Evidence Rate | Avg context (chars / chunks) | Median ms/query | Cost |',
             '|---|---|---|---|---|---|---|---|---|---|---|']
    for name in cfgs:  # table = every config already run on this split, in config order
        f = RESULTS / 'retrieval' / f'{name}_{args.split}' / 'summary.json'
        if not f.exists():
            continue
        s = load_json(f)
        p = s['config_params']
        method = s['method'] + (f" (based on {p['based_on']})" if p.get('based_on') else '')
        lines.append(f"| {s['config']} | {method} | {p['top_k']} | {p['context_budget']} | {p.get('neighbors', 0)} | "
                     f"{pct(s['hit1'], s['N_answerable'])} | {pct(s['hit3'], s['N_answerable'])} | "
                     f"{pct(s['context_hit'], s['N_answerable'])} | {s['avg_context_chars']} / {s['avg_context_chunks']} | "
                     f"{s['median_ms_per_query']} | 0 USD |")
    md = RESULTS / 'retrieval' / f'summary_{args.split}.md'
    md.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n' + '\n'.join(lines))


if __name__ == '__main__':
    main()
