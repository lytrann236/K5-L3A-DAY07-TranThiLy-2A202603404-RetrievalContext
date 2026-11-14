"""Run the Gemini agent on a split with one retrieval config. Usage (repo root):

  python3 src/run_agent_bench.py --retrieval M1_dense --split dev --limit 2     # smoke run
  python3 src/run_agent_bench.py --retrieval M1_dense --split dev
  python3 src/run_agent_bench.py --retrieval I3_rerank_maxseg --split dev
  python3 src/run_agent_bench.py --retrieval <...> --split test                 # only after configs/selected.json

Resumable: questions already in traces.json (status answered/abstain) are skipped; errors are retried.
The agent sees only the question text and the user-provided filter. Gold is used afterwards,
only to compute the agent's final-context evidence rate in summary.json.
Outputs: results/agent/<retrieval>_<split>/{traces,summary}.json
"""
import argparse
import statistics
import time
import traceback

from agent import GeminiController, RetrieverAdapter, agentic
from common import CONFIGS, RESULTS, chunk_hits, load_chunks, load_json, load_questions, public_question, write_json
from eval_retrieval import build

AGENT_DEFAULTS = {'max_searches': 3, 'top_k': 3, 'budget': 4000}


def slim_events(events):
    out = []
    for e in events:
        if e['event'] == 'search':
            out.append({'event': 'search', 'tool': 'search_book', 'query': e['query'], 'filter': e['filter'],
                        'results': [{'id': r['id'], 'rank': r.get('rank'), 'score': r.get('score'),
                                     'pages': r['metadata']['pages'], 'section': r['metadata']['section']}
                                    for r in e['results']],
                        'context_ids': e['context_ids'], 'context_chars': len(e['context'])})
        else:
            out.append(e)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--retrieval', required=True, help='config name in configs/retrieval.json')
    ap.add_argument('--split', required=True, choices=('dev', 'test'))
    ap.add_argument('--limit', type=int)
    ap.add_argument('--only', nargs='*')
    for k, v in AGENT_DEFAULTS.items():
        ap.add_argument(f'--{k.replace("_", "-")}', type=int, default=v)
    args = ap.parse_args()
    if args.split == 'test' and not (CONFIGS / 'selected.json').exists():
        raise SystemExit('Lock the selected config in configs/selected.json before running test.')
    cfg = load_json(CONFIGS / 'retrieval.json')[args.retrieval]
    if cfg.get('neighbors'):
        raise SystemExit('Neighbour expansion is evaluated in eval_retrieval only; pick a config without it.')
    chunks = load_chunks()
    retriever = RetrieverAdapter(build(cfg, {}, chunks))
    controller = GeminiController(max_searches=args.max_searches)
    out_dir = RESULTS / 'agent' / f'{args.retrieval}_{args.split}'
    path = out_dir / 'traces.json'
    traces = {t['query_id']: t for t in (load_json(path) if path.exists() else [])}
    questions = load_questions(args.split)
    if args.only:
        questions = [q for q in questions if q['id'] in args.only]
    todo = [q for q in questions if traces.get(q['id'], {}).get('status') not in ('answered', 'abstain')]
    if args.limit:
        todo = todo[:args.limit]
    for q in todo:
        pq = public_question(q)
        controller.reset()
        retriever.latencies = []
        t0 = time.perf_counter()
        rec = {'query_id': q['id'], 'split': q['split'], 'question': q['question'], 'filter': pq['filter'],
               'retrieval_config': args.retrieval, 'model': controller.model, 'agent_params': vars(args)}
        try:
            res = agentic.run_agent(pq['question'], controller, retriever, filters=pq['filter'],
                                    max_searches=args.max_searches, top_k=args.top_k, budget=args.budget)
            searches = [e for e in res['events'] if e['event'] == 'search']
            rec.update({'status': res['status'], 'answer': res['answer'], 'citations': res['citations'],
                        'events': slim_events(res['events']),
                        'final_context': searches[-1]['context'] if searches else '',
                        'final_context_ids': searches[-1]['context_ids'] if searches else [],
                        'raw_model_answer': next((c['decision'].get('args', {}).get('answer') for c in reversed(controller.calls)
                                                  if c['decision'].get('tool') == 'final_answer'), None)})
        except Exception as exc:  # provider / database errors are recorded, never replaced by fake output
            rec.update({'status': 'error', 'answer': None, 'citations': [], 'events': [],
                        'error': f'{type(exc).__name__}: {exc}', 'traceback': traceback.format_exc()[-1500:]})
        rec.update({'n_searches': sum(e['event'] == 'search' for e in rec['events']),
                    'model_calls': controller.calls, 'n_model_calls': len(controller.calls),
                    'prompt_tokens': sum(c['prompt_tokens'] or 0 for c in controller.calls),
                    'output_tokens': sum(c['output_tokens'] or 0 for c in controller.calls),
                    'retrieval_ms': retriever.latencies, 'wall_seconds': round(time.perf_counter() - t0, 2),
                    'cost_usd': 0.0})
        traces[q['id']] = rec
        write_json(path, list(traces.values()))  # save after every question (resumable)
        print(f"{q['id']:4} {rec['status']:8} searches={rec['n_searches']} calls={rec['n_model_calls']} "
              f"{rec['wall_seconds']}s  {(rec.get('answer') or rec.get('error') or '')[:90]}")

    # summary over all traces of this split (evaluation only: gold is read here, after the runs)
    gold = {q['id']: q for q in load_questions(args.split)}
    by_id = {c['id']: c for c in chunks}
    ts = [t for t in traces.values() if t['query_id'] in gold]
    ok = [t for t in ts if t['status'] != 'error']
    ans = [t for t in ok if gold[t['query_id']]['answerable']]
    ctx_hit = sum(any(chunk_hits(by_id[i], gold[t['query_id']]['gold']) for i in t.get('final_context_ids', []))
                  for t in ans)
    dist = {}
    for t in ok:
        dist[t['n_searches']] = dist.get(t['n_searches'], 0) + 1
    summary = {'retrieval_config': args.retrieval, 'split': args.split, 'model': controller.model,
               'agent_params': {k: getattr(args, k) for k in AGENT_DEFAULTS}, 'n_traces': len(ts),
               'errors': len(ts) - len(ok),
               'status': {s: sum(t['status'] == s for t in ts) for s in ('answered', 'abstain', 'error')},
               'tool_use': f"{sum(t['n_searches'] > 0 for t in ok)}/{len(ok)}",
               'searches_distribution': dict(sorted(dist.items())),
               'agent_context_evidence': f'{ctx_hit}/{len(ans)}',
               'answerable_abstained': f"{sum(t['status'] == 'abstain' for t in ans)}/{len(ans)}",
               'unanswerable_abstained': f"{sum(t['status'] == 'abstain' for t in ok if not gold[t['query_id']]['answerable'])}"
                                         f"/{sum(not gold[t['query_id']]['answerable'] for t in ok)}",
               'model_calls_total': sum(t['n_model_calls'] for t in ts),
               'prompt_tokens_total': sum(t['prompt_tokens'] for t in ts),
               'output_tokens_total': sum(t['output_tokens'] for t in ts),
               'median_wall_seconds': statistics.median(t['wall_seconds'] for t in ts) if ts else None,
               'cost_usd': 0.0, 'cost_note': 'Gemini API free tier'}
    write_json(out_dir / 'summary.json', summary)
    print(summary)


if __name__ == '__main__':
    main()
