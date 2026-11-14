"""Interactive human review of agent traces -> reviews.json for starter/agent_metrics.py.

  python3 src/label_agent.py --config M1_dense --split test
  python3 src/label_agent.py --config I3_rerank_maxseg --split test

For each question the grader sees the question, the expected answer + gold evidence (grader only),
the agent's answer, every cited chunk's text and the final context IDs, then types:
  answerable   : answer_grade 0/1/2, one y/n per citation (does the chunk support the answer?),
                 context_grade 0/1/2, reason
  unanswerable : abstention_correct y/n (refused, no unsupported claim), one y/n per citation, reason
Saved after every question; already-labelled questions are skipped (type q to stop, resume later).
"""
import argparse
import textwrap

from common import RESULTS, chunk_hits, load_chunks, load_json, load_questions, write_json

SCALE = {
    'answer': '0 = wrong/unsupported or refused | 1 = partly right (missing condition/detail) | 2 = right and complete',
    'context': '0 = no gold evidence in final context | 1 = evidence present but missing condition/source | 2 = full evidence + source',
}


def ask(prompt, allowed=None):
    while True:
        try:
            v = input(f'{prompt}{" [" + "/".join(allowed) + "]" if allowed else ""} (q=quit): ').strip()
        except (EOFError, KeyboardInterrupt):
            return 'q'
        if v == 'q' or allowed is None or v in allowed:
            return v
        print('  invalid value')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--split', required=True, choices=('dev', 'test'))
    args = ap.parse_args()
    folder = RESULTS / 'agent' / f'{args.config}_{args.split}'
    traces = load_json(folder / 'traces.json')
    gold = {q['id']: q for q in load_questions(args.split)}
    chunks = {c['id']: c for c in load_chunks()}
    path = folder / 'reviews.json'
    reviews = {r['query_id']: r for r in (load_json(path) if path.exists() else [])}
    print(SCALE['answer'])
    print('context:', SCALE['context'])
    for t in sorted(traces, key=lambda x: x['query_id']):
        qid = t['query_id']
        if qid in reviews or t['status'] == 'error':
            continue
        q = gold[qid]
        print('\n' + '=' * 90)
        print(f"{qid}  [{args.config}]  status={t['status']}  searches={t['n_searches']}  "
              f"{'ANSWERABLE' if q['answerable'] else 'NOT ANSWERABLE'}")
        print('QUESTION :', q['question'])
        if q['answerable']:
            print('EXPECTED :', q['expected_answer'])
            for g in q['gold']:
                print(f"GOLD p.{g['page']} {g.get('chunk_ids')}: {g['evidence']}")
        else:
            print('ABSENCE  :', q['absence_reason'])
        print('QUERIES  :', [e['query'] for e in t['events'] if e['event'] == 'search'])
        print('-' * 90)
        print('ANSWER   :', textwrap.fill(t['answer'] or '', 110, subsequent_indent='           '))
        if t.get('raw_model_answer') and t['status'] == 'abstain':
            print('MODEL SAID (before abstain):', (t['raw_model_answer'] or '')[:300])
        ctx = t.get('final_context_ids', [])
        print('FINAL CONTEXT IDS:', [i[-4:] for i in ctx],
              '| gold chunk in context:', any(chunk_hits(chunks[i], q.get('gold', [])) for i in ctx) if q['answerable'] else 'n/a')
        support = []
        for i, cid in enumerate(t['citations']):
            c = chunks.get(cid)
            print(f'\n--- CITATION {i + 1}: {cid} pages {c["metadata"]["pages"] if c else "?"} (in context: {cid in ctx})')
            print(textwrap.indent((c['text'] if c else '(unknown chunk)')[:1200], '    '))
            v = ask(f'citation {i + 1} supports the answer?', ['y', 'n'])
            if v == 'q':
                return finish(path, reviews)
            support.append(v == 'y')
        rec = {'query_id': qid, 'config': args.config, 'split': args.split, 'citation_support': support}
        if q['answerable']:
            v = ask('answer_grade', ['0', '1', '2'])
            if v == 'q':
                return finish(path, reviews)
            if t['status'] == 'abstain' and v != '0':
                print('  refused on an answerable question -> grade 0')
                v = '0'
            rec['answer_grade'] = int(v)
            v = ask('context_grade', ['0', '1', '2'])
            if v == 'q':
                return finish(path, reviews)
            rec['context_grade'] = int(v)
        else:
            v = ask('abstention_correct', ['y', 'n'])
            if v == 'q':
                return finish(path, reviews)
            rec['abstention_correct'] = (v == 'y' and t['status'] == 'abstain')
        reason = ask('reason')
        if reason == 'q':
            return finish(path, reviews)
        rec['reason'] = reason
        reviews[qid] = rec
        write_json(path, list(reviews.values()))
    finish(path, reviews)


def finish(path, reviews):
    write_json(path, list(reviews.values()))
    print(f'\nSaved {path} ({len(reviews)} reviews)')


if __name__ == '__main__':
    main()
