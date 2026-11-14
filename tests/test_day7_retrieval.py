"""Offline checks for Day 7 retrieval helpers (no model, no database)."""
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from common import chunk_hits, pack_context, public_question  # noqa: E402
from retrievers import BM25Retriever, HybridRRF, RerankRetriever, expand_neighbors  # noqa: E402


def ch(i, text, page=1, section='8.1', chapter=8):
    return {'id': f'c{i}', 'text': text, 'source_id': 'doc', 'source': 'doc.pdf', 'page': page,
            'metadata': {'pages': [page, page + 1], 'section': section, 'chapter': chapter}}


CHUNKS = [ch(0, 'find returns -1 if the substring\ndoes not exist.'), ch(1, 'count returns 0 when absent.'),
          ch(2, 'lists are mutable', section='9.1', chapter=9), ch(3, 'other text about lists', section='9.1', chapter=9)]


class Fake:
    def __init__(self, name, ids): self.name, self.ids = name, ids
    def search(self, q, *, filters=None, top_k=3):
        return [{**CHUNKS[i], 'score': 1.0, 'rank': r + 1, 'strategy': self.name} for r, i in enumerate(self.ids[:top_k])]


class Day7RetrievalTests(unittest.TestCase):
    def test_hit_is_whitespace_insensitive_and_uses_all_pages(self):
        gold = [{'source_id': 'doc', 'page': 2, 'evidence': 'if the substring does not exist'}]
        self.assertTrue(chunk_hits(CHUNKS[0], gold))
        self.assertFalse(chunk_hits(CHUNKS[1], gold))
        self.assertFalse(chunk_hits(CHUNKS[0], [{**gold[0], 'page': 9}]))

    def test_public_question_hides_gold(self):
        q = {'id': 'q', 'question': 'x?', 'gold': [1], 'expected_answer': 'y', 'answerable': True, 'filter': {'chapter': 8}}
        self.assertEqual(set(public_question(q)), {'id', 'question', 'filter'})

    def test_pack_context_respects_budget_and_dedup(self):
        ctx, ids = pack_context([CHUNKS[0], CHUNKS[0], CHUNKS[1]], budget=10_000)
        self.assertEqual(ids, ['c0', 'c1'])
        ctx, ids = pack_context(CHUNKS, budget=120)
        self.assertLessEqual(len(ctx), 120)

    def test_bm25_filter(self):
        r = BM25Retriever(CHUNKS).search('mutable lists', filters={'chapter': 8}, top_k=3)
        self.assertTrue(all(c['metadata']['chapter'] == 8 for c in r))

    def test_rrf_prefers_documents_found_by_both(self):
        h = HybridRRF([Fake('dense', [0, 1, 2]), Fake('bm25', [1, 3, 0])], k=60, candidates=3)
        out = h.search('q', top_k=3)
        self.assertEqual(out[0]['id'], 'c1')
        self.assertEqual(out[0]['component_ranks'], {'dense': 2, 'bm25': 1})

    def test_rerank_reorders_candidates(self):
        r = RerankRetriever(Fake('dense', [0, 1, 2]), scorer=lambda q, ts: [0.1, 0.9, 0.5], candidates=3)
        self.assertEqual([c['id'] for c in r.search('q')], ['c1', 'c2', 'c0'])

    def test_maxseg_rerank_uses_best_segment(self):
        long = {**CHUNKS[3], 'text': 'quiz line about loops\n' * 5 + 'count returns 0 when absent'}
        class B:
            name = 'dense'
            def search(self, q, *, filters=None, top_k=3):
                return [{**CHUNKS[0], 'score': 1, 'rank': 1}, {**long, 'score': 1, 'rank': 2}]
        scorer = lambda q, ts: [1.0 if t.strip() == 'count returns 0 when absent' else 0.0 for t in ts]
        whole = RerankRetriever(B(), scorer=scorer).search('count?')
        seg = RerankRetriever(B(), scorer=scorer, segment_chars=40).search('count?')
        self.assertEqual(whole[0]['id'], 'c0')
        self.assertEqual(seg[0]['id'], 'c3')

    def test_neighbors_same_section_only(self):
        res = [{**CHUNKS[2], 'score': 1, 'rank': 1}]
        out = expand_neighbors(res, CHUNKS, window=1)
        self.assertEqual([c['id'] for c in out], ['c2', 'c3'])  # c1 is in another section


if __name__ == '__main__':
    unittest.main()
