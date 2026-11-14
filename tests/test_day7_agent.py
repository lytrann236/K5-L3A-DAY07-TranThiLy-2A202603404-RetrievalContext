"""Offline checks of the Gemini controller wiring with a FAKE client (no model quality measured)."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
try:
    from agent import GeminiController, RetrieverAdapter, agentic
except ImportError:  # google-genai not installed
    GeminiController = None

CHUNK = {'id': 'B-x-0001', 'text': 'find() returns -1 if the substring does not exist.', 'source_id': 'doc',
         'source': 'doc.pdf', 'page': 207, 'score': 0.9, 'rank': 1,
         'metadata': {'pages': [207], 'section': '8.3', 'chapter': 8}}


class FakeModels:
    def __init__(self, calls): self.calls, self.prompts = iter(calls), []
    def generate_content(self, model, contents, config):
        self.prompts.append(contents)
        name, args = next(self.calls)
        return NS(function_calls=[NS(name=name, args=args)], text='',
                  usage_metadata=NS(prompt_token_count=100, candidates_token_count=10))


class FakeRetriever:
    name = 'fake'
    def search(self, q, *, filters=None, top_k=3): return [CHUNK]


@unittest.skipIf(GeminiController is None, 'google-genai not installed')
class Day7AgentTests(unittest.TestCase):
    def make(self, calls):
        models = FakeModels(calls)
        return GeminiController(client=NS(models=models), min_interval=0), models

    def test_search_then_cited_answer(self):
        c, models = self.make([('search_book', {'query': 'find not found'}),
                               ('final_answer', {'answer': 'It returns -1 [B-x-0001].', 'citations': ['B-x-0001'],
                                                 'insufficient_evidence': False})])
        res = agentic.run_agent('What does find() return?', c, RetrieverAdapter(FakeRetriever()), budget=4000)
        self.assertEqual(res['status'], 'answered')
        self.assertIn('CONTEXT:', models.prompts[1])
        self.assertIn('B-x-0001', models.prompts[1])
        self.assertEqual(len(c.calls), 2)

    def test_missing_inline_marker_is_appended_and_logged(self):
        c, _ = self.make([('search_book', {'query': 'q'}),
                          ('final_answer', {'answer': 'It returns -1.', 'citations': ['B-x-0001'],
                                            'insufficient_evidence': False})])
        res = agentic.run_agent('q?', c, RetrieverAdapter(FakeRetriever()), budget=4000)
        self.assertEqual(res['status'], 'answered')
        self.assertEqual(c.calls[-1]['citation_markers_appended'], ['B-x-0001'])

    def test_insufficient_evidence_abstains(self):
        c, _ = self.make([('search_book', {'query': 'strip'}),
                          ('final_answer', {'answer': 'Not covered.', 'citations': [], 'insufficient_evidence': True})])
        res = agentic.run_agent('What does strip() do?', c, RetrieverAdapter(FakeRetriever()), budget=4000)
        self.assertEqual(res['status'], 'abstain')

    def test_prompt_never_contains_gold_fields(self):
        c, models = self.make([('search_book', {'query': 'q'}),
                               ('final_answer', {'answer': 'x', 'citations': [], 'insufficient_evidence': True})])
        agentic.run_agent('Question?', c, RetrieverAdapter(FakeRetriever()), budget=4000)
        self.assertTrue(all('expected_answer' not in p and 'gold' not in p.lower() for p in models.prompts))


if __name__ == '__main__':
    unittest.main()
