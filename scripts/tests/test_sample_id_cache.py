"""Renumbered checkpoints must match both original content and judge protocol."""
import ast
import hashlib
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]


def sha(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     allow_nan=False).encode()).hexdigest()


class CacheMigrationTest(unittest.TestCase):
    def test_legacy_cache_requires_exact_content_and_protocol(self):
        path = ROOT / 'scripts/judge_saved_answers.py'
        tree = ast.parse(path.read_text())
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'cached_result')
        namespace = {'sha': sha}
        exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), 'exec'), namespace)
        lookup = namespace['cached_result']
        case = {'USER_PROMPT': 'question', 'MODEL_RESPONSE': 'answer'}
        old_key = sha({'protocol': 'protocol', 'id': 'old-id', 'case': case})
        new_key = sha({'protocol': 'protocol', 'id': 0, 'case': case})
        original = {'id': 0, 'original_sample_id': 'old-id', 'fingerprint': old_key,
                    'status': 'ok', 'judgments': {'score': 1}}
        legacy = {'0': original}
        result = lookup({}, legacy, new_key, 'protocol', 0, case)
        self.assertEqual(result, {'id': 0, 'fingerprint': new_key, 'status': 'ok',
                                  'judgments': {'score': 1}})
        self.assertIn('original_sample_id', original)
        self.assertIsNone(lookup({}, legacy, new_key, 'different-protocol', 0, case))
        self.assertIsNone(lookup({}, legacy, new_key, 'protocol', 0,
                                 {**case, 'MODEL_RESPONSE': 'different answer'}))
        self.assertIsNone(lookup({}, legacy, new_key, 'protocol', 1, case))
        self.assertIs(lookup({new_key: result}, {}, new_key, 'protocol', 0, case), result)


if __name__ == '__main__':
    unittest.main()
