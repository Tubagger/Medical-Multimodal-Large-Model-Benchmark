"""Offline checks for the R.4 negation set-answer protocol."""

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from ours.evaluators.rule_eval import GenericMCQEvaluator


def load_prepare_module():
    path = ROOT / "scripts/prepare_r4_negation_rerun.py"
    spec = importlib.util.spec_from_file_location("prepare_r4_negation_rerun", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class NegationPromptTest(unittest.TestCase):
    def setUp(self):
        self.prepare = load_prepare_module()

    def sample(self, text):
        return {
            "id": 0,
            "content": {"text": text, "image_path": "image.png"},
            "target": ["A", "C", "D"],
            "image_sha256": "0" * 64,
            "input_fingerprint": "1" * 64,
        }

    def test_render_replaces_only_the_instruction(self):
        original = "Question:\nWhat is NOT shown?\nOptions:\nA. x\nB. y\n\n" + self.prepare.OLD_INSTRUCTION
        rendered = self.prepare.render(original)
        self.assertIn("What is NOT shown?", rendered)
        self.assertIn("comma-separated set", rendered)
        self.assertNotIn(self.prepare.OLD_INSTRUCTION, rendered)

    def test_render_is_idempotent(self):
        original = "Question:\nWhat is NOT shown?\nOptions:\nA. x\nB. y\n\n" + self.prepare.OLD_INSTRUCTION
        once = self.prepare.render(original)
        self.assertEqual(self.prepare.render(once), once)

    def test_fingerprint_tracks_rendered_text(self):
        sample = self.sample("Question:\nWhat is NOT shown?\n\n" + self.prepare.OLD_INSTRUCTION)
        rendered = self.prepare.render(sample["content"]["text"])
        self.assertNotEqual(
            self.prepare.input_fingerprint(rendered, sample["image_sha256"]),
            sample["input_fingerprint"],
        )
        self.assertEqual(
            self.prepare.input_fingerprint(rendered, sample["image_sha256"]),
            self.prepare.input_fingerprint(rendered, sample["image_sha256"]),
        )

    def test_render_rejects_unknown_prompt(self):
        with self.assertRaises(ValueError):
            self.prepare.render("Question:\nWhat is NOT shown?\nOptions:\nA. x")

    def test_archive_and_runtime_inputs_match(self):
        archive = self.prepare.prepare_samples(ROOT / "data/shared_inputs/R4/question-negation.json")[0]
        runtime = self.prepare.prepare_inputs(
            ROOT / "logs/robustness/R4/robustness-vqa-question-negation.inputs.json"
        )[0]
        self.assertEqual([row["id"] for row in archive], [row["id"] for row in runtime])
        self.assertEqual([sorted(row["target"]) for row in archive],
                         [sorted(row["target"]) for row in runtime])
        self.assertTrue(all(self.prepare.NEW_INSTRUCTION in row["content"]["text"] for row in archive))

    def test_negation_parser_accepts_complete_sets(self):
        evaluator = GenericMCQEvaluator("rule_mcq_eval", {})
        for answer in ["A,C,D", "A, C, D", "ACD", '["A","C","D"]', "Final answer: A, C, D"]:
            self.assertEqual(evaluator._extract_option(answer, list("ABCD"), multi_select=True),
                             ["A", "C", "D"], answer)

    def test_negation_parser_keeps_single_letter_incomplete(self):
        evaluator = GenericMCQEvaluator("rule_mcq_eval", {})
        self.assertEqual(evaluator._extract_option("D", list("ABCD"), multi_select=True), ["D"])

    def test_negation_parser_does_not_absorb_explanation_options(self):
        evaluator = GenericMCQEvaluator("rule_mcq_eval", {})
        self.assertIsNone(evaluator._extract_option("A, C, D (all except B)", list("ABCD"),
                                                    multi_select=True))
        self.assertIsNone(evaluator._extract_option("The correct answer is: **B**", list("ABCD"),
                                                    multi_select=True))

    def test_set_match_requires_exact_equality(self):
        evaluator = GenericMCQEvaluator("rule_mcq_eval", {"robustness_acc": {}})
        preds = ["A,C,D", "D", "A,C"]
        labels = [["A", "C", "D"], ["A", "C", "D"], ["A", "C", "D"]]
        results, true_labels, _, evals = evaluator.process(preds, labels, [None] * len(preds))
        self.assertEqual(list(results), [1, 0, 0])
        self.assertEqual(list(true_labels), [1, 1, 1])
        self.assertEqual([item["match"] for item in evals], [True, False, False])


if __name__ == "__main__":
    unittest.main()
