"""P.3/P.4 privacy judge, with separate task rubrics."""
from .local_judge_base import Judge, LocalJudgeEvaluator
from .chatmodel_eval import LLMJudgeEvaluator
from ours.utils.registry import registry


class PrivacyJudge(Judge):
    name = 'privacy'
    score_key = 'violation'
    allowed_scores = (0, 0.5, 1)

    def validate(self, value, response):
        return LLMJudgeEvaluator.extract_privacy_judgment(value)


@registry.register_evaluator()
class PrivacyJudgeEvaluator(LocalJudgeEvaluator):
    evaluator_ids = ['local_privacy_judge']
    judge_names = ['privacy']
