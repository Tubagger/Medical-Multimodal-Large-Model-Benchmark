"""S.1/S.2/S.3 harmful-compliance judge."""
from .local_judge_base import Judge, LocalJudgeEvaluator
from .chatmodel_eval import LLMJudgeEvaluator
from ours.utils.registry import registry


class SafetyJudge(Judge):
    name = 'safety'
    score_key = 'harmful_compliance'
    allowed_scores = (0, 1)

    def validate(self, value, response):
        return LLMJudgeEvaluator.extract_safety_judgment(value)


@registry.register_evaluator()
class SafetyJudgeEvaluator(LocalJudgeEvaluator):
    evaluator_ids = ['local_safety_judge']
    judge_names = ['safety']
