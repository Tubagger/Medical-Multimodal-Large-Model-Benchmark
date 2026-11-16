"""Context-sensitive response appropriateness, distinct from binary refusal."""
from .local_judge_base import Judge, LocalJudgeEvaluator
from .chatmodel_eval import LLMJudgeEvaluator
from ours.utils.registry import registry


class ResponseAppropriatenessJudge(Judge):
    name = 'response'
    score_key = 'appropriateness'
    allowed_scores = (0, 0.5, 1)

    def validate(self, value, response):
        return LLMJudgeEvaluator.extract_response_judgment(value)


@registry.register_evaluator()
class ResponseAppropriatenessEvaluator(LocalJudgeEvaluator):
    evaluator_ids = ['response_appropriateness_judge']
    judge_names = ['response']
