"""Run the applicable domain judge and response judge in one model call."""
from ours.utils.registry import registry
from .local_judge_base import LocalJudgeEvaluator


@registry.register_evaluator()
class JointJudgeEvaluator(LocalJudgeEvaluator):
    evaluator_ids = ['local_joint_judge']
    # None selects privacy+response for P, safety+response for S.
    judge_names = None
