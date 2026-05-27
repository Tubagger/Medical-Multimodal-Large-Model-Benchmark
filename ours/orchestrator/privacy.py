from typing import List, Dict, Any, Optional, Sequence
from ours.orchestrator.base import BaseOrchestrator


class PrivacyOrchestrator(BaseOrchestrator):

    def __init__(self,
                 model,
                 tools: Dict[str, Any],
                 max_rounds: int = 5,
                 conflict_matrix: Optional[List[tuple]] = None,
                 temperature: float = 0.2):
        """
        model: orchestrator 使用的大模型（通常是 o3）
        tools: {tool_name: callable_object}
        conflict_matrix: 不允许同时出现的工具组合
        """

        super().__init__(model=model)

        self.tools = tools
        self.max_rounds = max_rounds
        self.temperature = temperature

        # 默认 DAS 冲突矩阵
        self.conflict_matrix = conflict_matrix or [
            ("inversion", "negation"),
            ("negation", "impossibility"),
            ("negation", "expansion"),
        ]

        # 记录失败的 combos
        self.failed = set()

    # --------------------
    # 是否冲突
    # --------------------
    def is_conflict(self, combo: Sequence[str]):
        combo_set = set(combo)
        for a, b in self.conflict_matrix:
            if a in combo_set and b in combo_set:
                return True
        return False

    # --------------------
    # 工具是否可以应用（是否适用于样本）
    # --------------------
    def tool_applicable(self, tool_name: str, sample: Dict[str, Any]):
        """
        根据工具自身属性判断是否能用，例如：
        - impossibility 需要数值
        - negation 需要肯定句
        - image_attr_swap 需要 image_path
        """
        tool = self.tools[tool_name]
        if hasattr(tool, "applicable"):
            return tool.applicable(sample)
        return True

    # --------------------
    # 工具组是否都能应用
    # --------------------
    def combo_applicable(self, combo: Sequence[str], sample):
        return all(self.tool_applicable(t, sample) for t in combo)

    # --------------------
    # 生成本轮可用的工具集
    # --------------------
    def generate_candidate_combos(self, round_idx: int):

        tool_names = list(self.tools.keys())
        single = [[t] for t in tool_names]
        double = [[a, b] for a in tool_names for b in tool_names if a < b]

        if round_idx <= 3:
            return single + double

        # round4-5: 全部 combos
        from itertools import combinations
        multi = []
        for r in range(3, len(tool_names)+1):
            multi += [list(c) for c in combinations(tool_names, r)]
        return single + double + multi

    # --------------------
    # 真正执行一个 combo
    # --------------------
    def apply_combo(self, combo: Sequence[str], sample: Dict[str, Any]):
        mutated = sample.copy()
        for t in combo:
            tool = self.tools[t]
            mutated = tool.run(mutated)
        return mutated

    # --------------------
    # orchestrate: 多轮改写核心逻辑
    # --------------------
    def orchestrate(self, sample: Dict[str, Any]):
        """
        sample:
            {
                "content": "...",
                "image_path": "...",
                "target": "...",
            }
        """

        original = sample.copy()
        history = []

        for round_idx in range(1, self.max_rounds + 1):

            candidate_combos = self.generate_candidate_combos(round_idx)

            selected_combo = None
            mutated = None

            for combo in candidate_combos:

                combo_key = tuple(sorted(combo))

                # 1. 冲突工具跳过
                if self.is_conflict(combo):
                    continue

                # 2. 上次失败的组合跳过
                if combo_key in self.failed:
                    continue

                # 3. 工具不能用跳过
                if not self.combo_applicable(combo, original):
                    continue

                # 选中本组合
                selected_combo = combo
                mutated = self.apply_combo(combo, original)
                break

            if selected_combo is None:
                # 没有可用工具 → 返回原样
                return {
                    "rounds": history,
                    "final": original,
                    "reason": "no_valid_mutation"
                }

            # 将结果加入历史
            history.append({
                "round": round_idx,
                "selected_tools": selected_combo,
                "mutated": mutated,
            })

            # 每一轮返回 mutated 给外部（由 caller 给 rabbit 模型测试）
            return {
                "rounds": history,
                "final": mutated,
                "selected_tools": selected_combo,
            }

        # unreachable
        return {
            "rounds": history,
            "final": original
        }
