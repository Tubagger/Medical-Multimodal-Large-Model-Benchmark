# file: ours/orchestrators/base_orchestrator.py
from typing import Dict, Any, List, Callable

class BaseOrchestrator:
    """
    A generic orchestrator that coordinates:
        - data preprocessing
        - privacy editing / rewriting
        - multimodal encoding
        - model inference
        - postprocessing

    Each step is optional and dynamically pluggable.
    """

    def __init__(self):
        self.preprocess_tools: List[Callable] = []
        self.privacy_tools: List[Callable] = []
        self.encode_tools: List[Callable] = []
        self.inference_fn: Callable = None
        self.postprocess_tools: List[Callable] = []

    # ----------- register functions -----------

    def register_preprocess(self, func: Callable):
        self.preprocess_tools.append(func)

    def register_privacy_tool(self, func: Callable):
        self.privacy_tools.append(func)

    def register_encode(self, func: Callable):
        self.encode_tools.append(func)

    def register_inference(self, func: Callable):
        self.inference_fn = func

    def register_postprocess(self, func: Callable):
        self.postprocess_tools.append(func)

    # ----------- pipeline flow -----------

    def run(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Orchestrate the full pipeline:
            input (text / image dict)
            → preprocess
            → privacy rewrite
            → multimodal encode
            → model inference
            → postprocess
        """

        # ---- 1. preprocess ----
        for tool in self.preprocess_tools:
            sample = tool(sample)

        # ---- 2. privacy rewrite ----
        for tool in self.privacy_tools:
            sample = tool(sample)

        # ---- 3. multimodal encode ----
        for tool in self.encode_tools:
            sample = tool(sample)

        # ---- 4. inference ----
        if self.inference_fn is None:
            raise ValueError("No inference function registered in Orchestrator.")
        output = self.inference_fn(sample)

        # ---- 5. postprocess ----
        for tool in self.postprocess_tools:
            output = tool(output)

        return output
