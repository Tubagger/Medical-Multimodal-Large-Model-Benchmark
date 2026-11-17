import torch
from typing import List, Dict
from PIL import Image
import yaml

from ours.models.base import BaseChat
from ours.utils.registry import registry
from ours.utils.utils import get_abs_path


@registry.register_chatmodel()
class MedGemma(BaseChat):

    MODEL_CONFIG = {
        "med-gemma-4b-it": "configs/models/med_gemma/med-gemma-4b-it.yaml",
        "med-gemma-27b-it": "configs/models/med_gemma/med-gemma-27b-it.yaml",
    }

    model_family = list(MODEL_CONFIG.keys())
    model_arch = "med-gemma"

    def __init__(self, model_id: str = "med-gemma-27b-it", **kwargs):
        from transformers import AutoProcessor, AutoModelForImageTextToText

        super().__init__(model_id=model_id)

        config_path = self.MODEL_CONFIG[self.model_id]
        with open(get_abs_path(config_path), "r") as f:
            self.model_config = yaml.safe_load(f)

        self.model_path = self.model_config.get(
            "model_path",
            "google/medgemma-27b-it"
        )

        # =========================
        # load model
        # =========================
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

        self.processor = AutoProcessor.from_pretrained(self.model_path)
        self.model.eval()

    # =========================
    # chat
    # =========================
    def chat(self, messages: List[Dict], **generation_kwargs):

        # =========================
        # 1. build HF messages
        # =========================
        hf_messages = []

        for msg in messages:
            role = msg["role"]

            if isinstance(msg["content"], dict):
                image = Image.open(msg["content"]["image_path"]).convert("RGB")

                content = [
                    {"type": "image", "image": image},
                    {"type": "text", "text": msg["content"]["text"]},
                ]
            else:
                content = [
                    {"type": "text", "text": msg["content"]}
                ]

            hf_messages.append({
                "role": role,
                "content": content
            })

        # =========================
        # 2. apply chat template
        # =========================
        inputs = self.processor.apply_chat_template(
            hf_messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        )

        # 🚨 IMPORTANT: separate device move (NOT dtype cast)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        # =========================
        # safety checks (VERY important for 1024/1025 bug)
        # =========================
        assert inputs["input_ids"].shape == inputs["attention_mask"].shape, \
            "input_ids and attention_mask shape mismatch!"

        input_len = inputs["input_ids"].shape[-1]

        # =========================
        # 3. generate
        # =========================
        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=generation_kwargs.get("max_new_tokens", 128),
                do_sample=False,
            )

        # =========================
        # 4. decode
        # =========================

        gen_tokens = output[0][input_len:]

        text = self.processor.tokenizer.decode(
            gen_tokens,
            skip_special_tokens=True
        ).strip()


        if len(text) == 0:
            return ""


        return text