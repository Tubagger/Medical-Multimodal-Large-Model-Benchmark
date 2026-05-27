import os
import yaml
import torch
from typing import List, Dict

from ours.models.base import BaseChat
from ours.utils.registry import registry
from ours.utils.utils import get_abs_path

from transformers import Qwen3VLForConditionalGeneration, AutoProcessor


@registry.register_chatmodel()
class Qwen3vl(BaseChat):
    MODEL_CONFIG = {
        "qwen3-vl-32b-instruct": "configs/models/qwen/qwen3-vl-32b-instruct.yaml",
    }

    model_family = list(MODEL_CONFIG.keys())
    model_arch = "qwen3-vl"

    def __init__(self, model_id: str = "qwen3-vl-32b-instruct", **kwargs):
        super().__init__(model_id=model_id)

        # =========================
        # config
        # =========================
        config_path = self.MODEL_CONFIG[self.model_id]
        with open(get_abs_path(config_path)) as f:
            self.model_config = yaml.load(f, Loader=yaml.FullLoader)

        self.model_path = self.model_config.get(
            "model_path",
            "/media/aiotlab/T7/lhx/Qwen3-VL-32B-Instruct"
        )

        # =========================
        # GPU check
        # =========================
        self.num_gpus = torch.cuda.device_count()
        # if self.num_gpus < 4:
        #     raise RuntimeError("Qwen3-VL-32B 建议至少 4 GPU")

        # =========================
        # MODEL (OFFICIAL STYLE)
        # =========================
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.model_path,
            dtype="auto",
            device_map="auto",
            trust_remote_code=True
        )

        self.processor = AutoProcessor.from_pretrained(self.model_path)

        self.model.eval()

    def chat(self, messages: List[Dict], **generation_kwargs):

        # =========================
        # 1. build image + text messages
        # =========================
        qwen_messages = []
        image_list = []

        for msg in messages:
            role = msg["role"]

            if isinstance(msg["content"], dict):
                image_path = msg["content"]["image_path"]

                content = [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": msg["content"]["text"]},
                ]

                image_list.append(image_path)

            else:
                content = [
                    {"type": "text", "text": msg["content"]}
                ]

            qwen_messages.append({
                "role": role,
                "content": content
            })

        # =========================
        # 2. processor (OFFICIAL STYLE)
        # =========================
        inputs = self.processor.apply_chat_template(
            qwen_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )

        # move to correct device
        inputs = inputs.to(self.model.device)

        # =========================
        # 3. generation
        # =========================
        max_new_tokens = min(
            generation_kwargs.get("max_new_tokens", 128),
            128
        )

        temperature = generation_kwargs.get("temperature", 0.0)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=False
            )

        # =========================
        # 4. trim prompt tokens (OFFICIAL)
        # =========================
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )

        return output_text[0]