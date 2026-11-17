import yaml
import torch
from typing import List, Dict

from ours.models.base import BaseChat
from ours.utils.registry import registry
from ours.utils.utils import get_abs_path


@registry.register_chatmodel()
class HealthGPTPro(BaseChat):

    MODEL_CONFIG = {
        "healthgpt-pro-8b": "configs/models/healthgpt/healthgpt-pro-8b.yaml",
    }

    model_family = list(MODEL_CONFIG.keys())
    model_arch = "healthgpt-pro"

    def __init__(self, model_id="healthgpt-pro-8b", **kwargs):
        super().__init__(model_id=model_id)

        from transformers import (
            Qwen3VLForConditionalGeneration,
            AutoProcessor,
        )

        config_path = self.MODEL_CONFIG[self.model_id]

        with open(get_abs_path(config_path)) as f:
            self.model_config = yaml.load(f, Loader=yaml.FullLoader)

        self.model_path = self.model_config["model_path"]

        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            # attn_implementation="flash_attention_2",
            attn_implementation="sdpa",
            device_map="auto",
        )

        self.processor = AutoProcessor.from_pretrained(self.model_path)

    def chat(self, messages: List[Dict], **generation_kwargs):

        from qwen_vl_utils import process_vision_info

        ####################################################
        # 转换成Qwen格式
        ####################################################
        qwen_messages = []

        for msg in messages:

            role = msg["role"]

            if isinstance(msg["content"], dict):

                content = [
                    {
                        "type": "image",
                        "image": msg["content"]["image_path"],
                    },
                    {
                        "type": "text",
                        "text": msg["content"]["text"],
                    },
                ]

            else:

                content = [
                    {
                        "type": "text",
                        "text": msg["content"],
                    }
                ]

            qwen_messages.append(
                {
                    "role": role,
                    "content": content,
                }
            )

        ####################################################
        # 官方推荐方式
        ####################################################

        text = self.processor.apply_chat_template(
            qwen_messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        image_inputs, video_inputs = process_vision_info(qwen_messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        inputs = inputs.to(self.model.device)

        ####################################################
        # generation
        ####################################################

        max_new_tokens = generation_kwargs.get(
            "max_new_tokens", 512
        )

        temperature = generation_kwargs.get(
            "temperature", 0.0
        )

        generated_ids = self.model.generate(
            **inputs,
            **({'bad_words_ids': generation_kwargs['bad_words_ids']}
               if generation_kwargs.get('bad_words_ids') else {}),
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature,
        )

        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(
                inputs.input_ids,
                generated_ids,
            )
        ]

        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        return output_text[0]
