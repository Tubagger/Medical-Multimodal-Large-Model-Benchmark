import os
import yaml
from ours.models.base import BaseChat
from ours.utils.registry import registry
from ours.utils.utils import get_abs_path
import torch
from typing import List, Dict, Any
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

@registry.register_chatmodel()
class Lingshu(BaseChat):
    MODEL_CONFIG = {
        "lingshu-7b": "configs/models/lingshu/lingshu-7b.yaml",
        "lingshu-32b": "configs/models/lingshu/lingshu-32b.yaml",
    }
    
    model_family = list(MODEL_CONFIG.keys())
    model_arch = "lingshu"

    def __init__(self,  model_id: str = "lingshu-32b", **kwargs):
        
        super().__init__(model_id=model_id)
        config_path = self.MODEL_CONFIG[self.model_id]
        with open(get_abs_path(config_path)) as f:
            self.model_config = yaml.load(f, Loader=yaml.FullLoader)

        # 参数
        self.max_retries = self.model_config.get("max_retries", 10)
        self.timeout = self.model_config.get("timeout", 2)
        self.model_path = self.model_config.get("model_path", "/media/aiotlab/T7/lhx/Lingshu-32B")


        # ✅ 加载本地模型
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
        )

        self.processor = AutoProcessor.from_pretrained(self.model_path)

    def chat(self, messages: List[Dict], **generation_kwargs):

        # =========================
        # 1. 转换 messages → Qwen格式
        # =========================
        qwen_messages = []

        for msg in messages:
            role = msg["role"]

            if isinstance(msg["content"], dict):
                # multimodal
                content = [
                    {
                        "type": "image",
                        "image": msg["content"]["image_path"],
                    },
                    {
                        "type": "text",
                        "text": msg["content"]["text"],
                    }
                ]
            else:
                # 纯文本
                content = [
                    {
                        "type": "text",
                        "text": msg["content"]
                    }
                ]

            qwen_messages.append({
                "role": role,
                "content": content
            })

        # =========================
        # 2. 构造输入
        # =========================
        text = self.processor.apply_chat_template(
            qwen_messages,
            tokenize=False,
            add_generation_prompt=True
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

        # =========================
        # 3. 推理
        # =========================
        max_new_tokens = generation_kwargs.get("max_new_tokens", 512)
        temperature = generation_kwargs.get("temperature", 0.0)

        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=False
        )

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