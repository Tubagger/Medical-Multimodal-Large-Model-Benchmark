import yaml
import torch
from typing import List, Dict

from transformers import AutoModelForCausalLM

from ours.models.base import BaseChat
from ours.utils.registry import registry
from ours.utils.utils import get_abs_path


@registry.register_chatmodel()
class DeepSeekVL2(BaseChat):

    MODEL_CONFIG = {
        "deepseek-vl2-small": "configs/models/deepseek/deepseek-vl2-small.yaml",
        "deepseek-vl2": "configs/models/deepseek/deepseek-vl2.yaml",
    }

    model_family = list(MODEL_CONFIG.keys())
    model_arch = "deepseek-vl2"

    def __init__(self,
                 model_id="deepseek-vl2-small",
                 **kwargs):

        super().__init__(model_id=model_id)

        from deepseek_vl2.models import (
            DeepseekVLV2Processor,
            DeepseekVLV2ForCausalLM,
        )

        config_path = self.MODEL_CONFIG[self.model_id]

        with open(get_abs_path(config_path)) as f:
            self.model_config = yaml.load(
                f,
                Loader=yaml.FullLoader,
            )

        self.model_path = self.model_config.get(
            "model_path",
            "deepseek-ai/deepseek-vl2-small",
        )

        self.processor = DeepseekVLV2Processor.from_pretrained(
            self.model_path
        )

        self.tokenizer = self.processor.tokenizer

        self.model: DeepseekVLV2ForCausalLM = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        ).eval()

    def chat(
        self,
        messages: List[Dict],
        **generation_kwargs
    ):

        from deepseek_vl2.utils.io import load_pil_images

        msg = messages[-1]

        content = msg["content"]

        # ----------------------------
        # 构造 conversation
        # ----------------------------

        if isinstance(content, dict):

            image_path = content["image_path"]

            question = content["text"]

            conversation = [
                {
                    "role": "<|User|>",
                    "content": "<image>\n" + question,
                    "images": [image_path],
                },
                {
                    "role": "<|Assistant|>",
                    "content": "",
                },
            ]

        else:

            conversation = [
                {
                    "role": "<|User|>",
                    "content": content,
                },
                {
                    "role": "<|Assistant|>",
                    "content": "",
                },
            ]

        # ----------------------------
        # Processor
        # ----------------------------

        pil_images = load_pil_images(conversation)

        prepare_inputs = self.processor(
            conversations=conversation,
            images=pil_images,
            force_batchify=True,
            system_prompt="",
        ).to(self.model.device)

        # ----------------------------
        # Vision Encoder
        # ----------------------------

        inputs_embeds = self.model.prepare_inputs_embeds(
            **prepare_inputs
        )

        # ----------------------------
        # Generation
        # ----------------------------

        outputs = self.model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=prepare_inputs.attention_mask,
            pad_token_id=self.tokenizer.eos_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=generation_kwargs.get(
                "max_new_tokens",
                512,
            ),
            do_sample=generation_kwargs.get(
                "temperature",
                0,
            ) > 0,
            temperature=generation_kwargs.get(
                "temperature",
                0,
            ),
            use_cache=True,
        )

        answer = self.tokenizer.decode(
            outputs[0].cpu().tolist(),
            skip_special_tokens=True,
        )

        return answer