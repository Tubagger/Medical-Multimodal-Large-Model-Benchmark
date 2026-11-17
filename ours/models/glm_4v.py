import yaml
import torch
from PIL import Image
from typing import List, Dict

from ours.models.base import BaseChat
from ours.utils.registry import registry
from ours.utils.utils import get_abs_path
import warnings
warnings.filterwarnings("ignore")

@registry.register_chatmodel()
class GLM4V(BaseChat):

    MODEL_CONFIG = {
        "glm-4v-9b": "configs/models/glm/glm-4v-9b.yaml",
    }


    model_family = list(MODEL_CONFIG.keys())

    model_arch = "glm4v"



    def __init__(
        self,
        model_id="glm-4v-9b",
        **kwargs
    ):

        super().__init__(
            model_id=model_id
        )


        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer
        )
        from transformers.utils import logging

        logging.set_verbosity_error()

        config_path = self.MODEL_CONFIG[self.model_id]


        with open(
            get_abs_path(config_path)
        ) as f:

            self.model_config = yaml.load(
                f,
                Loader=yaml.FullLoader
            )


        self.model_path = self.model_config.get(
            "model_path",
            "THUDM/glm-4v-9b"
        )


        self.device = "cuda"


        # tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
        )


        # model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map="auto",
        ).eval()



    def chat(
        self,
        messages: List[Dict],
        **generation_kwargs
    ):


        msg = messages[-1]


        content = msg["content"]


        # =====================
        # 多模态输入
        # =====================

        if isinstance(content, dict):

            image_path = content["image_path"]

            query = content["text"]


            image = Image.open(
                image_path
            ).convert("RGB")


            glm_messages = [
                {
                    "role": "user",
                    "image": image,
                    "content": query,
                }
            ]


        else:

            # 纯文本
            glm_messages = [
                {
                    "role": "user",
                    "content": content,
                }
            ]



        # =====================
        # 构造输入
        # =====================

        inputs = self.tokenizer.apply_chat_template(
            glm_messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
        )


        inputs = inputs.to(
            self.model.device
        )



        # =====================
        # generate
        # =====================

        max_new_tokens = generation_kwargs.get(
            "max_new_tokens",
            512
        )


        temperature = generation_kwargs.get(
            "temperature",
            0
        )


        gen_kwargs = {
            "max_length": 2500,
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
        }


        if temperature > 0:
            gen_kwargs["temperature"] = temperature



        with torch.no_grad():

            outputs = self.model.generate(
                **inputs,
                **gen_kwargs
            )


        # 去掉prompt

        outputs = outputs[
            :,
            inputs["input_ids"].shape[1]:
        ]


        response = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )


        return response