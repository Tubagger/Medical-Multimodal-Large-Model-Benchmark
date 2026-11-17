import os
import torch
import yaml
from typing import List, Dict, Any

from ours.models.base import BaseChat
from ours.utils.registry import registry
from ours.utils.utils import get_abs_path


@registry.register_chatmodel()
class HuatuoGPTVision(BaseChat):

    MODEL_CONFIG = {
        "huatuogpt-v-7b": "configs/models/huatuogpt/huatuogpt-v-7b.yaml",
        "huatuogpt-v-34b": "configs/models/huatuogpt/huatuogpt-v-34b.yaml",
    }

    model_family = list(MODEL_CONFIG.keys())
    model_arch = "huatuogpt-v"
    
    def __init__(self, model_id="huatuogpt-v-7b", **kwargs):
        super().__init__(model_id=model_id)

        with open(get_abs_path(self.MODEL_CONFIG[self.model_id])) as f:
            self.model_config = yaml.load(f, Loader=yaml.FullLoader)

        self.model_path = self.model_config["model_path"]
        from ours.models.huatuogptcli import HuatuoChatbot

        self.gen_kwargs = {
            "do_sample": False,
            "max_new_tokens": 512,
            "min_new_tokens": 1,
            "temperature": 0.0,
            "repetition_penalty": 1.2,
        }

        self.device = "cuda"

        self.model = HuatuoChatbot(self.model_path)


    def chat(self, messages, **generation_kwargs):

        query = ""
        image_paths = []

        for msg in messages:

            if msg["role"] != "user":
                continue

            if isinstance(msg["content"], dict):

                query = msg["content"]["text"]
                image_paths.append(
                    msg["content"]["image_path"]
                )

            else:

                query = msg["content"]

        outputs = self.model.inference(
            query,
            image_paths,
            **generation_kwargs,
        )

        return outputs[0]
