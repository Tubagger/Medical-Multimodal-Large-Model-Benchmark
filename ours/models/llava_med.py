import os
from pathlib import Path
from typing import List, Dict, Any

import torch
import yaml
from PIL import Image

from ours.models.base import BaseChat
from ours.utils.registry import registry
from ours.utils.utils import get_abs_path

import warnings
warnings.filterwarnings("ignore")


@registry.register_chatmodel()
class LlavaMed(BaseChat):

    MODEL_CONFIG = {
        "llava-med-v1.5-mistral-7b":
            "configs/models/llava_med/llava-med-v1.5-mistral-7b.yaml"
    }

    model_family = list(MODEL_CONFIG.keys())
    model_arch = "llava_med"


    def __init__(
        self,
        model_id: str = "llava-med-v1.5-mistral-7b",
        **kwargs,
    ):

        super().__init__(
            model_id=model_id
        )

        from transformers import (
            AutoProcessor,
            LlavaForConditionalGeneration,
        )


        if self.model_id not in self.MODEL_CONFIG:
            raise ValueError(
                f"Unsupported model_id: {self.model_id}"
            )


        config_path = get_abs_path(
            self.MODEL_CONFIG[self.model_id]
        )


        with open(
            config_path,
            "r",
            encoding="utf-8"
        ) as f:

            self.model_config = yaml.load(
                f,
                Loader=yaml.FullLoader
            )


        self.model_path = self.model_config.get(
            "model_path"
        )


        self.max_new_tokens = self.model_config.get(
            "max_new_tokens",
            512
        )


        self.device_map = self.model_config.get(
            "device_map",
            "auto"
        )


        self.attn_implementation = self.model_config.get(
            "attn_implementation",
            "eager"
        )


        dtype_name = self.model_config.get(
            "torch_dtype",
            "bfloat16"
        )


        self.torch_dtype = self._get_torch_dtype(
            dtype_name
        )


        # =========================
        # Load model
        # =========================

        self.model = (
            LlavaForConditionalGeneration
            .from_pretrained(
                self.model_path,
                torch_dtype=self.torch_dtype,
                device_map=self.device_map,
                attn_implementation=
                    self.attn_implementation,
                low_cpu_mem_usage=True,
            )
        )


        self.model.eval()


        # =========================
        # Load processor
        # =========================

        self.processor = (
            AutoProcessor
            .from_pretrained(
                self.model_path,
                use_fast=False,
                trust_remote_code=True,
            )
        )
        # LLaVA-Med v1.5 required config
        self.processor.patch_size = 14
        self.processor.vision_feature_select_strategy = "default"


    @staticmethod
    def _get_torch_dtype(
        dtype_name: str
    ):

        mapping = {

            "float16":
                torch.float16,

            "fp16":
                torch.float16,

            "bfloat16":
                torch.bfloat16,

            "bf16":
                torch.bfloat16,

            "float32":
                torch.float32,

            "fp32":
                torch.float32,
        }


        dtype_name = str(
            dtype_name
        ).lower()


        if dtype_name not in mapping:

            raise ValueError(
                f"Unsupported dtype {dtype_name}"
            )


        return mapping[dtype_name]



    @staticmethod
    def _load_image(
        image_path: str
    ):

        if image_path.startswith(
            "file://"
        ):
            image_path = image_path[7:]


        image_path = os.path.expanduser(
            image_path
        )


        if not os.path.exists(
            image_path
        ):
            raise FileNotFoundError(
                image_path
            )


        return (
            Image.open(image_path)
            .convert("RGB")
        )



    @staticmethod
    def _normalize_content(
        content
    ):


        if isinstance(
            content,
            str
        ):

            return [
                {
                    "type":
                        "text",

                    "text":
                        content
                }
            ]


        if isinstance(
            content,
            dict
        ):

            result = []


            if content.get(
                "image_path"
            ):

                result.append(
                    {
                        "type":
                            "image",

                        "image_path":
                            content["image_path"]
                    }
                )


            if content.get(
                "text"
            ):

                result.append(
                    {
                        "type":
                            "text",

                        "text":
                            content["text"]
                    }
                )


            return result



        if isinstance(
            content,
            list
        ):

            result=[]


            for item in content:

                if item["type"]=="text":

                    result.append(
                        {
                            "type":
                                "text",

                            "text":
                                item["text"]
                        }
                    )


                elif item["type"] in [
                    "image",
                    "image_url"
                ]:

                    path = (
                        item.get("image_path")
                        or
                        item.get("image")
                    )


                    result.append(
                        {
                            "type":
                                "image",

                            "image_path":
                                path
                        }
                    )


            return result



        raise TypeError(
            type(content)
        )



    def _prepare_conversation(
        self,
        messages
    ):

        hf_messages=[]

        images=[]


        for message in messages:


            content = self._normalize_content(
                message["content"]
            )


            hf_content=[]


            for item in content:


                if item["type"]=="image":


                    image = self._load_image(
                        item["image_path"]
                    )


                    images.append(
                        image
                    )


                    hf_content.append(
                        {
                            "type":
                                "image"
                        }
                    )


                else:


                    hf_content.append(
                        {
                            "type":
                                "text",

                            "text":
                                item["text"]
                        }
                    )



            hf_messages.append(
                {
                    "role":
                        message["role"],

                    "content":
                        hf_content
                }
            )



        return hf_messages, images
    
    def _build_prompt(
        self,
        messages
    ):
        rendered = []
        for message in messages:
            content = message["content"]
            text = "".join(item["text"] for item in content if item["type"] == "text")
            if any(item["type"] == "image" for item in content):
                text = "<image>\n" + text
            rendered.append({"role": message["role"], "content": text})

        template = (Path(self.model_path) / "chat_template.jinja").read_text(encoding="utf-8")
        return self.processor.tokenizer.apply_chat_template(
            rendered, chat_template=template, tokenize=False, add_generation_prompt=True
        )
    def _get_input_device(
        self
    ):
        """
        Get model input device when using device_map='auto'
        """

        try:
            return (
                self.model
                .get_input_embeddings()
                .weight.device
            )

        except Exception:
            pass


        try:
            return next(
                self.model.parameters()
            ).device

        except StopIteration:

            return torch.device(
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

    def chat(
        self,
        messages: List[Dict[str, Any]],
        **generation_kwargs,
    ):

        # =========================
        # 1. Prepare messages
        # =========================

        hf_messages, images = (
            self._prepare_conversation(
                messages
            )
        )


        # =========================
        # 2. Build LLaVA prompt
        # =========================

        prompt = self._build_prompt(
            hf_messages
        )


        # print("\n===== PROMPT =====")
        # print(prompt)
        # print("==================\n")



        # =========================
        # 3. Processor
        # =========================

        inputs = self.processor(
            text=prompt,
            images=images if images else None,
            padding=True,
            return_tensors="pt",
        )



        # =========================
        # 4. Move device
        # =========================

        input_device = (
            self._get_input_device()
        )


        inputs = {
            k:
            v.to(input_device)
            if isinstance(
                v,
                torch.Tensor
            )
            else v

            for k,v in inputs.items()
        }



        # =========================
        # 5. Generate
        # =========================

        max_new_tokens = (
            generation_kwargs.get(
                "max_new_tokens",
                self.max_new_tokens
            )
        )



        with torch.inference_mode():

            output_ids = (
                self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                )
            )



        # print(
        #     "input length:",
        #     inputs["input_ids"].shape[1]
        # )

        # print(
        #     "output length:",
        #     output_ids.shape[1]
        # )



        new_token_ids = output_ids[:, inputs["input_ids"].shape[1]:]
        response = self.processor.tokenizer.batch_decode(
            new_token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return response.strip()
