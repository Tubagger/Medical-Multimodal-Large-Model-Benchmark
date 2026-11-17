# import os
# import torch
# import yaml
# from typing import List, Dict, Any

# from ours.models.base import BaseChat
# from ours.utils.registry import registry
# from ours.utils.utils import get_abs_path


# @registry.register_chatmodel()
# class HuluMed(BaseChat):

#     MODEL_CONFIG = {
#         "hulu-med-7b": "configs/models/hulumed/hulu-med-7b.yaml",
#         "hulu-med-32b": "configs/models/hulumed/hulu-med-32b.yaml",
#     }

#     model_family = list(MODEL_CONFIG.keys())
#     model_arch = "hulu-med"


#     def __init__(
#         self,
#         model_id: str = "hulu-med-32b",
#         **kwargs
#     ):

#         super().__init__(
#             model_id=model_id
#         )

#         from transformers import (
#             AutoModelForCausalLM,
#             AutoProcessor
#         )


#         # =========================
#         # load config
#         # =========================

#         config_path = self.MODEL_CONFIG[
#             self.model_id
#         ]

#         with open(
#             get_abs_path(config_path)
#         ) as f:

#             self.model_config = yaml.load(
#                 f,
#                 Loader=yaml.FullLoader
#             )


#         self.model_path = self.model_config.get(
#             "model_path",
#             "ZJU-AI4H/Hulu-Med-32B"
#         )


#         # =========================
#         # load model
#         # =========================

#         self.model = AutoModelForCausalLM.from_pretrained(
#             self.model_path,
#             trust_remote_code=True,
#             torch_dtype=torch.bfloat16,
#             device_map="auto",
#             attn_implementation="eager",
#         ).eval()



#         # =========================
#         # processor
#         # =========================

#         self.processor = AutoProcessor.from_pretrained(
#             self.model_path,
#             trust_remote_code=True
#         )


#         self.tokenizer = (
#             self.processor.tokenizer
#         )



#     def chat(
#         self,
#         messages: List[Dict],
#         **generation_kwargs
#     ):


#         # =========================
#         # 1. convert conversation
#         # =========================

#         conversation = []


#         for msg in messages:

#             role = msg["role"]


#             if isinstance(
#                 msg["content"],
#                 dict
#             ):

#                 content = [

#                     {
#                         "type": "image",
#                         "image": {
#                             "image_path":
#                             msg["content"]["image_path"]
#                         }
#                     },

#                     {
#                         "type": "text",
#                         "text":
#                         msg["content"]["text"]
#                     }

#                 ]


#             else:

#                 content = [

#                     {
#                         "type":"text",
#                         "text":msg["content"]
#                     }

#                 ]



#             conversation.append(
#                 {
#                     "role": role,
#                     "content": content
#                 }
#             )



#         # =========================
#         # 2. processor
#         # =========================

#         inputs = self.processor(
#             conversation=conversation,
#             add_system_prompt=True,
#             add_generation_prompt=True,
#             return_tensors="pt"
#         )



#         # =========================
#         # move input device
#         # =========================

#         device = next(
#             self.model.parameters()
#         ).device


#         inputs = {

#             k:
#             v.to(device)
#             if isinstance(v, torch.Tensor)
#             else v

#             for k,v in inputs.items()

#         }



#         # image dtype

#         if "pixel_values" in inputs:

#             inputs["pixel_values"] = (
#                 inputs["pixel_values"]
#                 .to(self.model.dtype)
#             )



#         # =========================
#         # 3. generation
#         # =========================

#         default_generation_kwargs = {

#             "do_sample": False,

#             "max_new_tokens": 512,

#         }


#         default_generation_kwargs.update(
#             generation_kwargs
#         )


#         with torch.no_grad():

#             output_ids = self.model.generate(

#                 **inputs,

#                 **default_generation_kwargs

#             )



#         # =========================
#         # 4. decode
#         # =========================

#         outputs = self.processor.batch_decode(

#             output_ids,

#             skip_special_tokens=True,

#             use_think=False

#         )[0].strip()



#         return outputs


import os
import math
import tempfile
import uuid

import torch
import yaml

from PIL import Image, ImageOps
from typing import List, Dict, Any

from ours.models.base import BaseChat
from ours.utils.registry import registry
from ours.utils.utils import get_abs_path


@registry.register_chatmodel()
class HuluMed(BaseChat):

    MODEL_CONFIG = {
        "hulu-med-7b": "configs/models/hulumed/hulu-med-7b.yaml",
        "hulu-med-32b": "configs/models/hulumed/hulu-med-32b.yaml",
    }

    model_family = list(MODEL_CONFIG.keys())
    model_arch = "hulu-med"

    def __init__(
        self,
        model_id: str = "hulu-med-32b",
        **kwargs
    ):

        super().__init__(
            model_id=model_id
        )

        from transformers import (
            AutoModelForCausalLM,
            AutoProcessor
        )

        # =========================
        # 1. 加载配置
        # =========================

        config_path = self.MODEL_CONFIG[
            self.model_id
        ]

        with open(
            get_abs_path(config_path)
        ) as f:

            self.model_config = yaml.load(
                f,
                Loader=yaml.FullLoader
            )

        self.model_path = self.model_config.get(
            "model_path",
            "ZJU-AI4H/Hulu-Med-32B"
        )

        # =========================
        # 2. 图片分辨率限制
        # =========================

        # 最大约 60 万像素。
        # 对正方形图片大约相当于 776 × 776。
        self.max_image_pixels = self.model_config.get(
            "max_image_pixels",
            768 * 28 * 28
        )

        # 是否打印缩放前后的图片尺寸
        self.print_resize_info = self.model_config.get(
            "print_resize_info",
            False
        )

        # =========================
        # 3. 加载模型
        # =========================

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="eager",
        ).eval()

        # =========================
        # 4. 加载 Processor
        # =========================

        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )

        self.tokenizer = self.processor.tokenizer

    def _resize_image(
        self,
        image_path: str,
        temp_dir: str
    ) -> str:
        """
        当图片总像素数超过 max_image_pixels 时，
        保持宽高比进行缩放，并保存到临时目录。

        未超过限制时直接返回原始路径。
        """

        if not isinstance(image_path, str):
            raise TypeError(
                "image_path 必须是字符串路径，"
                f"当前类型为：{type(image_path)}"
            )

        if not os.path.isfile(image_path):
            raise FileNotFoundError(
                f"图片文件不存在：{image_path}"
            )

        try:
            with Image.open(image_path) as image:

                # 根据 EXIF 信息纠正手机照片旋转方向
                image = ImageOps.exif_transpose(image)

                original_width, original_height = image.size
                original_pixels = (
                    original_width * original_height
                )

                # 图片没有超过限制，直接使用原图
                if original_pixels <= self.max_image_pixels:

                    # if self.print_resize_info:
                    #     print(
                    #         "[HuluMed image] "
                    #         f"保持原尺寸：{image_path} | "
                    #         f"{original_width}×{original_height}"
                    #     )

                    return image_path

                # 根据最大总像素数计算等比例缩放比例
                scale = math.sqrt(
                    self.max_image_pixels
                    / original_pixels
                )

                new_width = max(
                    1,
                    int(original_width * scale)
                )

                new_height = max(
                    1,
                    int(original_height * scale)
                )

                # 确保缩放后的总像素不超过限制
                while (
                    new_width * new_height
                    > self.max_image_pixels
                ):
                    if new_width >= new_height:
                        new_width -= 1
                    else:
                        new_height -= 1

                # Pillow 新旧版本兼容
                if hasattr(Image, "Resampling"):
                    resample_method = (
                        Image.Resampling.LANCZOS
                    )
                else:
                    resample_method = Image.LANCZOS

                resized_image = image.resize(
                    (new_width, new_height),
                    resample=resample_method
                )

                # 大多数视觉模型最终需要三通道图片
                if resized_image.mode != "RGB":
                    resized_image = resized_image.convert(
                        "RGB"
                    )

                resized_path = os.path.join(
                    temp_dir,
                    f"{uuid.uuid4().hex}_resized.jpg"
                )

                resized_image.save(
                    resized_path,
                    format="JPEG",
                    quality=95,
                    subsampling=0
                )

                # if self.print_resize_info:
                #     print(
                #         "[HuluMed image] "
                #         f"{image_path}: "
                #         f"{original_width}×{original_height} "
                #         f"→ {new_width}×{new_height}"
                #     )

                return resized_path

        except Exception as error:
            raise RuntimeError(
                f"处理图片失败：{image_path}"
            ) from error

    def chat(
        self,
        messages: List[Dict],
        **generation_kwargs
    ):

        # 临时目录会在本次推理结束后自动删除
        with tempfile.TemporaryDirectory(
            prefix="hulumed_resized_"
        ) as temp_dir:

            # =========================
            # 1. 转换 conversation
            # =========================

            conversation = []

            for msg in messages:

                role = msg["role"]

                if isinstance(
                    msg["content"],
                    dict
                ):

                    original_image_path = (
                        msg["content"]["image_path"]
                    )

                    # 在送入 Processor 前缩放图片
                    resized_image_path = (
                        self._resize_image(
                            image_path=original_image_path,
                            temp_dir=temp_dir
                        )
                    )

                    content = [
                        {
                            "type": "image",
                            "image": {
                                "image_path":
                                resized_image_path
                            }
                        },
                        {
                            "type": "text",
                            "text":
                            msg["content"]["text"]
                        }
                    ]

                else:

                    content = [
                        {
                            "type": "text",
                            "text": msg["content"]
                        }
                    ]

                conversation.append(
                    {
                        "role": role,
                        "content": content
                    }
                )

            # =========================
            # 2. Processor
            # =========================

            inputs = self.processor(
                conversation=conversation,
                add_system_prompt=True,
                add_generation_prompt=True,
                return_tensors="pt"
            )

            # =========================
            # 3. 移动输入到模型入口设备
            # =========================

            device = next(
                self.model.parameters()
            ).device

            inputs = {
                key: (
                    value.to(device)
                    if isinstance(value, torch.Tensor)
                    else value
                )
                for key, value in inputs.items()
            }

            # 图片张量转换为模型 dtype
            if "pixel_values" in inputs:
                inputs["pixel_values"] = (
                    inputs["pixel_values"]
                    .to(self.model.dtype)
                )

            # =========================
            # 4. Generation
            # =========================

            default_generation_kwargs = {
                "do_sample": False,
                "max_new_tokens": 512,
            }

            default_generation_kwargs.update(
                generation_kwargs
            )

            with torch.no_grad():

                output_ids = self.model.generate(
                    **inputs,
                    **default_generation_kwargs
                )

            # =========================
            # 5. Decode
            # =========================

            outputs = self.processor.batch_decode(
                output_ids,
                skip_special_tokens=True,
                use_think=False
            )[0].strip()

            return outputs