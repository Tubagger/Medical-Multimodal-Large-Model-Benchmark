
# import yaml
# import torch
# from typing import List, Dict



# from ours.models.base import BaseChat
# from ours.utils.registry import registry
# from ours.utils.utils import get_abs_path


# IMAGENET_MEAN = (0.485, 0.456, 0.406)
# IMAGENET_STD = (0.229, 0.224, 0.225)

    
# def build_transform(input_size):
#     import torchvision.transforms as T
#     from torchvision.transforms.functional import InterpolationMode
#     MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
#     transform = T.Compose([
#         T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
#         T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
#         T.ToTensor(),
#         T.Normalize(mean=MEAN, std=STD)
#     ])
#     return transform
# def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
#     best_ratio_diff = float('inf')
#     best_ratio = (1, 1)
#     area = width * height
#     for ratio in target_ratios:
#         target_aspect_ratio = ratio[0] / ratio[1]
#         ratio_diff = abs(aspect_ratio - target_aspect_ratio)
#         if ratio_diff < best_ratio_diff:
#             best_ratio_diff = ratio_diff
#             best_ratio = ratio
#         elif ratio_diff == best_ratio_diff:
#             if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
#                 best_ratio = ratio
#     return best_ratio

# def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
#     orig_width, orig_height = image.size
#     aspect_ratio = orig_width / orig_height

#     # calculate the existing image aspect ratio
#     target_ratios = set(
#         (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
#         i * j <= max_num and i * j >= min_num)
#     target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

#     # find the closest aspect ratio to the target
#     target_aspect_ratio = find_closest_aspect_ratio(
#         aspect_ratio, target_ratios, orig_width, orig_height, image_size)

#     # calculate the target width and height
#     target_width = image_size * target_aspect_ratio[0]
#     target_height = image_size * target_aspect_ratio[1]
#     blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

#     # resize the image
#     resized_img = image.resize((target_width, target_height))
#     processed_images = []
#     for i in range(blocks):
#         box = (
#             (i % (target_width // image_size)) * image_size,
#             (i // (target_width // image_size)) * image_size,
#             ((i % (target_width // image_size)) + 1) * image_size,
#             ((i // (target_width // image_size)) + 1) * image_size
#         )
#         # split the image
#         split_img = resized_img.crop(box)
#         processed_images.append(split_img)
#     assert len(processed_images) == blocks
#     if use_thumbnail and len(processed_images) != 1:
#         thumbnail_img = image.resize((image_size, image_size))
#         processed_images.append(thumbnail_img)
#     return processed_images
# def load_image(image_file, input_size=448, max_num=12):
#     from PIL import Image
#     image = Image.open(image_file).convert('RGB')
#     transform = build_transform(input_size=input_size)
#     images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
#     pixel_values = [transform(image) for image in images]
#     pixel_values = torch.stack(pixel_values)
#     return pixel_values
# @registry.register_chatmodel()
# class InternVL3_5(BaseChat):

#     MODEL_CONFIG = {
#         "internvl3.5-14b": "configs/models/internvl/internvl3.5-14b.yaml",
#         "internvl3.5-38b": "configs/models/internvl/internvl3.5-38b.yaml",
#     }

#     model_family = list(MODEL_CONFIG.keys())
#     model_arch = "internvl"



#     def __init__(self, model_id="internvl3.5-14b", **kwargs):
#         super().__init__(model_id=model_id)
#         from transformers import AutoModel, AutoTokenizer
#         config_path = self.MODEL_CONFIG[self.model_id]

#         with open(get_abs_path(config_path)) as f:
#             self.model_config = yaml.load(f, Loader=yaml.FullLoader)

#         self.model_path = self.model_config["model_path"]
#         self.max_num = self.model_config.get("max_num", 12)

#         self.model = AutoModel.from_pretrained(
#             self.model_path,
#             torch_dtype=torch.bfloat16,
#             low_cpu_mem_usage=True,
#             trust_remote_code=True,
#             device_map="auto",
#             use_flash_attn=False,
#         ).eval()

#         self.tokenizer = AutoTokenizer.from_pretrained(
#             self.model_path,
#             trust_remote_code=True,
#             use_fast=False,
#         )

#     def chat(self, messages: List[Dict], **generation_kwargs):

#         msg = messages[-1]
#         content = msg["content"]

#         if isinstance(content, dict):

#             image_path = content["image_path"]
#             question = content["text"]

#             pixel_values = load_image(
#                 image_path,
#                 max_num=self.max_num,
#             )

#             pixel_values = pixel_values.to(
#                 dtype=torch.bfloat16,
#                 device=self.model.device,
#             )

#             question = "<image>\n" + question

#         else:

#             pixel_values = None
#             question = content

#         generation_config = dict(
#             max_new_tokens=generation_kwargs.get(
#                 "max_new_tokens",
#                 512,
#             ),
#             temperature=generation_kwargs.get(
#                 "temperature",
#                 0,
#             ),
#             do_sample=generation_kwargs.get(
#                 "temperature",
#                 0,
#             ) > 0,
#         )

#         response = self.model.chat(
#             tokenizer=self.tokenizer,
#             pixel_values=pixel_values,
#             question=question,
#             generation_config=generation_config,
#         )

#         if isinstance(response, tuple):
#             response = response[0]

#         return response

import yaml
import torch

from typing import List, Dict
from PIL import Image

from ours.models.base import BaseChat
from ours.utils.registry import registry
from ours.utils.utils import get_abs_path


IMAGENET_MEAN = (
    0.485,
    0.456,
    0.406,
)

IMAGENET_STD = (
    0.229,
    0.224,
    0.225,
)


def build_transform(input_size: int = 448):
    """
    将每个图像块处理为固定尺寸，并进行 ImageNet 标准化。
    """

    import torchvision.transforms as T

    from torchvision.transforms.functional import (
        InterpolationMode,
    )

    transform = T.Compose([
        T.Lambda(
            lambda image: (
                image.convert("RGB")
                if image.mode != "RGB"
                else image
            )
        ),
        T.Resize(
            (input_size, input_size),
            interpolation=InterpolationMode.BICUBIC,
        ),
        T.ToTensor(),
        T.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
        ),
    ])

    return transform


def find_closest_aspect_ratio(
    aspect_ratio,
    target_ratios,
    width,
    height,
    image_size,
):
    """
    在允许的网格比例中，寻找最接近原图宽高比的比例。
    """

    best_ratio_diff = float("inf")
    best_ratio = (1, 1)

    area = width * height

    for ratio in target_ratios:
        target_aspect_ratio = (
            ratio[0] / ratio[1]
        )

        ratio_diff = abs(
            aspect_ratio - target_aspect_ratio
        )

        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio

        elif ratio_diff == best_ratio_diff:
            target_area = (
                0.5
                * image_size
                * image_size
                * ratio[0]
                * ratio[1]
            )

            if area > target_area:
                best_ratio = ratio

    return best_ratio


def dynamic_preprocess(
    image: Image.Image,
    min_num: int = 1,
    max_num: int = 4,
    image_size: int = 448,
    use_thumbnail: bool = True,
):
    """
    将原图动态缩放，并切分成多个 image_size × image_size 图像块。

    max_num:
        最多生成多少个局部图像块。

    use_thumbnail:
        当局部图像块数量大于 1 时，
        是否额外加入一张完整图片缩略图。

    例如：
        max_num=4 且 use_thumbnail=True
        最多生成 4 个局部块 + 1 个缩略图。
    """

    original_width, original_height = image.size

    if original_width <= 0 or original_height <= 0:
        raise ValueError(
            f"无效的图片尺寸：{image.size}"
        )

    aspect_ratio = (
        original_width / original_height
    )

    # 构造所有允许的图像网格比例。
    #
    # 例如 max_num=4 时可能包含：
    # 1×1、1×2、2×1、1×3、3×1、2×2 等。
    target_ratios = {
        (width_blocks, height_blocks)
        for number_blocks in range(
            min_num,
            max_num + 1,
        )
        for width_blocks in range(
            1,
            number_blocks + 1,
        )
        for height_blocks in range(
            1,
            number_blocks + 1,
        )
        if (
            min_num
            <= width_blocks * height_blocks
            <= max_num
        )
    }

    target_ratios = sorted(
        target_ratios,
        key=lambda ratio: (
            ratio[0] * ratio[1]
        ),
    )

    target_aspect_ratio = (
        find_closest_aspect_ratio(
            aspect_ratio=aspect_ratio,
            target_ratios=target_ratios,
            width=original_width,
            height=original_height,
            image_size=image_size,
        )
    )

    width_blocks = target_aspect_ratio[0]
    height_blocks = target_aspect_ratio[1]

    target_width = (
        image_size * width_blocks
    )

    target_height = (
        image_size * height_blocks
    )

    number_blocks = (
        width_blocks * height_blocks
    )

    if hasattr(Image, "Resampling"):
        resize_method = (
            Image.Resampling.BICUBIC
        )
    else:
        resize_method = Image.BICUBIC

    # 将原图缩放到目标网格尺寸。
    resized_image = image.resize(
        (target_width, target_height),
        resample=resize_method,
    )

    processed_images = []

    for index in range(number_blocks):
        column = index % width_blocks
        row = index // width_blocks

        left = column * image_size
        upper = row * image_size
        right = left + image_size
        lower = upper + image_size

        split_image = resized_image.crop(
            (
                left,
                upper,
                right,
                lower,
            )
        )

        processed_images.append(
            split_image
        )

    if len(processed_images) != number_blocks:
        raise RuntimeError(
            "图片切块数量异常："
            f"expected={number_blocks}, "
            f"actual={len(processed_images)}"
        )

    # 当图片被切成多个局部块时，
    # 额外加入一张完整图片缩略图。
    if (
        use_thumbnail
        and len(processed_images) > 1
    ):
        thumbnail_image = image.resize(
            (image_size, image_size),
            resample=resize_method,
        )

        processed_images.append(
            thumbnail_image
        )

    return processed_images


def load_image(
    image_file: str,
    input_size: int = 448,
    max_num: int = 4,
    use_thumbnail: bool = True,
    print_image_info: bool = False,
):
    """
    加载并预处理一张图片。

    返回：
        pixel_values:
        [num_tiles, 3, input_size, input_size]
    """

    with Image.open(image_file) as image:
        image = image.convert("RGB")
        original_size = image.size

        transform = build_transform(
            input_size=input_size
        )

        processed_images = dynamic_preprocess(
            image=image,
            min_num=1,
            max_num=max_num,
            image_size=input_size,
            use_thumbnail=use_thumbnail,
        )

        pixel_values = [
            transform(processed_image)
            for processed_image
            in processed_images
        ]

        pixel_values = torch.stack(
            pixel_values
        )

    if print_image_info:
        print(
            "[InternVL image processing]"
        )

        print(
            "Original image size:",
            f"{original_size[0]}×{original_size[1]}",
        )

        print(
            "Tile size:",
            f"{input_size}×{input_size}",
        )

        print(
            "Number of image tiles:",
            pixel_values.shape[0],
        )

        print(
            "pixel_values shape:",
            tuple(pixel_values.shape),
        )

    return pixel_values


@registry.register_chatmodel()
class InternVL3_5(BaseChat):

    MODEL_CONFIG = {
        "internvl3.5-14b":
            "configs/models/internvl/internvl3.5-14b.yaml",

        "internvl3.5-38b":
            "configs/models/internvl/internvl3.5-38b.yaml",
    }

    model_family = list(
        MODEL_CONFIG.keys()
    )

    model_arch = "internvl"

    def __init__(
        self,
        model_id: str = "internvl3.5-14b",
        **kwargs,
    ):
        super().__init__(
            model_id=model_id
        )

        from transformers import (
            AutoModel,
            AutoTokenizer,
        )

        config_path = self.MODEL_CONFIG[
            self.model_id
        ]

        with open(
            get_abs_path(config_path)
        ) as file:
            self.model_config = yaml.load(
                file,
                Loader=yaml.FullLoader,
            )

        self.model_path = (
            self.model_config["model_path"]
        )

        # =========================
        # 图片预处理参数
        # =========================

        # InternVL 通常使用 448×448 的图像块。
        self.input_size = (
            self.model_config.get(
                "input_size",
                448,
            )
        )

        # 最大局部图像块数量。
        #
        # 原代码默认为 12，
        # 这里默认改成 4，以降低视觉 token 和显存。
        self.max_num = (
            self.model_config.get(
                "max_num",
                4,
            )
        )

        # 当存在多个局部块时，
        # 是否额外增加一张完整图片缩略图。
        self.use_thumbnail = (
            self.model_config.get(
                "use_thumbnail",
                True,
            )
        )

        # 是否打印图片切块信息。
        self.print_image_info = (
            self.model_config.get(
                "print_image_info",
                False,
            )
        )

        if self.max_num < 1:
            raise ValueError(
                "max_num 必须大于或等于 1，"
                f"当前值为：{self.max_num}"
            )

        # =========================
        # 加载模型
        # =========================

        device_map = self.model_config.get(
            "device_map",
            "auto",
        )
        required_cuda_devices = int(
            self.model_config.get(
                "required_cuda_devices",
                1,
            )
        )
        available_cuda_devices = torch.cuda.device_count()
        if available_cuda_devices < required_cuda_devices:
            raise RuntimeError(
                f"{self.model_id} requires at least {required_cuda_devices} CUDA devices, "
                f"but only {available_cuda_devices} are visible"
            )

        max_memory = None
        max_memory_per_gpu = self.model_config.get(
            "max_memory_per_gpu"
        )
        if max_memory_per_gpu:
            max_memory = {
                index: max_memory_per_gpu
                for index in range(required_cuda_devices)
            }

        print(
            f"Loading {self.model_id} with device_map={device_map}, "
            f"max_memory={max_memory}",
            flush=True,
        )

        self.model = AutoModel.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map=device_map,
            max_memory=max_memory,
            use_flash_attn=False,
        ).eval()

        # =========================
        # 加载 tokenizer
        # =========================

        self.tokenizer = (
            AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                use_fast=False,
            )
        )

    def chat(
        self,
        messages: List[Dict],
        **generation_kwargs,
    ):
        # 当前实现保持你原来的逻辑：
        # 只处理 messages 中最后一条消息。
        message = messages[-1]
        content = message["content"]

        if isinstance(content, dict):
            image_path = content[
                "image_path"
            ]

            question = content[
                "text"
            ]

            pixel_values = load_image(
                image_file=image_path,
                input_size=self.input_size,
                max_num=self.max_num,
                use_thumbnail=self.use_thumbnail,
                print_image_info=(
                    self.print_image_info
                ),
            )

            pixel_values = pixel_values.to(
                dtype=torch.bfloat16,
                device=self.model.device,
            )

            question = (
                "<image>\n" + question
            )

        else:
            pixel_values = None
            question = content

        temperature = generation_kwargs.get(
            "temperature",
            0,
        )

        generation_config = {
            "max_new_tokens":
                generation_kwargs.get(
                    "max_new_tokens",
                    512,
                ),

            "temperature":
                temperature,

            "do_sample":
                temperature > 0,
        }

        if generation_kwargs.get('bad_words_ids'):
            generation_config['bad_words_ids'] = generation_kwargs['bad_words_ids']

        with torch.inference_mode():
            response = self.model.chat(
                tokenizer=self.tokenizer,
                pixel_values=pixel_values,
                question=question,
                generation_config=(
                    generation_config
                ),
            )

        if isinstance(response, tuple):
            response = response[0]

        return response
