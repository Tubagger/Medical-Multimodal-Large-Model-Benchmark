from typing import List, Dict, Any
import yaml
from ours.utils.registry import registry
from ours.models.base import BaseChat, Response
from ours.utils.utils import get_abs_path
from openai import OpenAI
import os
import io
import base64
import time
from PIL import Image


@registry.register_chatmodel()
class QwenChat(BaseChat):
    """
    Chat class for Qwen multimodal models
    """
    
    MODEL_CONFIG = {
        "qwen3-32b":"configs/models/qwen/qwen3-32b.yaml"
    }
    
    model_family = list(MODEL_CONFIG.keys())
    model_arch = "qwen"

    def __init__(self, model_id: str = "qwen2-vl", **kwargs):
        super().__init__(model_id=model_id)
        config_path = self.MODEL_CONFIG[self.model_id]
        with open(get_abs_path(config_path)) as f:
            self.model_config = yaml.load(f, Loader=yaml.FullLoader)

        # 获取API Key
        api_key = os.getenv("qwen_apikey", "")
        assert api_key, "qwen_apikey is empty"
        self.api_key = api_key

        # 参数
        self.max_retries = self.model_config.get("max_retries", 10)
        self.timeout = self.model_config.get("timeout", 2)
        
        # 初始化OpenAI兼容接口客户端（阿里Qwen使用同样协议）
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.model_config.get("base_url", "https://qianfan.baidubce.com/v2")
        )

    def chat(self, messages: List[Dict[str, Any]], **generation_kwargs):
        """
        与模型对话，支持图片+文本输入。
        """
        conversation = []
        for message in messages:
            if message["role"] not in ["system", "user", "assistant"]:
                raise ValueError("Unsupported role. Only system, user, assistant supported.")
            
            if isinstance(message["content"], dict):
                text = message["content"].get("text", "")
                image_path = message["content"].get("image_path", "")
                local_image = os.path.exists(image_path)

                content = [
                    {"type": "text",
                     "text": text},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{self.encode_image(image_path)}" if local_image else image_path
                        }
                    }
                ]
            else:
                content = message["content"]

            conversation.append({"role": message["role"], "content": content})

        # 构造请求体
        raw_request = {
            "model": self.model_id,
            "messages": conversation,
            "temperature": generation_kwargs.get("temperature", 1.0),
            "max_tokens": generation_kwargs.get("max_new_tokens", 512),
            "n": generation_kwargs.get("num_return_sequences", 1)
        }

        if "stop_sequences" in generation_kwargs:
            raw_request["stop"] = generation_kwargs["stop_sequences"]
        if not generation_kwargs.get("do_sample", True):
            raw_request["temperature"] = 0.0

        # 请求发送 + 自动重试
        for i in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(**raw_request)
                break
            except Exception as e:
                print(f"[Retry {i+1}/{self.max_retries}] Error: {e}")
                time.sleep(self.timeout)
                response = None

        if response is None:
            return Response(self.model_id, "Error: failed to generate response", None, None)
        
        # 解析结果
        response_message = response.choices[0].message.content
        finish_reason = response.choices[0].finish_reason
        logprobs = response.choices[0].logprobs

        return Response(self.model_id, response_message, logprobs, finish_reason)

    @classmethod
    def encode_image(cls, image_path: str):
        """
        将图像转为base64编码,可选压缩至400px。
        """
        buffer = io.BytesIO()
        with open(image_path, "rb") as image_file:
            img = Image.open(image_file).convert("RGB")

            # 限制图像尺寸
            max_size = 400
            if img.width > max_size or img.height > max_size:
                ratio = min(max_size / img.width, max_size / img.height)
                new_size = (int(img.width * ratio), int(img.height * ratio))
                img = img.resize(new_size, Image.LANCZOS)

            img.save(buffer, format="JPEG")
            encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return encoded
