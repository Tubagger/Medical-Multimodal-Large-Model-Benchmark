from ours.utils.registry import registry
from ours.models.ernie_chat import ErnieChat
from ours.models.openai_chat import OpenAIChat
from ours.models.qwen_chat import QwenChat
from ours.models.qwen3_vl import Qwen3vl
from ours.models.deepseek_chat import DeepseekChat
from ours.models.lingshu import Lingshu
from ours.models.base import BaseChat, Response
from typing import List

def load_chatmodel(model_id:str, device:str="cuda:0") -> 'BaseChat':
    return registry.get_chatmodel_class(model_id)(model_id=model_id, device=device)


def model_zoo() -> List['BaseChat']:
    return registry.list_chatmodels()