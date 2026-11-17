from ours.utils.registry import registry

from ours.models.openai_chat import OpenAIChat
from ours.models.claude_chat import ClaudeChat
from ours.models.gemmni_chat import GemminiChat
from ours.models.deepseek_chat import DeepseekChat

from ours.models.qwen3_vl import Qwen3vl
from ours.models.deepseek_vl2 import DeepSeekVL2
from ours.models.lingshu import Lingshu
from ours.models.med_gemma import MedGemma
from ours.models.hulumed import HuluMed
from ours.models.huatuogpt_v import HuatuoGPTVision
from ours.models.healthgpt_pro import HealthGPTPro
from ours.models.internvl3_5 import InternVL3_5
from ours.models.glm_4v import GLM4V
from ours.models.llava_med import LlavaMed

from ours.models.base import BaseChat, Response
from typing import List

def load_chatmodel(model_id:str, device:str="cuda:0") -> 'BaseChat':

    return registry.get_chatmodel_class(model_id)(model_id=model_id, device=device)


def model_zoo() -> List['BaseChat']:
    return registry.list_chatmodels()
