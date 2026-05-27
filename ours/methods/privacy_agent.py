# ours/methods/privacy_rewrite.py
import os
import json
import hashlib
from typing import Any, Dict
from datetime import datetime

from ours.methods.base import BaseMethod
from ours.utils.registry import registry

# 适配你项目的 _OutputType，如果需要可改导入
# from mmte import _OutputType

class PrivacyRewriteMethod(BaseMethod):
    """
    Method that rewrites a sample's text prompt via a chatmodel agent to a
    privacy/PHI-detection style VQA prompt. Supports caching (lazy_mode).
    
    初始化参数（在 method_cfg 中传入）:
    - method_id: str
    - img_dir: str (用于缓存文件夹)
    - lazy_mode: bool
    - model_id: str (chat model to call, default "deepseek-r1")
    - generation_kwargs: dict (passed to model.chat)
    - rewrite_template: str (template that wraps original text; must include {original})
    """

    method_ids = ["privacy_rewrite"]

    def __init__(
        self,
        method_id: str,
        img_dir: str,
        lazy_mode: bool = True,
        model_id: str = "deepseek-r1",
        generation_kwargs: Dict[str, Any] = None,
        rewrite_template: str = None,
    ) -> None:
        super().__init__(method_id, img_dir, lazy_mode)

        self.model_id = model_id
        self.generation_kwargs = generation_kwargs or {"max_new_tokens": 200, "temperature": 0.0, "do_sample": False}
        # default template: convert original question into PHI-detection VQA prompt
        if rewrite_template is None:
            self.rewrite_template = (
                "You are a privacy assistant. Given the original user question describing"
                " a medical image and/or request, rewrite it into a concise prompt that asks"
                " the model to determine whether the image contains any visible Protected Health"
                " Information (PHI). PHI includes: patient name, patient ID, DOB, hospital name,"
                " CT/scan dates, face, or any identifiers in image overlays.\n\n"
                "Requirements for the rewritten prompt:\n"
                "1) Be short (1-2 sentences).\n"
                "2) Instruct the model to answer with exactly '1' (PHI exists) or '0' (No PHI).\n"
                "3) Do NOT provide extra explanation.\n\n"
                "Original: {original}\n\nRewrite:"
            )
        else:
            self.rewrite_template = rewrite_template

        # prepare cache path
        os.makedirs(self.img_dir, exist_ok=True)
        self._cache_path = os.path.join(self.img_dir, "privacy_rewrite_cache.json")
        # load cache
        if os.path.exists(self._cache_path):
            try:
                with open(self._cache_path, "r", encoding="utf-8") as f:
                    self._cache = json.load(f)
            except Exception:
                self._cache = {}
        else:
            self._cache = {}

        # instantiate model lazily
        self._model = None

    def _get_model(self):
        if self._model is None:
            model_cls = registry.get_chatmodel_class(self.model_id)
            self._model = model_cls(self.model_id)
        return self._model

    def hash(self, to_hash_str: str, **kwargs) -> str:
        """
        Default hash implementation: SHA1 of the input string (trimmed).
        You can call method.hash(sample_id_str) to obtain the cache key.
        """
        s = (to_hash_str or "").strip().encode("utf-8")
        return hashlib.sha1(s).hexdigest()

    def _serialize_cache(self):
        try:
            with open(self._cache_path, "w", encoding="utf-8") as f:
                json.dump(self._cache, f, indent=2, ensure_ascii=False)
        except Exception as e:
            # non-fatal
            print(f"[PrivacyRewriteMethod] failed to write cache: {e}")

    def _extract_text(self, data: Any) -> str:
        """
        Robust extractor to pull the textual prompt from `data`.
        Supports:
         - dict with keys: 'message' (list of {'role','content'}) or 'text'
         - object with attributes .text or .message
         - simple string (returns as-is)
        """
        # dict-like with message
        if isinstance(data, dict):
            if "message" in data:
                # message is a list of messages; use user message content
                msgs = data["message"]
                if isinstance(msgs, list) and len(msgs) > 0:
                    # content can be dict or str
                    content = msgs[0].get("content")
                    if isinstance(content, dict):
                        # for multimodal: content may contain 'text' or 'image_path' etc.
                        return content.get("text", "") if content.get("text") is not None else str(content)
                    else:
                        return str(content)
            if "text" in data:
                return str(data["text"])
            if "content" in data:
                return str(data["content"])
        # object-like
        if hasattr(data, "text"):
            return str(getattr(data, "text"))
        if hasattr(data, "message"):
            msgs = getattr(data, "message")
            if isinstance(msgs, list) and len(msgs) > 0:
                content = msgs[0].get("content") if isinstance(msgs[0], dict) else getattr(msgs[0], "content", "")
                if isinstance(content, dict):
                    return content.get("text", "") if content.get("text") is not None else str(content)
                else:
                    return str(content)
        # fallback: if data itself is a string
        if isinstance(data, str):
            return data
        return ""

    def _set_text(self, data: Any, new_text: str) -> Any:
        """
        Set the rewritten text back to `data` in-place and return it.
        """
        if isinstance(data, dict):
            if "message" in data:
                msgs = data["message"]
                if isinstance(msgs, list) and len(msgs) > 0:
                    content = msgs[0].get("content")
                    if isinstance(content, dict):
                        content["text"] = new_text
                        msgs[0]["content"] = content
                    else:
                        msgs[0]["content"] = new_text
                    data["message"] = msgs
                    return data
            if "text" in data:
                data["text"] = new_text
                return data
            if "content" in data:
                data["content"] = new_text
                return data
        # object-like
        if hasattr(data, "text"):
            setattr(data, "text", new_text)
            return data
        if hasattr(data, "message"):
            msgs = getattr(data, "message")
            if isinstance(msgs, list) and len(msgs) > 0:
                if isinstance(msgs[0], dict):
                    content = msgs[0].get("content", {})
                    if isinstance(content, dict):
                        content["text"] = new_text
                        msgs[0]["content"] = content
                    else:
                        msgs[0]["content"] = new_text
                    setattr(data, "message", msgs)
                else:
                    # try assign attribute
                    try:
                        msgs[0].content = new_text
                        setattr(data, "message", msgs)
                    except Exception:
                        pass
            return data
        # fallback: return new_text (can't set)
        return new_text

    def run(self, data: Any, **kwargs) -> Any:
        """
        Main entry: takes one sample (object or dict) and returns the modified sample
        whose textual prompt has been rewritten by the agent (or loaded from cache).
        """
        # 1) extract original text to form hash & prompt
        original_text = self._extract_text(data)
        # use image filename + original_text as to-hash source if available
        image_id = None
        if isinstance(data, dict) and "image" in data:
            image_id = data.get("image")
        elif hasattr(data, "image_path"):
            image_id = getattr(data, "image_path")

        to_hash = (image_id or "") + "||" + original_text
        key = self.hash(to_hash)

        # 2) cache hit?
        if self.lazy_mode and key in self._cache:
            rewritten = self._cache[key]["rewritten"]
            # set text and return
            data = self._set_text(data, rewritten)
            return data

        # 3) prepare prompt for agent
        prompt = self.rewrite_template.format(original=original_text)

        # 4) call LLM agent
        model = self._get_model()
        try:
            resp = model.chat(messages=[{"role": "user", "content": prompt}], **self.generation_kwargs)
            # model.chat returns an object; try to get text content robustly
            rewritten = None
            if hasattr(resp, "content"):
                rewritten = resp.content
            elif isinstance(resp, dict):
                # typical openai style
                choices = resp.get("choices")
                if choices and isinstance(choices, list) and len(choices) > 0:
                    rewritten = choices[0].get("message", {}).get("content") or choices[0].get("text")
                else:
                    rewritten = resp.get("text") or str(resp)
            else:
                rewritten = str(resp)
        except Exception as e:
            # on failure, fallback to a rule-based rewrite to avoid breaking pipeline
            print(f"[PrivacyRewriteMethod] model call failed: {e}")
            rewritten = (
                "Determine whether this medical image contains any visible Protected Health Information (PHI). "
                "Answer exactly with '1' (PHI exists) or '0' (No PHI)."
            )

        # optional: post-process rewritten text to ensure it ends with the required instruction
        if "1" not in rewritten and "0" not in rewritten:
            # ensure instruction present
            rewritten = rewritten.strip() + "\n\nAnswer with exactly '1' or '0'."

        # 5) cache result
        self._cache[key] = {
            "created_at": datetime.now().isoformat(),
            "original": original_text,
            "rewritten": rewritten
        }
        self._serialize_cache()

        # 6) set rewritten text into sample and return
        data = self._set_text(data, rewritten)
        return data
