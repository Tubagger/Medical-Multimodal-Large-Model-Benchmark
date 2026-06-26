import copy
import os, json, hashlib
from datetime import datetime
import random
import re
from typing import Any, Dict
from ours import methods
from ours.methods.base import BaseMethod
from ours.utils.registry import registry


@registry.register_method()
class RobustnessAgent(BaseMethod):
    """
    Each method_id has its own (system,user) prompt template.
    The agent generates ONE bias-inducing sentence and injects it into the 'Clinical note' section
    of the provided question_template.

    Usage:
      data = method.run(data, tool="cognition-bias", unbiased_model_choice="C")
    """

    method_ids = [
        "answer-flip-robustness",
        "question-negation-robustness",
        "option-expansion-robustness",
        "narrative-distraction-robustness",
    ]

    def __init__(
        self,
        model_id: str,
        method_id: str,
        ref_cache_dir: str,
        lazy_mode: bool = True,
        generation_kwargs: dict = None,
        **kwargs,  # swallow extra YAML keys safely
    ):
        super().__init__(model_id, method_id, ref_cache_dir, lazy_mode)

        self.model_id = model_id
                
        self.generation_kwargs = generation_kwargs or {
            "choice_model_id": 'deepseek-r1',
            "rewrite_mdoel_id": 'DeepSeek-V3.2',
            "max_new_tokens": 120,
            "temperature": 0.0,
            "do_sample": False,
        }


        self.cache_path = os.path.join(
            self.ref_cache_dir,
            f"{self.model_id}",
            f"{self.method_id}.jsonl"
        )
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        
        self.cache = {}
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue

                        obj = json.loads(line)
                        id = obj.get("id", None)

                        if id is not None:
                            self.cache[id] = obj
            except Exception as e:
                print(f"[Cache Load Error] {e}")
                self.cache = {}


    # ===== required by BaseMethod =====
    def hash(self, s: str, **_) -> str:
        return hashlib.sha1((s or "").encode("utf-8")).hexdigest()

    def _get_model(self,model_id):
        if not hasattr(self, "_model"):
            model_cls = registry.get_chatmodel_class(model_id)
            self._model = model_cls(model_id)

        return self._model

    def _extract_text(self, data: Any) -> str:
        if isinstance(data, dict):
            if "message" in data:
                msgs = data["message"]
                if isinstance(msgs, list) and msgs:
                    content = msgs[0].get("content")
                    if isinstance(content, dict):
                        return content.get("text", "") if content.get("text") is not None else str(content)
                    return str(content)
            if "text" in data:
                return str(data["text"])
            if "content" in data:
                return str(data["content"])
        if hasattr(data, "text"):
            return str(getattr(data, "text"))
        if hasattr(data, "message"):
            msgs = getattr(data, "message")
            if isinstance(msgs, list) and msgs:
                m0 = msgs[0]
                content = m0.get("content") if isinstance(m0, dict) else getattr(m0, "content", "")
                if isinstance(content, dict):
                    return content.get("text", "") if content.get("text") is not None else str(content)
                return str(content)
        return str(data) if isinstance(data, str) else ""

    def _set_text(self, data: Any, new_text: str, new_target: Any = None) -> Any:

        # =========================
        # 1. dict case
        # =========================
        if isinstance(data, dict):

            # ---- message format ----
            if "message" in data:
                msgs = data["message"]
                if isinstance(msgs, list) and msgs:

                    content = msgs[0].get("content")

                    if isinstance(content, dict):
                        content["text"] = new_text
                        msgs[0]["content"] = content
                    else:
                        msgs[0]["content"] = new_text

                    data["message"] = msgs

            # ---- normal text ----
            if "text" in data:
                data["text"] = new_text

            # ---- content field ----
            if "content" in data:
                data["content"] = new_text

            # =========================
            # 2. update target (关键新增)
            # =========================
            if new_target is not None:
                data["target"] = new_target

            return data

        # =========================
        # 3. object case
        # =========================
        if hasattr(data, "text"):
            setattr(data, "text", new_text)

        if new_target is not None and hasattr(data, "target"):
            setattr(data, "target", new_target)

        return data

    def _inject_into_clinical_note(self, question_template: str, bias_sentence: str) -> str:
        bias_sentence = (bias_sentence or "").strip().strip('"').strip()
        if not bias_sentence:
            return question_template

        if "Clinical note:" in question_template:
            head, rest = question_template.split("Clinical note:", 1)
            rest = rest.lstrip()
            return f"{head}Clinical note:\n{bias_sentence}\n{rest}"
        else:
            # if template doesn't have the section, append one
            return question_template.rstrip() + f"\n\nClinical note:\n{bias_sentence}\n"

    def _call_bias_sentence_agent(self, tool: str, question_template: str, unbiased_model_choice: str) -> str:
        cfg = self.PROMPTS.get(tool) or self.PROMPTS["cognition-bias"]

        system_prompt = cfg["system"].format(unbiased_model_choice=unbiased_model_choice)
        user_msg = cfg["user"].format(
            question_template=question_template,
            unbiased_model_choice=unbiased_model_choice,
        )

        resp = self._get_model().chat(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg},
            ],
            **self.generation_kwargs
        )
        print(f"=========use ")
        out = resp.content if hasattr(resp, "content") else str(resp)
        # enforce single sentence-ish (light cleanup)
        out = out.strip().strip('"').strip()
        out = out.replace("\n", " ").strip()
        return out

    def run(self, data: Any, **kwargs) -> Any:
        id = data.id

        if id in self.cache:
            label = self.cache[id]["label"]
            rewritten = self.cache[id]["rewritten"]
            rewrite_model_id = self.cache[id]["rewrite_model_id"]
            test_model_id = self.cache[id]["test_model_id"]

            print(
                f"[{self.method_id}][Cache] "
                f"id={data.id} | "
                f"label={label} | "
                f"test_model={test_model_id} | "
                f"rewrite_model={rewrite_model_id}"
            )


            return self._set_text(
                data,
                rewritten,
                label
            )
        else:
            if self.method_id == "answer-flip-robustness":

                label, rewritten = (
                    self.answer_flip(
                        data,
                    )
                )

   
            elif self.method_id == "question-negation-robustness":
                label, rewritten = (
                    self.question_negation(
                        data,
                        model_id=self.generation_kwargs['rewrite_model_id']
                    )
                )


            elif self.method_id == 'option-expansion-robustness':
                label, rewritten = (
                    self.option_expansion(
                        data, 
                        model_id=self.generation_kwargs['rewrite_model_id']
                    )
                )


            elif self.method_id == 'narrative-distraction-robustness':
                label, rewritten = (
                    self.narrative_distraction(
                        data,
                        model_id=self.generation_kwargs['rewrite_model_id']
                    )
                )


            self.cache[id] = {
                "label": label,
                "rewritten": rewritten
            }

            with open(self.cache_path, "a", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {
                            "id": id,
                            "label": label,
                            "rewritten": rewritten,
                            "test_model_id": self.model_id,
                            "rewrite_model_id": self.generation_kwargs['rewrite_model_id'],
                            "temperature": self.generation_kwargs['rewrite_model_id']
                        },
                        ensure_ascii=False
                    ) + "\n"
                )

            data.text = rewritten
            data.target = label
            
            return data



    def answer_flip(self, data):

        # =========================
        # 0. safe copy
        # =========================
        text = data.text
        answer = data.target

        if not text or answer is None:
            return  None, None

        # =========================
        # 1. extract Options block
        # =========================
        pattern = r"(Options:\n)(.*?)(\nInstruction:)"
        match = re.search(pattern, text, re.DOTALL)

        if not match:
            return data, None, None

        options_block = match.group(2)
        lines = options_block.strip().split("\n")

        new_lines = []
        flipped = False
        rewritten = None

        # =========================
        # 2. replace correct option
        # =========================
        for line in lines:

            if line.startswith(answer + "."):

                new_line = f"{answer}. None of the options are correct"
                new_lines.append(new_line)

                flipped = True

            else:
                new_lines.append(line)

        # =========================
        # 3. safety check
        # =========================
        if not flipped:
            return None, None

        # =========================
        # 4. rebuild text
        # =========================
        new_options_block = "\n".join(new_lines)
        new_text = text.replace(options_block, new_options_block)

        print(
        f"[answer_flip] "
        f"id={data.id} | "
        f"answer={answer} | "
        f"rewritten={new_text}"
        )

        return answer, new_text



    def question_negation(self, data, model_id=None):

        text = data.text
        answer = data.target

        if answer is None:
            return data, None, None

        # =========================
        # 1. extract question
        # =========================
        pattern = r"(Question:\n)(.*?)(\nOptions:)"
        match = re.search(pattern, text, re.DOTALL)

        if not match:
            return None, None

        question = match.group(2)

        # =========================
        # 2. LLM rewrite question (ONLY this part)
        # =========================
        system_prompt = """
        You are a medical question rewriting assistant.

        Task:
        Rewrite the given question into a NEGATED form.

        Requirements:
        - Keep medical meaning consistent
        - Do NOT change options
        - Do NOT change meaning of entities
        - Make it natural (NOT / incorrect / false version)
        - Do NOT add extra information

        Return ONLY JSON:
        {
            "question": "..."
        }
        """

        user_prompt = f"""
        Question:
        {question}
        """

        model_cls = registry.get_chatmodel_class(model_id)
        chatmodel = model_cls(model_id)

        response = chatmodel.chat(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_new_tokens=128,
            do_sample=False,
        )

        try:
            negated_question = response.content
        except:
            # fallback to safe rule-based
            negated_question = "Which of the following is NOT correct: " + question

        # =========================
        # 3. replace question in text
        # =========================
        new_text = text.replace(question, negated_question)

        # =========================
        # 4. get options (UNCHANGED LOGIC)
        # =========================
        options_pattern = r"Options:\n(.*?)(\nInstruction:)"
        opt_match = re.search(options_pattern, text, re.DOTALL)

        if not opt_match:
            return None, None

        options_block = opt_match.group(1)
        lines = options_block.strip().split("\n")

        # =========================
        # 5. multi-label shift (KEEP YOUR LOGIC)
        # =========================
        labels = []
        for l in lines:
            if "." in l:
                labels.append(l.split(".")[0].strip())

        new_candidates = [l for l in labels if l != answer]
        new_label = sorted(list(set(new_candidates)))



        print(
            f"[question_negation-LLM] "
            f"id={getattr(data, 'id', None)} | "
            f"old={answer} | "
            f"new={new_label}"
        )

        return new_label, new_text
        

    def option_expansion(self, data, model_id=None):

        text = data.text
        answer = data.target

        if answer is None:
            return data, None, None

        # =========================
        # 1. extract options block
        # =========================
        pattern = r"(Options:\n)(.*?)(\nInstruction:)"
        match = re.search(pattern, text, re.DOTALL)

        if not match:
            return None, None

        options_block = match.group(2)
        lines = options_block.strip().split("\n")

        # =========================
        # 2. convert options to prompt
        # =========================
        existing_options = "\n".join(lines)

        system_prompt = """
        You are a medical dataset generator.
        Task:
        Generate ONE additional incorrect but plausible distractor option
        for a multiple-choice medical question.
        Rules:
        - Must be medically plausible
        - Must be clearly incorrect
        - Must NOT duplicate existing options
        - Keep it short (1 phrase)
        Return ONLY the option text.
        """

        user_prompt = f"""
        Existing options:
        {existing_options}
        Generate one new distractor option.
        """

        # =========================
        # 3. call LLM
        # =========================
        model_cls = registry.get_chatmodel_class(model_id)
        chatmodel = model_cls(model_id)

        response = chatmodel.chat(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_new_tokens=64,
            do_sample=False,
        )

        new_option_text = response["content"].strip()

        # =========================
        # 4. assign new label
        # =========================
        existing_keys = [l.split(".")[0] for l in lines if "." in l]
        last_key = sorted(existing_keys)[-1]
        next_key = chr(ord(last_key) + 1)

        new_line = f"{next_key}. {new_option_text}"

        # =========================
        # 5. append option
        # =========================
        new_lines = lines + [new_line]
        new_options_block = "\n".join(new_lines)

        new_text = text.replace(options_block, new_options_block)
        # =========================
        # 6. outputs
        # =========================
        label = answer


        print(
        f"[option_expansion] "
        f"id={data.id} | "
        f"answer={label} | "
        f"rewritten={new_text}"
        )

        return label, new_text
        
    def narrative_distraction(self, data, model_id=None):

        text = data.text
        answer = data.target


        # =========================
        # 1. extract question block
        # =========================
        pattern = r"(Question:\n)(.*?)(\nOptions:)"
        match = re.search(pattern, text, re.DOTALL)

        if not match:
            return None, None

        question = match.group(2)

        # =========================
        # 2. LLM generate distraction sentence
        # =========================
        system_prompt = """
        You are a medical dataset adversarial assistant.
        Generate ONE short irrelevant but realistic clinical sentence.
        Must NOT affect diagnosis.
        Return ONLY the sentence.
        """

        user_prompt = f"""
        Question:
        {question}
        """

        model_cls = registry.get_chatmodel_class(model_id)
        chatmodel = model_cls(model_id)

        response = chatmodel.chat(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_new_tokens=64,
            do_sample=False,
        )

        distraction_sentence = response["content"].strip()

        # =========================
        # 3. SAFE INSERT (关键修改)
        # =========================
        new_question = question.strip() + " " + distraction_sentence

        # ✔ 用 span 替换，避免结构错乱
        start, end = match.span(2)

        new_text = text[:start] + new_question + text[end:]

        # =========================
        # 4. debug log
        # =========================
        print(
            f"[narrative_distraction] "
            f"id={getattr(data, 'id', None)} | "
            f"answer={answer} | "
            f"rewritten={new_text}"
        )

        # =========================
        # 5. return
        # =========================
        return answer, new_text