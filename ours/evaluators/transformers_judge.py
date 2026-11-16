"""Offline multimodal judge transport; scoring stays in local_judge_engine."""
import base64
import copy
import io
import time
from pathlib import Path


def convert_messages(messages):
    from PIL import Image
    converted = copy.deepcopy(messages)
    for message in converted:
        if isinstance(message['content'], str):
            continue
        for part in message['content']:
            if part['type'] == 'image_url':
                url = part['image_url']['url']
                if not url.startswith('data:image/') or ';base64,' not in url:
                    raise ValueError('Offline judge accepts embedded images only')
                with Image.open(io.BytesIO(base64.b64decode(url.split(',', 1)[1]))) as source:
                    picture = source.convert('RGB')
                part.clear()
                part.update(type='image', image=picture)
    return converted


class TransformersJudgeClient:
    def __init__(self, config):
        import torch
        from transformers import AutoProcessor, AutoModelForImageTextToText
        folder = Path(config['model_path']).expanduser().resolve()
        if not (folder / 'config.json').is_file():
            raise ValueError(f'Local judge config.json missing: {folder}')
        if not torch.cuda.is_available():
            raise RuntimeError('Transformers judge requires an available CUDA GPU')
        self.torch = torch
        self.generation = config['generation']
        self.processor = AutoProcessor.from_pretrained(folder, local_files_only=True)
        self.model = AutoModelForImageTextToText.from_pretrained(
            folder, local_files_only=True, torch_dtype=torch.bfloat16,
            device_map=config.get('device_map', 'auto')).eval()
        template = self.generation.get('extra_body', {}).get('chat_template_kwargs', {})
        print(f"[Judge] thinking={template.get('enable_thinking', True)}, "
              f"max_new_tokens={self.generation['max_tokens']}, "
              f"sampling={self.generation.get('temperature', 0) > 0}", flush=True)

    def __call__(self, messages):
        started = time.perf_counter()
        generation = self.generation
        template = generation.get('extra_body', {}).get('chat_template_kwargs', {})
        inputs = self.processor.apply_chat_template(
            convert_messages(messages), tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors='pt', **template)
        inputs = inputs.to(self.model.device)
        options = {'max_new_tokens': generation['max_tokens'],
                   'do_sample': generation.get('temperature', 0) > 0,
                   'use_cache': True}
        if options['do_sample']:
            options.update(temperature=generation['temperature'], top_p=generation.get('top_p', 1))
            if 'top_k' in generation.get('extra_body', {}):
                options['top_k'] = generation['extra_body']['top_k']
        with self.torch.inference_mode():
            output = self.model.generate(**inputs, **options)
        tokens = output[0, inputs['input_ids'].shape[1]:]
        content = self.processor.decode(tokens, skip_special_tokens=True)
        eos = self.model.generation_config.eos_token_id
        eos = eos if isinstance(eos, list) else [eos]
        stopped = bool(len(tokens)) and int(tokens[-1]) in eos
        return {'content': content, 'finish_reason': 'stop' if stopped else 'length',
                'reasoning_content': None, 'generated_tokens': len(tokens),
                'elapsed_seconds': round(time.perf_counter() - started, 3)}
