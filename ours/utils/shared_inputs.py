"""Atomic rewrite-input files owned by their task branches."""
import fcntl
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile


def input_path(log_file, dataset_id, model_id='', debug=False):
    if not log_file:
        raise ValueError('Rewritten inputs require a log_file')
    folder=Path(log_file).parent
    if folder.name==model_id:
        folder=folder.parent
    suffix='.debug.inputs.json' if debug else '.inputs.json'
    return folder/(dataset_id+suffix)


def validate_inputs(rows, path):
    if not isinstance(rows,list) or not rows:
        raise ValueError(f'Empty or invalid rewrite input file: {path}')
    ids=set()
    for row in rows:
        if not isinstance(row,dict) or 'id' not in row or 'content' not in row:
            raise ValueError(f'Invalid rewrite input row: {path}')
        key=str(row['id'])
        if key in ids:raise ValueError(f'Duplicate rewrite input ID {key}: {path}')
        ids.add(key)
        content=row['content']
        text=content.get('text') if isinstance(content,dict) else content
        if not isinstance(text,str) or not text.strip():raise ValueError(f'Empty rewrite {key}: {path}')
    return rows


def ensure_inputs(path, factory):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if path.exists():
            return validate_inputs(json.loads(path.read_text()),path)
        rows=validate_inputs(factory(),path)
        with NamedTemporaryFile('w',encoding='utf-8',dir=path.parent,prefix='.'+path.name,suffix='.tmp',delete=False) as f:
            temporary=Path(f.name)
            try:
                json.dump(rows,f,ensure_ascii=False,indent=2)
                f.write('\n');f.flush();os.fsync(f.fileno())
            except BaseException:
                temporary.unlink(missing_ok=True)
                raise
        temporary.replace(path)
        return rows


def to_record(sample):
    from ours import ImageTxtSample
    content=({'text':sample.text,'image_path':sample.image_path}
             if isinstance(sample,ImageTxtSample) else sample.text)
    row={'id':sample.id,'content':content,'target':sample.target}
    if sample.extra:row['extra']=sample.extra
    return row


def as_dataset(rows):
    from ours import ImageTxtSample,TxtSample
    from ours.datasets.base import BaseDataset
    class SharedDataset(BaseDataset):
        def __init__(self):
            self.method_hook=None
            self.dataset=[]
            for row in rows:
                content=row['content']
                common=dict(id=row['id'],target=row.get('target'),extra=row.get('extra'))
                if isinstance(content,dict):
                    sample=ImageTxtSample(text=content['text'],image_path=content['image_path'],**common)
                else:sample=TxtSample(text=content,**common)
                self.dataset.append(sample)
        def __len__(self):return len(self.dataset)
        def __getitem__(self,index):return self.dataset[index]
    return SharedDataset()
