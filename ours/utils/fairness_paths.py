from pathlib import Path
from ours.utils.response_records import default_score_file

def existing_fairness_log(path):
    path = Path(path)
    if path.suffix == '.json':
        canonical = Path(default_score_file(str(path)))
        if canonical.is_file(): return canonical
    return path
