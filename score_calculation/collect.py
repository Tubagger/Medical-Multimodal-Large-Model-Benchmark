#!/usr/bin/env python3
"""Collect saved task metrics; never invoke inference or a judge."""
import argparse,csv,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',help='Optional target model ID')
    parser.add_argument('--output-dir',type=Path,default=ROOT/'score_calculation/results')
    args=parser.parse_args()
    rows=[]
    for folder in [ROOT/'logs']:
        if not folder.exists():continue
        for path in sorted(folder.rglob('*.scored.json')):
            if args.model and args.model not in path.parts:continue
            try:payload=json.loads(path.read_text())
            except ValueError:continue
            if not isinstance(payload,dict) or not isinstance(payload.get('total_results'),dict):continue
            metrics=payload['total_results']
            rows.append({'source':str(path.relative_to(ROOT)),
                         'status':'scored' if metrics else 'unscored',
                         'records':len(payload.get('per_sample_results',[])),
                         'metrics':metrics})
    args.output_dir.mkdir(parents=True,exist_ok=True)
    (args.output_dir/'task_metrics.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n')
    with (args.output_dir/'task_metrics.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['source','status','records','metrics']);writer.writeheader()
        for row in rows:writer.writerow({**row,'metrics':json.dumps(row['metrics'],ensure_ascii=False)})
    print(f'Collected {len(rows)} result files -> {args.output_dir}')

if __name__=='__main__':main()
