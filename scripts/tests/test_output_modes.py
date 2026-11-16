"""Offline regression checks; no target model, rewrite API, or judge is called."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from ours import TxtSample
from ours.tasks.base import BaseTask
from ours.utils.shared_inputs import as_dataset, ensure_inputs, input_path
from ours.utils import shared_jobs


class FakeModel:
    calls=0
    def chat(self,**kwargs):
        type(self).calls+=1
        return 'A'

class FakeTask(BaseTask):
    loads=0
    rewrites=0
    shared_manifest_path=None
    def get_model(self):
        type(self).loads+=1
        return FakeModel()
    def get_dataset(self):
        method=getattr(self,'method',None)
        class Dataset:
            def __init__(self):self.dataset=[TxtSample(id=1,text='question',target='A')]
            def __len__(self):return len(self.dataset)
            def __getitem__(self,index):
                sample=copy.deepcopy(self.dataset[index])
                return method.run(sample) if method else sample
        return Dataset()
    def get_method(self):
        class Method:
            def run(self,sample):
                FakeTask.rewrites+=1
                sample.text='rewritten once'
                return sample
        return Method() if self.method_cfg else None
    def get_evaluators(self):
        if self.score_mode=='none':return None
        return [lambda preds,labels,extras,log,**kw: ({'accuracy':1.0},{'match':[True]*len(preds)})]

class ModesTest(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.root=Path(self.tmp.name)
        FakeTask.loads=FakeTask.rewrites=FakeModel.calls=0
        manifest=self.root/'data/shared_inputs/manifest.json'
        manifest.parent.mkdir(parents=True)
        manifest.write_text(json.dumps({'jobs':[{'dataset_id':'toy'}]}))
        FakeTask.shared_manifest_path=manifest
    def tearDown(self):self.tmp.cleanup()
    def task(self,mode='none',model='m1',rewrite=False,dataset_id='toy'):
        return FakeTask(dataset_id=dataset_id,model_id=model,orchestrator_id=None,
                        score_mode=mode,log_file=str(self.root/'task'/model/(dataset_id+'.json')),
                        method_cfg={'answer-flip-robustness':{}} if rewrite else {},
                        generation_kwargs={})
    def test_legacy_provenance_is_removed_when_resuming(self):
        task=self.task();task.pipeline()
        path=Path(task.response_file)
        row=json.loads(path.read_text())
        row['extra']={'input_fingerprint':'old','shared_source_model':'old','generated_on':'old'}
        path.write_text(json.dumps(row)+'\n')
        self.task().pipeline()
        self.assertNotIn('extra',json.loads(path.read_text()))
    def test_none_writes_only_jsonl(self):
        task=self.task();task.pipeline()
        self.assertEqual([p.name for p in Path(task.log_file).parent.iterdir()],['toy.responses.jsonl'])
        self.assertEqual(FakeModel.calls,1)
    def test_all_keeps_answers_and_scores(self):
        task=self.task('all');task.pipeline()
        result=json.loads(Path(task.log_file).read_text())
        self.assertEqual(result['total_results']['accuracy'],1.0)
        self.assertTrue(Path(task.response_file).is_file())
    def test_later_scoring_does_not_load_target(self):
        first=self.task();first.pipeline()
        FakeTask.loads=FakeModel.calls=0
        second=self.task('all');second.pipeline()
        self.assertEqual((FakeTask.loads,FakeModel.calls),(0,0))
        self.assertEqual(len(json.loads(Path(second.log_file).read_text())['per_sample_results']),1)
    def test_shared_rewrites_generated_only_once(self):
        # Registering adapters is mocked to guarantee this test cannot access a service.
        dataset_id='robustness-vqa-toy'
        with patch.dict(sys.modules,{'ours.models':types.ModuleType('ours.models')}):
            a=self.task(rewrite=True,dataset_id=dataset_id);a.pipeline()
            b=self.task(model='m2',rewrite=True,dataset_id=dataset_id);b.pipeline()
        self.assertEqual(FakeTask.rewrites,1)
        shared=self.root/'task'/(dataset_id+'.inputs.json')
        self.assertEqual(json.loads(shared.read_text())[0]['content'],'rewritten once')
        self.assertEqual(json.loads(Path(a.response_file).read_text())['content'],
                         json.loads(Path(b.response_file).read_text())['content'])
        self.assertFalse(Path(a.log_file).exists())
        self.assertFalse(Path(b.log_file).exists())
    def test_existing_task_rewrite_skips_rewrite_method(self):
        dataset_id='robustness-vqa-toy'
        path=self.root/'task'/(dataset_id+'.inputs.json')
        ensure_inputs(path,lambda:[{'id':1,'content':'prewritten','target':'A'}])
        with patch.dict(sys.modules,{'ours.models':types.ModuleType('ours.models')}):
            task=self.task(rewrite=True,dataset_id=dataset_id);task.pipeline()
        self.assertEqual(FakeTask.rewrites,0)
        self.assertEqual(json.loads(Path(task.response_file).read_text())['content'],'prewritten')
    def test_model_dependent_bias_keeps_rewrites_in_model_folder(self):
        dataset_id='bias-vqa-toy'
        with patch.dict(sys.modules,{'ours.models':types.ModuleType('ours.models')}):
            task=self.task(rewrite=True,dataset_id=dataset_id);task.pipeline()
        self.assertEqual(FakeTask.rewrites,1)
        self.assertTrue((self.root/'task/m1'/(dataset_id+'.inputs.json')).is_file())
        self.assertFalse((self.root/'task'/(dataset_id+'.inputs.json')).exists())
        self.assertTrue(Path(task.response_file).is_file())
    def test_cached_inputs_skip_factory(self):
        path=self.root/'toy.inputs.json'
        rows=[{'id':1,'content':'ready','target':None}]
        ensure_inputs(path,lambda:rows)
        self.assertEqual(ensure_inputs(path,lambda:self.fail('factory called')),rows)
    def test_two_workers_publish_one_rewrite(self):
        from concurrent.futures import ThreadPoolExecutor
        import time
        path=self.root/'parallel.inputs.json'
        calls=[]
        def factory():
            calls.append(1)
            time.sleep(0.05)
            return [{'id':1,'content':'shared','target':None}]
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures=[pool.submit(ensure_inputs,path,factory) for _ in range(2)]
            results=[future.result() for future in futures]
        self.assertEqual(len(calls),1)
        self.assertEqual(results[0],results[1])
    def test_shared_uses_logs_without_frozen_source(self):
        shared_jobs.ROOT=self.root
        folder=self.root/'data/shared_inputs';folder.mkdir(parents=True,exist_ok=True)
        job={'output_folder':'privacy/P4','dataset_id':'toy','count':1,'samples':'missing.json'}
        (folder/'manifest.json').write_text(json.dumps({'jobs':[job]}))
        image=self.root/'image.png';image.touch()
        path=self.root/'logs/privacy/P4/toy.inputs.json'
        rows=[{'id':1,'content':{'text':'existing rewrite','image_path':'image.png'},'target':None}]
        ensure_inputs(path,lambda:rows)
        _,jobs=shared_jobs.load_jobs()
        self.assertEqual(jobs[0][1],rows)
    def test_shared_raw_then_rules_without_target_reload(self):
        shared_jobs.ROOT=self.root
        job={'task_id':'R.4','branch':'answer-flip','output_folder':'robustness/R4','dataset_id':'toy',
             'generation_kwargs':{},'evaluator_seq_cfgs':[{'rule_mcq_eval':{'metrics_cfg':{'accuracy_score':{}}}}]}
        rows=[{'id':1,'content':{'text':'Pick A or B','image_path':'image.png'},'target':'A'}]
        from ours.utils.registry import registry
        from unittest.mock import patch
        fake=types.ModuleType('ours.models')
        class Model(FakeModel):
            def __init__(self,**kwargs):pass
        with patch.object(shared_jobs,'load_jobs',return_value=({},[(job,rows)])), \
             patch.dict(sys.modules,{'ours.models':fake}), \
             patch.object(registry,'get_chatmodel_class',return_value=Model):
            shared_jobs.infer_model('toy-model','none','local')
            folder=self.root/'logs/robustness/R4/toy-model'
            self.assertEqual([p.name for p in folder.iterdir()],['toy.responses.jsonl'])
            # The second run may score but must not instantiate the target again.
            with patch.object(registry,'get_chatmodel_class',side_effect=AssertionError('target reloaded')):
                shared_jobs.infer_model('toy-model','rules','local')
            self.assertTrue((folder/'toy.scored.json').is_file())
            self.assertTrue((folder/'toy.responses.jsonl').is_file())
    def test_failed_rewrite_does_not_publish_inputs(self):
        path=self.root/'toy.inputs.json'
        def broken():raise RuntimeError('rewrite interrupted')
        with self.assertRaises(RuntimeError):ensure_inputs(path,broken)
        self.assertFalse(path.exists())
    def test_changed_question_rejects_answer_cache(self):
        task=self.task();task.pipeline()
        path=Path(task.response_file)
        row=json.loads(path.read_text());row['content']='different'
        path.write_text(json.dumps(row)+'\n')
        with self.assertRaises(ValueError):self.task().pipeline()
    def test_truncated_thinking_is_regenerated(self):
        task=self.task();task.pipeline()
        path=Path(task.response_file)
        row=json.loads(path.read_text());row['response']='<think>unfinished reasoning'
        path.write_text(json.dumps(row)+'\n')
        FakeModel.calls=0
        self.task().pipeline()
        self.assertEqual(FakeModel.calls,1)
        self.assertEqual(json.loads(path.read_text())['response'],'A')
    def test_error_answer_is_regenerated(self):
        task=self.task();task.pipeline()
        path=Path(task.response_file)
        row=json.loads(path.read_text());row['response']='Error in generation: upstream failed'
        path.write_text(json.dumps(row)+'\n')
        FakeModel.calls=0
        self.task().pipeline()
        self.assertEqual(FakeModel.calls,1)
        self.assertEqual(json.loads(path.read_text())['response'],'A')
    def test_scored_and_jsonl_cache_merge_without_duplicate_answers(self):
        task=self.task();task.pipeline()
        row=json.loads(Path(task.response_file).read_text())
        scored=Path(task.log_file)
        scored.write_text(json.dumps({'total_results':{},
                                      'per_sample_results':[{**row,'response':''}]}))
        Path(task.response_file).write_text(json.dumps(row)+'\n')
        FakeTask.loads=FakeModel.calls=0
        self.task().pipeline()
        self.assertEqual((FakeTask.loads,FakeModel.calls),(0,0))
        self.assertEqual(len(Path(task.response_file).read_text().splitlines()),1)
        self.assertEqual(json.loads(Path(task.response_file).read_text())['response'],'A')
    def test_rejudge_cannot_silently_run_without_scoring(self):
        with self.assertRaises(ValueError):self.task().rejudge()
    def test_scoring_none_does_not_write_result_json(self):
        path=self.root/'result.json'
        shared_jobs.save_scored({'task_id':'P.4'},[],path,'none')
        shared_jobs.save_scored({'task_id':'P.4'},[],path,'rules')
        self.assertFalse(path.exists())
    def saved_scoring_fixture(self, count=1):
        shared_jobs.ROOT=self.root
        job={'task_id':'R.4','output_folder':'robustness/R4','dataset_id':'toy','count':count,
             'evaluator_seq_cfgs':[{'rule_mcq_eval':{'metrics_cfg':{'accuracy_score':{}}}}]}
        folder=self.root/'data/shared_inputs';folder.mkdir(parents=True,exist_ok=True)
        (folder/'manifest.json').write_text(json.dumps({'jobs':[job]}))
        source=self.root/'logs/robustness/R4/m1/toy.responses.jsonl'
        source.parent.mkdir(parents=True)
        source.write_text(json.dumps({'id':1,'content':{'text':'Choose A','image_path':'missing-image.png'},
                                      'target':'A','response':'A'})+'\n')
        return shared_jobs,source
    def test_strict_scoring_uses_only_raw_answers(self):
        module,source=self.saved_scoring_fixture()
        before=source.read_bytes()
        with patch.object(module,'load_jobs',side_effect=AssertionError('rewrite path used')):
            module.score_saved_model('m1','rules')
        self.assertEqual(source.read_bytes(),before)
        self.assertTrue(source.with_name('toy.scored.json').is_file())
        self.assertFalse((self.root/'logs/robustness/R4/toy.inputs.json').exists())
    def test_strict_scoring_refuses_incomplete_answers(self):
        module,source=self.saved_scoring_fixture(count=2)
        with patch.object(module,'save_scored',side_effect=AssertionError('partial file scored')):
            with self.assertRaises(ValueError):module.score_saved_model('m1','rules')
        self.assertFalse(source.with_name('toy.scored.json').exists())
    def test_strict_scoring_refuses_missing_answers(self):
        module,source=self.saved_scoring_fixture();source.unlink()
        with self.assertRaises(ValueError):module.score_saved_model('m1','rules')
        self.assertFalse(source.exists())
    def test_run_model_eval_only_skips_weights(self):
        spec=importlib.util.spec_from_file_location('model_cli',ROOT/'run_model.py')
        entry=importlib.util.module_from_spec(spec);spec.loader.exec_module(entry)
        entry.ROOT=self.root
        calls=[]
        scorer=types.ModuleType('scripts.judge_saved_answers')
        scorer.parse_args=lambda argv:argv
        scorer.run_model_batch=lambda argv:calls.append(argv)
        original=Path.cwd()
        try:
            with patch.object(entry,'model_specs',return_value={'toy':{'env':'missing-env','machine':'local'}}), \
                 patch.object(entry,'checkpoint_path',side_effect=AssertionError('weights checked')), \
                 patch.dict(sys.modules,{'scripts.judge_saved_answers':scorer}), \
                 patch.object(sys,'argv',['run_model.py','--model','toy','--eval-only','--_in-env']):
                entry.main()
        finally:os.chdir(original)
        self.assertEqual(calls,[['--model','toy']])
    def test_unified_judge_forwards_selection_and_preflight(self):
        spec=importlib.util.spec_from_file_location('model_cli',ROOT/'run_model.py')
        entry=importlib.util.module_from_spec(spec);spec.loader.exec_module(entry)
        entry.ROOT=self.root
        calls=[]
        scorer=types.ModuleType('scripts.judge_saved_answers')
        scorer.parse_args=lambda argv:argv
        scorer.run_model_batch=lambda argv:calls.append(argv)
        original=Path.cwd()
        try:
            with patch.object(entry,'model_specs',return_value={}), \
                 patch.dict(sys.modules,{'scripts.judge_saved_answers':scorer}), \
                 patch.object(sys,'argv',['run_model.py','--model','gpt-5.4','--eval-only','--_in-env',
                                         '--tasks','P.4','S.3','--base-url','http://localhost:9000/v1',
                                         '--judge-model','judge','--check-only']):
                entry.main()
        finally:os.chdir(original)
        self.assertEqual(calls,[['--model','gpt-5.4','--base-url','http://localhost:9000/v1',
                                 '--judge-model','judge','--tasks','P.4','S.3','--check-only']])
    def test_llm_scores_are_saved_next_to_raw_answers(self):
        spec=importlib.util.spec_from_file_location('judge_entry',ROOT/'scripts/judge_saved_answers.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        module.ROOT=self.root
        source=self.root/'logs/safety/S1/m1/safety-risk.responses.jsonl'
        source.parent.mkdir(parents=True)
        source.write_text(json.dumps({'id':1,'content':'question','response':'answer'})+'\n')
        before=source.read_bytes()
        args=module.parse_args(['--model','m1','--tasks','S.1'])
        def fake_run(branch_args,client=None):
            payload={'total_results':{'status_counts':{'ok':1}},'new_judge_calls':1}
            Path(branch_args.output).write_text(json.dumps(payload))
            return payload
        with patch.object(module,'BATCH_LAYOUT',{'S.1':('safety/S1',[('safety-risk',1)])}), \
             patch.object(module,'load_config',return_value=({'model':'judge'},[])), \
             patch.object(module,'prepare_case',return_value=({},None)), \
             patch.object(module,'run',side_effect=fake_run):
            report=module.run_model_batch(args)
        self.assertTrue(source.with_name('safety-risk.scored.json').is_file())
        self.assertEqual(source.read_bytes(),before)
        self.assertFalse((self.root/'logs_judged').exists())
        self.assertTrue((self.root/'run_state/judging/judge/m1.json').is_file())
        self.assertEqual(Path(report['branches'][0]['output']).parent,source.parent)
    def test_canonical_result_names_do_not_double_suffixes(self):
        from ours.utils.response_records import default_response_file,default_score_file
        for path in ['task.json','task.scored.json','task.responses.jsonl','task.rejudged.json']:
            self.assertEqual(default_score_file(path),'task.scored.json')
            self.assertEqual(default_response_file(path),'task.responses.jsonl')
    def test_rejudge_defaults_to_canonical_score_file(self):
        task=self.task();task.pipeline()
        before=Path(task.response_file).read_bytes()
        scored=self.task('all');scored.rejudge()
        self.assertTrue(Path(scored.log_file).name.endswith('.scored.json'))
        self.assertTrue(Path(scored.log_file).exists())
        self.assertEqual(Path(scored.response_file).read_bytes(),before)
    def test_aggregation_writes_canonical_score_name(self):
        from ours.utils.bias_utils import compute_unchange_intersection
        inputs=[]
        payload={'total_results':{'rule_mcq_eval:choice_unchange_rate':{'unchange_ids':[0]},
                                 'rule_mcq_eval:processed_summary':{'processed_preds':[1],'processed_labels':[0]}}}
        for name in ['a','b']:
            (self.root/(name+'.scored.json')).write_text(json.dumps(payload))
            inputs.append(str(self.root/(name+'.json')))
        compute_unchange_intersection(inputs,str(self.root/'summary.json'))
        self.assertTrue((self.root/'summary.scored.json').exists())
        self.assertFalse((self.root/'summary.json').exists())
    def test_repeated_samples_keep_each_answer_and_majority(self):
        task=self.task()
        task.generation_kwargs={'repeat_times':3}
        task.pipeline()
        self.assertFalse(Path(task.log_file).exists())
        record=json.loads(Path(task.response_file).read_text())
        self.assertEqual(record['pred'],['A']*3)
        self.assertEqual(record['major_answer'],'A')
        self.assertEqual(record['major_ratio'],1.0)
        self.assertEqual([extra['repeat_idx'] for extra in record['extra']],[0,1,2])

if __name__=='__main__':unittest.main()
