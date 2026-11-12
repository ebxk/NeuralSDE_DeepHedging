import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import json


class AcceptanceTests(unittest.TestCase):
    def test_prediction_requires_current_source_and_input_provenance(self):
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder);directory=out/'predictions';directory.mkdir()
            path=directory/'2025_gbm.json';csv=path.with_suffix('.csv')
            pd.DataFrame(dict(origin=[0,1,2],horizon=[1,5,20],n=[4]*3,residual_crps=[.1]*3,
                residual_observed=[.2]*3,residual_status=['own_model_price_increments']*3)).to_csv(csv,index=False)
            record=dict(complete=True,results=3,excluded=[])
            path.write_text(json.dumps(record),encoding='utf8');a=self.m.Auditor();a.coverage['generator.2025.gbm']={'status':'complete'}
            with patch.object(self.m,'OUT',out):
                with self.assertRaises(ValueError):a.prediction(2025,'gbm')
                source=out/'input';source.write_bytes(b'frozen input')
                record.update(model='gbm',end=2025,config=json.loads(json.dumps(self.m.CONFIG)),
                    source=self.m.digest(self.m.HERE/'prediction_audit.py'),parents={str(source):self.m.digest(source)},
                    output_sha256={csv.name:self.m.digest(csv)})
                path.write_text(json.dumps(record),encoding='utf8');self.assertEqual(a.prediction(2025,'gbm')['results'],3)
                source.write_bytes(b'changed input')
                with self.assertRaises(ValueError):a.prediction(2025,'gbm')

    def test_measured_dupire_direct_rejection_is_preserved_as_failure(self):
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder);source=out/'producer.py';source.write_bytes(b'current financial source')
            record=dict(complete=False,status='numerical_failed',model='localvol',end=2025,
                error_type='ValueError',error='Dupire invalid: nonpositive variance derivative or density denominator; no clipping applied',
                sources={str(source):self.m.digest(source)},parents={str(source):self.m.digest(source)})
            path=out/'direct_dynamic/2025/localvol/status.json';path.parent.mkdir(parents=True);path.write_text(json.dumps(record),encoding='utf8')
            auditor=self.m.Auditor()
            with patch.object(self.m,'OUT',out):
                result=auditor.direct(2025,'localvol')
                self.assertEqual(result['status'],'declared_failure')
                self.assertEqual(result['failure_kind'],'quantified_fit_failure')
                source.write_bytes(b'changed')
                with self.assertRaises(ValueError):auditor.direct(2025,'localvol')

    @classmethod
    def setUpClass(cls):
        path=Path(__file__).with_name('acceptance_manifest.py');cls.exists=path.exists()
        if cls.exists:
            spec=importlib.util.spec_from_file_location('acceptance',path)
            cls.m=importlib.util.module_from_spec(spec);spec.loader.exec_module(cls.m)

    def setUp(self):self.assertTrue(self.exists,'full-coverage acceptance manifest must exist')

    def test_expected_keys_are_fixed_and_empty_cannot_complete(self):
        keys=self.m.expected_stage_keys()
        self.assertEqual(len(keys),len(set(keys)))
        self.assertEqual(sum(k.startswith('policy.') for k in keys),432)
        for prefix in ['direct.static.','direct.nsde.','factor.','prediction.','evaluation.job.']:
            self.assertTrue(any(k.startswith(prefix) for k in keys))
        result=self.m.audit_coverage({},keys,{})
        self.assertFalse(result['complete']);self.assertTrue(result['pending'])

    def test_missing_direct_predict_or_factor_blocks_delivery(self):
        with tempfile.TemporaryDirectory() as d:
            artifact=Path(d)/'evidence.json';artifact.write_text('{}',encoding='utf8')
            hashes={str(artifact):self.m.digest(artifact)}
            required=('direct.static.2025','prediction.2025.sabr','factor.2025.nsde')
            rows={key:dict(status='complete',artifacts=hashes) for key in required}
            self.assertTrue(self.m.audit_coverage(rows,required,hashes)['complete'])
            for absent in required:
                partial={key:value for key,value in rows.items() if key!=absent}
                result=self.m.audit_coverage(partial,required,hashes)
                self.assertFalse(result['complete']);self.assertIn(absent,result['pending'])

    def test_blocked_is_explicit_and_requires_verified_chain(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'failure.json';p.write_text('{}',encoding='utf8');hashes={str(p):self.m.digest(p)}
            generator='generator.2025.sabr';prediction='prediction.2025.sabr'
            rows={generator:dict(status='declared_failure',failure_kind='generator_blocked',category='numerical_rejection',reason='numerical failure',artifacts=hashes),
                  prediction:dict(status='declared_failure',failure_kind='blocked_by_generator',blocked_by=generator,reason='same generator',artifacts=hashes)}
            result=self.m.audit_coverage(rows,(generator,prediction),hashes)
            self.assertTrue(result['complete']);self.assertEqual(len(result['declared_failures']),2)
            self.assertNotEqual(result['coverage'][prediction]['status'],'complete')
            del rows[generator]
            result=self.m.audit_coverage(rows,(generator,prediction),hashes)
            self.assertFalse(result['complete']);self.assertTrue(result['stage_errors'])

    def test_empty_workspace_writes_incomplete_and_stale_artifact_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)
            with patch.object(self.m,'OUT',out):result=self.m.build_manifest()
            self.assertFalse(result['complete']);self.assertTrue((out/'acceptance_manifest.json').exists())
            p=out/'artifact';p.write_bytes(b'actual')
            result=self.m.audit_coverage({'stage':{'status':'complete','artifacts':{str(p):'0'*64}}},('stage',),{str(p):'0'*64})
            self.assertFalse(result['complete']);self.assertTrue(result['stage_errors'])

    def test_executed_scientific_failure_survives_stage_wrapper(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'result';p.write_text('measured failure',encoding='utf8')
            a=self.m.Auditor()
            def measured():
                a.artifact(p)
                return dict(status='declared_failure',failure_kind='quantified_fit_failure',reason='five failed numerical fits')
            row=a.stage('factor.fixture',measured)
            self.assertEqual(row['status'],'declared_failure')
            self.assertTrue(self.m.audit_coverage(a.coverage,('factor.fixture',),a.artifacts)['complete'])

    def test_decimal_lambda_evaluation_checks_actual_full_filename(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d);evaluation=out/'evaluation';evaluation.mkdir()
            job={'key':'2025_sabr_surface_l0.1_s0','end':2025};records=[]
            for bp,execution in self.m.CASES:
                stem=evaluation/f'{job["key"]}_cost{bp}_{execution}'
                pd.DataFrame({'episode':[3,8],'pnl':[.1,-.1],'cross_dividend':[0,1]}).to_csv(Path(str(stem)+'.paths.csv'),index=False)
                Path(str(stem)+'.ledger.csv.gz').write_bytes(b'financial ledger bytes')
                np.savez(Path(str(stem)+'.npz'),pnl=np.array([.1,-.1]))
                records.append(dict(key=job['key'],cost_bp=bp,execution=execution,n=2,ledger_max_error=0.))
            pd.DataFrame(records).to_csv(evaluation/'overall.csv',index=False)
            a=self.m.Auditor();a.coverage['policy.'+job['key']]={'status':'complete'}
            a.cross_cache[2025]={3:0,8:1}
            with patch.object(self.m,'OUT',out),patch.object(a,'paths',return_value=np.array([3,8])):
                a.evaluation_job(job)

    def test_infrastructure_and_stale_training_failure_do_not_close_scope(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d);job=self.m._jobs()[0]
            ready=out/f'inputs/{job["end"]}/sim_{job["model"]}.ready.json';ready.parent.mkdir(parents=True)
            ready.write_text('{}',encoding='utf8')
            failure=out/f'failures/{job["key"]}.json';failure.parent.mkdir()
            context={'ready':self.m.digest(ready),'trainer':self.m.digest(self.m.HERE/'train_common.py'),
                     'policy':self.m.digest(self.m.HERE/'common_policy.py')}
            record=dict(status='training_failed_current_generation',key=job['key'],context=context,
                        category='infrastructure_failure',reason='script crashed')
            a=self.m.Auditor();a.coverage[f'generator.{job["end"]}.{job["model"]}']={'status':'complete'}
            with patch.object(self.m,'OUT',out):
                failure.write_text(json.dumps(record),encoding='utf8')
                with self.assertRaises(ValueError):a.policy(job)
                record['category']='numeric';record['context']=dict(context,ready='0'*64)
                failure.write_text(json.dumps(record),encoding='utf8')
                with self.assertRaises(ValueError):a.policy(job)
                record['context']=context;record['reason']='nonfinite training objective'
                failure.write_text(json.dumps(record),encoding='utf8')
                self.assertEqual(a.policy(job)['status'],'declared_failure')


if __name__=='__main__':unittest.main(verbosity=2)
