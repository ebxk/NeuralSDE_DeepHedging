import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch,Mock
import run_pipeline as pipeline
import run_training_queue as queue
from study_core import CONFIG,digest,write_json
from prediction_audit import calendar_interval


class RecoveryTests(unittest.TestCase):
    def test_expiry_forecast_uses_the_same_economic_payoff_as_the_ledger(self):
        from prediction_audit import observed_increment
        first={'option_close':.10};last={'spot':2.40,'strike':2.30,'option_close':.096}
        self.assertAlmostEqual(observed_increment(first,last,0.),0.)
        self.assertAlmostEqual(observed_increment(first,last,1/365),-.004)

    def test_quantified_localvol_rejection_does_not_stop_other_benchmarks(self):
        import subprocess
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder);source=out/'producer.py';source.write_bytes(b'current financial source')
            record=dict(complete=False,status='numerical_failed',model='localvol',end=2025,
                error_type='ValueError',error='Dupire invalid: nonpositive variance derivative or density denominator; no clipping applied',
                sources={str(source):digest(source)},parents={str(source):digest(source)})
            status=out/'direct_dynamic/2025/localvol/status.json';write_json(status,record)
            error=subprocess.CalledProcessError(1,['direct_dynamic.py'])
            def rejected(*args):write_json(status,record);raise error
            with patch.object(pipeline,'OUT',out),patch.object(pipeline,'command',side_effect=rejected):
                result=pipeline.direct_benchmark(2025,'localvol')
                self.assertEqual(result['status'],'numerical_failed')
                self.assertEqual(json.loads(status.read_text()),record)
                record['error']='unexpected input or software defect';write_json(status,record)
                with self.assertRaises(subprocess.CalledProcessError):pipeline.direct_benchmark(2025,'localvol')
            record['error']='Dupire invalid: measured negative variance';write_json(status,record)
            with patch.object(pipeline,'OUT',out),patch.object(pipeline,'command',side_effect=error):
                with self.assertRaises(subprocess.CalledProcessError):pipeline.direct_benchmark(2025,'localvol')

    def test_actual_queue_rebuilt_ready_supersedes_block(self):
        job={'key':'fixture','model':'gbm','end':2025}
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder);inp=out/'inputs/2025';inp.mkdir(parents=True)
            data=inp/'sim_gbm.npz';data.write_bytes(b'array fixture')
            source=out/'producer.py';source.write_bytes(b'source')
            r=dict(complete=True,model='gbm',end=2025,config=CONFIG,input_sha256=digest(data),
                sources={str(source):digest(source)},parents={})
            write_json(out/'failures/fixture.json',{'status':'blocked_by_generator'})
            with patch.object(queue,'OUT',out):
                self.assertEqual(queue.resolved(job),'blocked')
                write_json(inp/'sim_gbm.ready.json',r)
                self.assertEqual(queue.resolved(job),'train')
                source.write_bytes(b'changed source')
                self.assertEqual(queue.resolved(job),'blocked')

    def test_dead_producer_is_reported_even_without_terminal_receipt(self):
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder);child=Mock();child.poll.return_value=1
            with patch.object(pipeline,'OUT',out),patch.object(pipeline,'OWNED',{'surface':child}):
                with self.assertRaisesRegex(RuntimeError,'producer exited'):pipeline.check_producers()
            self.assertEqual(json.loads((out/'producer_failure.json').read_text())['returncode'],1)

    def test_environment_does_not_mix_python_interpreters(self):
        with patch.dict(os.environ,{'PYTHONHOME':'wrong','PYTHONPATH':'wrong','VIRTUAL_ENV':'wrong'}):
            env=pipeline.process_environment()
        for key in ('PYTHONHOME','PYTHONPATH','VIRTUAL_ENV'):self.assertNotIn(key,env)

    def test_cancellation_stops_only_registered_live_writers(self):
        live=Mock(pid=123);live.poll.return_value=None
        exited=Mock(pid=456);exited.poll.return_value=0
        with patch.object(pipeline,'OWNED',{'surface':live,'seed0':exited}),patch.object(pipeline,'COMMANDS',set()),\
                patch.object(pipeline.subprocess,'run') as kill:
            pipeline.stop_owned()
        if pipeline.os.name=='nt':
            kill.assert_called_once();self.assertEqual(kill.call_args.args[0],['taskkill','/PID','123','/T','/F'])
        self.assertTrue(pipeline.CANCELLED.is_set());pipeline.CANCELLED.clear()

    def test_calendar_expiry_is_exact_zero_not_float32_subtraction(self):
        first={'date':'2024-01-02','expiry':'2024-01-24'};last={'date':'2024-01-24'}
        elapsed,left=calendar_interval(first,last)
        self.assertEqual(left,0);self.assertEqual(elapsed,22/365)

if __name__=='__main__':unittest.main()
