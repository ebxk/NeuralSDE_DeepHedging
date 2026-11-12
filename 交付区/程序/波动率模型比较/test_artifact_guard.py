import json
import tempfile
import unittest
from pathlib import Path
from artifact_guard import verify_ready,queue_decision,require_delivery_manifest
from study_core import digest,write_json

class ArtifactGuardTests(unittest.TestCase):
    def test_ready_checks_artifact_parent_and_source_bytes(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder);data=p/'sim.npz';parent=p/'parent';source=p/'producer.py'
            for f in (data,parent,source):f.write_bytes(b'original')
            ready=p/'sim.ready.json'
            r=dict(complete=True,input_sha256=digest(data),parents={str(parent):digest(parent)},
                sources={str(source):digest(source)})
            write_json(ready,r);verify_ready(ready,data)
            for field in ('parent','source','data'):
                changed={'parent':parent,'source':source,'data':data}[field]
                changed.write_bytes(b'changed')
                with self.assertRaises(ValueError):verify_ready(ready,data)
                changed.write_bytes(b'original')
            r['complete']=False;write_json(ready,r)
            with self.assertRaises(ValueError):verify_ready(ready,data)

    def test_rebuilt_generator_supersedes_blocked_failure(self):
        self.assertEqual(queue_decision(True,False,{'status':'blocked_by_generator'}),'train')
        self.assertEqual(queue_decision(False,False,{'status':'blocked_by_generator'}),'blocked')
        self.assertEqual(queue_decision(False,False,None),'wait')

    def test_missing_diagnostic_prevents_complete_delivery(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ValueError):require_delivery_manifest(Path(folder)/'missing.json')

if __name__=='__main__':unittest.main()
