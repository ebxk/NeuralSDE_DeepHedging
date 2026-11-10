import bootstrap
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from bottleneck_core import jobs
from migrate_main_identity import migrate

class MigrationTests(unittest.TestCase):
    def test_main_old_path_is_checked_even_if_source_hash_was_already_current(self):
        project=bootstrap.PROJECT
        source=project/'工作区/数据/处理后/瓶颈验证实验'
        task=next(t for t in jobs() if t['key']=='2025_teacher_s0')
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder)
            for name in ('policies','initial'):(out/name).mkdir()
            for name in ('policies/2025_teacher_s0.pt','policies/2025_teacher_s0.json',
                         'initial/2025_wide_s0.pt','foundational_train_v1.py'):
                shutil.copyfile(source/name,out/name)
            marker=out/'policies/2025_teacher_s0.json'
            payload=json.loads(marker.read_text(encoding='utf8'))
            old=str(project/'工作区/数据/处理后/完整认购/episode_days.csv')
            new=str(project/'工作区/数据/处理后/创新组合实验/2025/episode_days.csv')
            if new in payload['identity']['inputs']:
                payload['identity']['inputs'][old]=payload['identity']['inputs'].pop(new)
                marker.write_text(json.dumps(payload),encoding='utf8')
            cfg=json.loads((Path(__file__).parent/'study.json').read_text(encoding='utf8'))
            migrate(project,out,cfg,task)
            corrected=json.loads(marker.read_text(encoding='utf8'))
            self.assertIn(new,corrected['identity']['inputs'])
            self.assertNotIn(old,corrected['identity']['inputs'])
            self.assertTrue(corrected['migration']['validation_checked'])

if __name__=='__main__':unittest.main()
