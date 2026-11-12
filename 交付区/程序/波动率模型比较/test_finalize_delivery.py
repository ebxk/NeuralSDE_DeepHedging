import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


class LocalDeliveryTests(unittest.TestCase):
    def test_direct_models_share_cost_and_method_columns(self):
        import pandas as pd
        with tempfile.TemporaryDirectory() as d:
            out=Path(d);folder=out/'direct_dynamic/2025/heston';folder.mkdir(parents=True);(out/'evaluation').mkdir()
            pd.DataFrame([dict(model='heston',hedge='delta',bp=5,execution='same_close',paths=2,
                rmse=.1,cvar=.2,mean_cost=.01,mean_turnover=2.)]).to_csv(folder/'groups.csv',index=False)
            with patch.object(self.m,'OUT',out),patch.object(self.m,'CONFIG',{'folds':[(2020,2024,2025)],'models':[]}):
                self.m.collect_aggregate_diagnostics()
            row=pd.read_csv(out/'evaluation/direct_summary.csv').iloc[0]
            self.assertEqual(row['method'],'delta');self.assertEqual(row['cost_bp'],5)
            self.assertAlmostEqual(row['cost'],.01);self.assertAlmostEqual(row['turnover'],2.)

    @classmethod
    def setUpClass(cls):
        path=Path(__file__).with_name('finalize_delivery.py');spec=importlib.util.spec_from_file_location('delivery',path)
        cls.m=importlib.util.module_from_spec(spec);spec.loader.exec_module(cls.m)

    def test_acceptance_gate_precedes_financial_reload_or_copy(self):
        with patch.object(self.m,'require_delivery_manifest',side_effect=ValueError('not accepted')),patch.object(self.m.shutil,'copy2') as copy:
            with self.assertRaises(ValueError):self.m.run()
            copy.assert_not_called()

    def test_private_market_factor_and_weight_artifacts_only_local_params(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);out=root/'out';params=root/'交付区/参数与依赖/波动率模型比较';params.mkdir(parents=True)
            names=['policies/job.pt','markets/2025/nsde/market.weights.h5','markets/2025/nsde/simulation.npz',
                   'representations/2025/original.npz','acceptance_manifest.json','registry.json','account_acceptance.json',
                   'data/shared_surfaces.npz','data/clean_base/status.json']
            for name in names:
                p=out/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'private fixture')
            with patch.object(self.m,'OUT',out):self.m.copy_local_parameters(params)
            self.assertTrue((params/'weights/job.pt').exists())
            self.assertFalse((params/'markets/2025/nsde/simulation.npz').exists())
            import json
            refs=json.loads((params/'local_artifact_references.json').read_text(encoding='utf8'))
            self.assertTrue(any(r['path'].endswith('simulation.npz') for r in refs['artifacts']))
            self.assertTrue((params/'representations/2025/original.npz').exists())
            self.assertFalse((root/'交付区/结果').exists())
            for name in ['surface_comparison.csv','surface_status.json','numerical_diagnostics.csv','compute_costs.csv','numerical_status.json']:
                self.assertIn(name,self.m.AGGREGATE_RESULTS)

    def test_direct_nsde_tail_and_turnover_are_aggregated(self):
        import pandas as pd
        with tempfile.TemporaryDirectory() as d:
            out=Path(d);folder=out/'direct_nsde/2025';folder.mkdir(parents=True);(out/'evaluation').mkdir()
            pd.DataFrame(dict(method=['delta']*2,cost_bp=[5]*2,execution=['same_close']*2,
                pnl=[-.2,.1],cost=[.01,.02],clip_decisions=[1,0],turnover=[2.,4.])).to_csv(folder/'pnl.csv',index=False)
            with patch.object(self.m,'OUT',out),patch.object(self.m,'CONFIG',{'folds':[(2020,2024,2025)],'models':[]}):
                self.m.collect_aggregate_diagnostics()
            row=pd.read_csv(out/'evaluation/direct_summary.csv').iloc[0]
            self.assertAlmostEqual(row['cvar'],.2);self.assertAlmostEqual(row['std'],.15)
            self.assertAlmostEqual(row['turnover'],3.)


if __name__=='__main__':unittest.main(verbosity=2)

