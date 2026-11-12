import importlib.util
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
import tempfile
import json
from unittest.mock import patch


class FactorAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path=Path(__file__).with_name('factor_quote_audit.py');cls.exists=path.exists()
        if cls.exists:
            spec=importlib.util.spec_from_file_location('audit',path)
            cls.m=importlib.util.module_from_spec(spec);spec.loader.exec_module(cls.m)

    def setUp(self):self.assertTrue(self.exists,'factor quote audit module must exist')

    def fixture(self):
        from volatility_surfaces import bs_call
        ts=np.repeat([14/365,60/365],5);ks=np.tile([.9,.95,1.,1.05,1.1],2)
        center=bs_call(ts,ks,.2);vega=(bs_call(ts,ks,.25)-center)/.05
        basis=np.stack([vega,vega*(ks-1)*10])
        rep=dict(grid_t=ts,grid_k=ks,center=center,basis=basis,W=np.r_[np.eye(2),-np.eye(2)],bounds=np.full(4,-.03),
                 dates=np.array(['2019-01-01','2019-01-02','2019-01-03','2025-01-02']),
                 xi=np.array([[.001,0],[0,.001],[-.001,-.001],[999,999]]))
        calls=center+np.array([.01,-.005])@basis
        q=pd.DataFrame(dict(date='2025-01-02',ttm=ts,k=ks,price=calls,holdout=np.tile([False,True,False,True,False],2)))
        return rep,q

    def test_masked_fit_never_uses_hidden_prices_or_current_full_xi(self):
        rep,q=self.fixture();starts=self.m.training_starts(rep,'2020',20261112)
        self.assertEqual(starts.shape,(5,2));self.assertLess(np.max(np.abs(starts)),.01)
        fit=self.m.fit_factor_view(rep,q,starts,'heldout')
        q.loc[q.holdout,'price']=.8;rep['xi'][-1]=[-999,-999]
        changed=self.m.fit_factor_view(rep,q,self.m.training_starts(rep,'2020',20261112),'heldout')
        np.testing.assert_array_equal(fit['xi'],changed['xi'])
        np.testing.assert_allclose(fit['xi'],[.01,-.005],atol=1e-5)
        self.assertTrue(fit['valid']);self.assertGreater(changed['heldout_rmse'],fit['heldout_rmse'])

    def test_disjoint_quote_metrics_and_duplicate_mask_leak_rejected(self):
        rep,q=self.fixture();starts=self.m.training_starts(rep,'2020',20261112)
        fit=self.m.fit_factor_view(rep,q,starts,'full')
        self.assertEqual(fit['visible_n'],6);self.assertEqual(fit['heldout_n'],4)
        self.assertEqual(fit['fit_n'],10)
        masked=self.m.fit_factor_view(rep,q,starts,'heldout')
        self.assertEqual(masked['fit_n'],6)
        duplicate=q.iloc[[0]].copy();duplicate['holdout']=True
        with self.assertRaises(ValueError):self.m.clean_quotes(pd.concat([q,duplicate],ignore_index=True))

    def test_original_visible_grid_and_l1_repair(self):
        from volatility_surfaces import bs_call
        rep,q=self.fixture();grid_t=rep['grid_t'];grid_k=rep['grid_k']
        a=np.r_[np.eye(10),-np.eye(10)];b=np.r_[np.maximum(1-grid_k,0),-np.ones(10)]
        fixed,diag=self.m.repair_grid(np.full(10,-.1),a,b)
        self.assertGreaterEqual(np.min(a@fixed-b),-1e-8);self.assertTrue(diag['repaired'])
        result=self.m.fit_original_view(q,grid_t,grid_k,a,b,'heldout')
        q.loc[q.holdout,'price']=.8
        changed=self.m.fit_original_view(q,grid_t,grid_k,a,b,'heldout')
        np.testing.assert_array_equal(result['grid'],changed['grid'])
        self.assertTrue(result['valid'])

    def test_runner_end_cutoff_and_missing_basis_receipt(self):
        rep,q=self.fixture()
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);out=root/'out';old=root/'工作区/数据/处理后/创新组合实验/2025';old.mkdir(parents=True)
            (old/'config.json').write_text(json.dumps({'split_ends':[2019,2020,2025]}),encoding='utf8')
            newbase=out/'representations/2025';newbase.mkdir(parents=True)
            np.savez(newbase/'original.npz',**rep)
            np.savez(old/'C1.npz',xi=np.array([object()],dtype=object))
            data=out/'data';data.mkdir(parents=True)
            train=q.copy();train['date']='2019-02-01'
            future=q.copy();future['date']='2026-01-01';future['price']=-1
            pd.concat([train,q,future],ignore_index=True).to_csv(data/'quotes.csv',index=False)
            with patch.object(self.m,'OUT',out),patch.object(self.m,'PROJECT',root),patch.object(self.m,'base_factors',side_effect=lambda end:out/f'representations/{end}/original.npz'):
                result=self.m.run(2025,'nsde')
                self.assertTrue(result['complete']);self.assertEqual(result['dates'],2)
                summary=json.loads((out/'factor_quote_audit/2025/nsde/summary.json').read_text(encoding='utf8'))
                self.assertEqual({r['split'] for r in summary},{'training','historical_evaluation'})
                with self.assertRaises(FileNotFoundError):self.m.run(2025,'svi')
            receipt=json.loads((out/'factor_quote_audit/2025/svi/status.json').read_text(encoding='utf8'))
            self.assertFalse(receipt['complete']);self.assertEqual(receipt['status'],'failed')


if __name__=='__main__':unittest.main(verbosity=2)
