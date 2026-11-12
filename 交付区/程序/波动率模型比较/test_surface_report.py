import importlib.util
from pathlib import Path
import tempfile
import json
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd


class SurfaceReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path=Path(__file__).with_name('surface_report.py');cls.exists=path.exists()
        if cls.exists:
            spec=importlib.util.spec_from_file_location('report',path)
            cls.m=importlib.util.module_from_spec(spec);spec.loader.exec_module(cls.m)

    def setUp(self):self.assertTrue(self.exists,'three-layer static report must exist')

    def quotes(self):
        from volatility_surfaces import bs_call
        t=np.array([.02,.1,.3,.5]);k=np.array([1.,.95,1.2,1.])
        return pd.DataFrame(dict(date='2025-01-02',ttm=t,k=k,price=bs_call(t,k,.2),holdout=[False,True,True,False],code=['a','b','c','d']))

    def test_masked_native_reprices_held_model_not_full_hidden_metric(self):
        from volatility_surfaces import bs_call
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);directory=root/'surfaces/ssvi/2025-01-02';directory.mkdir(parents=True)
            for view,vol in [('full',.2),('heldout',.3)]:
                record={'status':'complete','identity':{'kind':'ssvi','day':'2025-01-02','view':view},'model':{'kind':'ssvi','times':[.1,.5],'parameters':[vol*vol*.1,vol*vol*.5],
                    'rho':0.,'eta':0.,'diagnostics':{'valid':True}},'metrics':{'heldout':{'n':2,'rmse':0.}}}
                (directory/f'{view}.json').write_text(json.dumps(record),encoding='utf8')
            q=self.quotes();reader=self.m.Reader()
            with patch.object(self.m,'OUT',root):p=self.m.native_prediction(reader,'ssvi',q,'heldout')
            records,_=self.m.score_day(2025,'historical_evaluation','heldout',q,{'ssvi_native':p})
            allrow=next(r for r in records if r['region']=='all')
            expected=bs_call(q.ttm.to_numpy(),q.k.to_numpy(),.3)-q.price.to_numpy()
            self.assertEqual(allrow['price_n'],2)
            self.assertAlmostEqual(allrow['price_sse'],float(np.sum(expected[q.holdout]**2)))
            self.assertGreater(allrow['price_sse'],0)

    def test_weighted_sse_and_common_qualified_dates(self):
        q=self.quotes();p=q.price.to_numpy();records=[]
        for day,valid in [('2025-01-02',True),('2025-01-03',False)]:
            q=q.copy();q['date']=day
            rows,_=self.m.score_day(2025,'historical_evaluation','full',q,{
                'a':{'prices':p+.01,'valid':True,'status':'complete'},
                'b':{'prices':p+.02,'valid':valid,'status':'complete' if valid else 'numerical_rejected'}})
            records.extend(rows)
        comparison=self.m.aggregate_daily(records)
        a=comparison[(comparison.method=='a')&(comparison.region=='all')].iloc[0]
        b=comparison[(comparison.method=='b')&(comparison.region=='all')].iloc[0]
        self.assertAlmostEqual(a.price_rmse,.01);self.assertAlmostEqual(b.price_rmse,.02)
        self.assertEqual(a.matched_dates,1);self.assertEqual(b.matched_dates,1)
        self.assertEqual(a.matched_price_n,4);self.assertEqual(b.matched_price_n,4)
        # Changing quote counts must use SSE/N rather than the mean day RMSE.
        c=self.m.aggregate_daily([dict(records[0],price_sse=1.,price_n=1,valid=True),dict(records[0],price_sse=81.,price_n=9,valid=True)]).iloc[0]
        self.assertAlmostEqual(c.price_rmse,np.sqrt(82/10))

    def test_independent_iv_vega_and_domain_masks(self):
        q=self.quotes();r=self.m.independent_iv(q.ttm.to_numpy(),q.k.to_numpy(),q.price.to_numpy())
        np.testing.assert_allclose(r['iv'],.2,atol=1e-8)
        bad=self.m.independent_iv(np.array([.1]),np.array([1.]),np.array([1.1]))
        self.assertFalse(bad['identifiable'][0])
        masks=self.m.region_masks(q)
        self.assertTrue(masks['short'][0]);self.assertTrue(masks['wings'][2]);self.assertTrue(masks['out_of_tenor'][3])

    def test_missing_sources_write_failed_not_fake_complete(self):
        with tempfile.TemporaryDirectory() as d:
            with patch.object(self.m,'OUT',Path(d)):status=self.m.build_report()
            self.assertFalse(status['complete']);self.assertTrue(status['source_errors'])
            self.assertTrue((Path(d)/'evaluation/surface_status.json').exists())

    def test_blocked_method_has_no_fake_matched_values_and_low_vega_excluded(self):
        from volatility_surfaces import bs_call
        q=pd.DataFrame(dict(date=['2025-01-02'],ttm=[.05],k=[1.25],price=bs_call(.05,1.25,.2).reshape(1),holdout=[True]))
        records,_=self.m.score_day(2025,'historical_evaluation','heldout',q,{
            'a':{'prices':bs_call(.05,1.25,.21).reshape(1),'valid':True,'status':'complete'},
            'blocked':{'prices':None,'valid':False,'status':'blocked','blocked':True}})
        comparison=self.m.aggregate_daily(records)
        a=comparison[(comparison.method=='a')&(comparison.region=='all')].iloc[0]
        b=comparison[(comparison.method=='blocked')&(comparison.region=='all')].iloc[0]
        self.assertEqual(a.iv_n,0);self.assertEqual(a.low_vega_n,1)
        self.assertEqual(b.matched_price_n,0);self.assertTrue(pd.isna(b.matched_price_rmse))

    def test_global_and_factor_providers_use_separate_saved_masked_states(self):
        from volatility_surfaces import bs_call
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);global_dir=root/'markets/2025/sabr';global_dir.mkdir(parents=True)
            (global_dir/'global.json').write_text(json.dumps({'model':{'nu':0.,'rho':0.,'diagnostics':{'valid':True}}}),encoding='utf8')
            (global_dir/'status.json').write_text(json.dumps({'complete':True}),encoding='utf8')
            pd.DataFrame({'date':['2025-01-02'],'state':[.2],'masked_state':[.3]}).to_csv(global_dir/'states.csv',index=False)
            directory=root/'factor_quote_audit/2025/nsde';directory.mkdir(parents=True)
            (directory/'status.json').write_text(json.dumps({'status':'complete'}),encoding='utf8')
            pd.DataFrame({'date':['2025-01-02']*2,'view':['full','heldout'],'valid':[True,True]}).to_csv(directory/'daily.csv',index=False)
            records=[{'date':'2025-01-02','view':view,'valid':True,'xi':state} for view,state in [('full',[0,0]),('heldout',[1,0])]]
            (directory/'daily.jsonl').write_text('\n'.join(json.dumps(r) for r in records),encoding='utf8')
            ts=np.repeat([.1,.5],3);ks=np.tile([.8,1.,1.3],2);center=bs_call(ts,ks,.2)
            rep=root/'original.npz';np.savez(rep,grid_t=ts,grid_k=ks,center=center,basis=np.stack([bs_call(ts,ks,.3)-center,np.zeros(6)]))
            q=self.quotes();reader=self.m.Reader()
            with patch.object(self.m,'OUT',root),patch.object(self.m,'base_factors',return_value=rep):
                global_provider=self.m.Provider(reader,2025,'sabr');factor_provider=self.m.Provider(reader,2025,'nsde_factor')
                actual=global_provider.predict(q,'heldout')['prices']
                np.testing.assert_allclose(actual,bs_call(q.ttm.to_numpy(),q.k.to_numpy(),.3))
                self.assertFalse(factor_provider.failure)
                self.assertGreater(np.max(np.abs(factor_provider.predict(q,'heldout')['prices']-factor_provider.predict(q,'full')['prices'])),.001)


if __name__=='__main__':unittest.main(verbosity=2)
