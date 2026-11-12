import importlib.util
from pathlib import Path
import tempfile
import unittest
import json
import numpy as np
from unittest.mock import patch


class ExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path=Path(__file__).with_name('generator_export.py');cls.exists=path.exists()
        if cls.exists:
            spec=importlib.util.spec_from_file_location('exporter',path)
            cls.m=importlib.util.module_from_spec(spec);spec.loader.exec_module(cls.m)

    def setUp(self):
        self.assertTrue(self.exists,'generator exporter must exist')

    def fixture(self,root):
        import pandas as pd
        old=root/'工作区/数据/处理后/创新组合实验/2025';(old/'C1').mkdir(parents=True)
        cfg={'split_ends':[2022,2023,2025],'origin_seed':104,'shock_seed':105,'simulation_train':4,'simulation_valid':2}
        (old/'config.json').write_text(json.dumps(cfg),encoding='utf8')
        rows=[]
        for ep,date,spot,strike in [(10,'2020-01-01',2.,2.),(2,'2020-02-01',3.,3.1)]:
            for day in range(4):
                rows.append(dict(episode=ep,split_end=2022,date=str(np.datetime64(date)+day),spot=spot,strike=strike,option_close=.1))
        episodes=pd.DataFrame(rows);episodes.to_csv(old/'episode_days.csv',index=False)
        selected=np.random.default_rng(104).integers(2,size=6)
        starts=np.array([3.,2.])[selected];strikes=np.array([3.1,2.])[selected]
        times=np.broadcast_to(np.arange(4)/365,(6,4)).astype('f').copy()
        common={'time':times,'warm':np.full((6,20),.01,dtype='f'),'strike':strikes,'premium':np.full(6,.1),'gbm':np.broadcast_to(starts[:,None],(6,4)).copy()}
        np.savez(old/'C1/common.npz',**common)
        ts=np.repeat([14/365,30/365,60/365,90/365],5);ks=np.tile([.9,.95,1.,1.05,1.1],4)
        training_dates=np.arange(np.datetime64('2020-01-01'),np.datetime64('2020-02-05')).astype(str)
        out=root/'out';basis=out/'representations/2025';basis.mkdir(parents=True)
        np.savez(basis/'original.npz',grid_t=ts,grid_k=ks,dates=training_dates)
        np.savez(old/'C1.npz',grid_t=np.array(['old_basis_must_not_be_used']))
        raw=root/'工作区/数据/原始数据/50ETF';raw.mkdir(parents=True)
        pd.DataFrame({'交易日期':training_dates,'收盘价(元)':2*np.exp(np.arange(len(training_dates))*.0001)}).to_csv(
            raw/'510050_华夏上证50ETF.csv',index=False,encoding='gb18030')
        out=root/'out';dest=out/'markets/2025/sabr';dest.mkdir(parents=True)
        model={'nu':.2,'rho':-.4,'diagnostics':{'valid':True}}
        for name,value in [('global.json',{'model':model}),('p_parameters.json',{'state_log_drift':0.,'spot_drift':.03}),('status.json',{'complete':True})]:
            (dest/name).write_text(json.dumps(value),encoding='utf8')
        pd.DataFrame({'date':['2020-01-01','2020-02-01'],'state':[.2,.3]}).to_csv(dest/'states.csv',index=False)
        return out,common,selected

    def test_actual_small_sabr_atomic_ready_and_shared_inputs(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);out,common,selected=self.fixture(root)
            cfg=dict(self.m.CONFIG,simulation_train=4,simulation_valid=2)
            with patch.object(self.m,'OUT',out),patch.object(self.m,'PROJECT',root),patch.object(self.m,'CONFIG',cfg),patch.object(self.m,'base_factors',side_effect=lambda end:out/f'representations/{end}/original.npz'):
                receipt=self.m.export_generator(2025,'sabr')
            ready=out/'inputs/2025/sim_sabr.ready.json'
            self.assertTrue(ready.exists());self.assertTrue(receipt['complete'])
            data=dict(np.load(out/'inputs/2025/sim_sabr.npz'))
            self.assertEqual(data['f'].shape,(6,3,7))
            self.assertEqual(data['f'].shape[-1]+1,8) # prior position is the eighth slot
            np.testing.assert_array_equal(data['spot'][:,0],common['gbm'][:,0])
            np.testing.assert_array_equal(data['strike_end'],common['strike'].astype('f'))
            np.testing.assert_allclose(data['f'][:,0,2],4*np.sqrt(252)*.01,rtol=1e-6)
            expected_alpha=np.array([.3,.2])[selected]
            from stochastic_volatility import price_sabr
            from linear_market import shared_summary
            ts=np.array([14/365,30/365]);ks=np.array([.9,.95,1.,1.05,1.1])
            expected=shared_summary(price_sabr(ts[None,:,None],ks[None,None,:],expected_alpha[:,None,None],.2,-.4),ts,ks)
            np.testing.assert_allclose(data['f'][:,0,3:],expected,atol=1e-6)
            self.assertEqual(receipt['input_sha256'],self.m.digest(out/'inputs/2025/sim_sabr.npz'))
            sidecar=out/'inputs/2025'/receipt['state_sidecar']['file']
            self.assertEqual(receipt['state_sidecar']['sha256'],self.m.digest(sidecar))
            side=dict(np.load(sidecar))
            self.assertEqual(side['latent'].shape,(6,4))
            np.testing.assert_allclose(side['latent'][:,0],expected_alpha)
            np.testing.assert_array_equal(side['episode_id'],np.array([2,10])[selected])
            self.assertEqual(list((out/'inputs/2025').glob('*.tmp')),[])

    def test_shocks_pair_and_full_truncation_count(self):
        rng=np.random.default_rng(105)
        z4,z2=self.m._shocks(rng,3,4)
        np.testing.assert_array_equal(z4,np.random.default_rng(105).normal(size=(3,3,4,3)))
        np.testing.assert_allclose(z2,z4.reshape(3,3,2,2,3).sum(3)/np.sqrt(2))
        z=np.zeros((1,1,2,3));z[...,1]=-100
        _,state,diag=self.m._simulate_stochastic('heston',{'volvol':1.,'rho':0.},
            {'state_kappa':1.,'state_theta':.04,'spot_drift':0.},np.array([1.]),np.array([.04]),np.array([[0.,.1]]),z)
        self.assertGreater(diag['negative_variance_truncations'],0)
        self.assertTrue(np.all(state>=0))

    def test_failure_receipt_no_ready(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);out,_,_=self.fixture(root)
            cfg=dict(self.m.CONFIG,simulation_train=4,simulation_valid=2)
            with patch.object(self.m,'OUT',out),patch.object(self.m,'PROJECT',root),patch.object(self.m,'CONFIG',cfg),patch.object(self.m,'base_factors',side_effect=lambda end:out/f'representations/{end}/original.npz'):
                with self.assertRaises(FileNotFoundError):self.m.export_generator(2025,'heston')
            failure=json.loads((out/'inputs/2025/sim_heston.failure.json').read_text(encoding='utf8'))
            self.assertFalse(failure['complete'])
            self.assertFalse((out/'inputs/2025/sim_heston.ready.json').exists())

    def test_linear_and_localvol_real_pde_exports(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);out,common,_=self.fixture(root)
            directory=out/'markets/2025/svi_linear';directory.mkdir(parents=True)
            (directory/'market.json').write_text(json.dumps({'status':'complete'}),encoding='utf8')
            from linear_market import shared_summary
            from volatility_surfaces import bs_call
            ts=np.array([14/365,30/365]);ks=np.array([.9,.95,1.,1.05,1.1])
            summary=shared_summary(bs_call(ts[:,None],ks[None,:],.2),ts,ks)
            np.savez(directory/'simulation.npz',spot=common['gbm'],summary=np.broadcast_to(summary,(6,4,4)),xi=np.zeros((6,4,2)),
                **{k:common[k] for k in ['time','warm','strike','premium']})
            surface={'kind':'essvi','times':[.05,.2,.4],'parameters':[[.04*t,0.,0.] for t in [.05,.2,.4]],'diagnostics':{'valid':True}}
            for day in ['2020-01-01','2020-02-01']:
                dest=out/f'surfaces/essvi/{day}';dest.mkdir(parents=True)
                (dest/'full.json').write_text(json.dumps({'status':'complete','model':surface}),encoding='utf8')
            cfg=dict(self.m.CONFIG,simulation_train=4,simulation_valid=2)
            with patch.object(self.m,'OUT',out),patch.object(self.m,'PROJECT',root),patch.object(self.m,'CONFIG',cfg),patch.object(self.m,'base_factors',side_effect=lambda end:out/f'representations/{end}/original.npz'):
                linear=self.m.export_generator(2025,'svi_linear')
                lv=self.m.export_generator(2025,'localvol')
            self.assertTrue(linear['complete']);self.assertTrue(lv['complete'])
            data=dict(np.load(out/'inputs/2025/sim_localvol.npz'))
            np.testing.assert_allclose(data['f'][:,0,3],summary[0],atol=.005)
            self.assertLessEqual(lv['diagnostics']['localvol']['fixed_query_interpolation_max_error'],1e-4)
            self.assertLessEqual(lv['diagnostics']['localvol']['fixed_query_iv_max_error'],.005)

    def test_local_origin_fallback_past_only_then_rv20(self):
        import pandas as pd
        with tempfile.TemporaryDirectory() as d:
            out=Path(d);surface={'kind':'essvi','times':[.1,.3],
                'parameters':[[.004,0,0],[.012,0,0]],'diagnostics':{'valid':True}}
            for day in ['2020-01-30','2020-02-02']:
                path=out/f'surfaces/essvi/{day}';path.mkdir(parents=True)
                (path/'full.json').write_text(json.dumps({'status':'complete','model':surface}),encoding='utf8')
            calendar=pd.Series(np.ones(10),index=pd.date_range('2020-01-27',periods=10).strftime('%Y-%m-%d'))
            with patch.object(self.m,'OUT',out):
                _,meta=self.m._local_origin_surface('2020-02-01',calendar,np.full(20,.01),[.1,.3],lambda p:p)
                self.assertEqual(meta['source_date'],'2020-01-30');self.assertEqual(meta['stale_days'],2)
                (out/'surfaces/essvi/2020-01-30/full.json').unlink()
                flat,meta=self.m._local_origin_surface('2020-02-01',calendar,np.full(20,.01),[.1,.3],lambda p:p)
                self.assertEqual(meta['fallback_source'],'origin_past20_RV_flat')
                self.assertAlmostEqual(flat['parameters'][0][0],252*.01**2*.1)

    def test_fresh_nsde_export_requires_nonlinear_refit_identity(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);out,common,_=self.fixture(root)
            directory=out/'markets/2025/nsde';directory.mkdir(parents=True)
            from linear_market import shared_summary
            from volatility_surfaces import bs_call
            ts=np.array([14/365,30/365]);ks=np.array([.9,.95,1.,1.05,1.1])
            summary=shared_summary(bs_call(ts[:,None],ks[None,:],.2),ts,ks)
            np.savez(directory/'simulation.npz',spot=common['gbm'],summary=np.broadcast_to(summary,(6,4,4)),xi=np.zeros((6,4,2)),
                **{k:common[k] for k in ['time','warm','strike','premium']})
            status={'status':'complete','identity':{'head_widths':[64,64,7],'market_weights_reused':False,
                    'source':self.m.digest(Path(__file__).with_name('reference_market.py'))},
                    'output_sha256':{'simulation.npz':self.m.digest(directory/'simulation.npz')}}
            (directory/'market.json').write_text(json.dumps(status),encoding='utf8')
            cfg=dict(self.m.CONFIG,simulation_train=4,simulation_valid=2)
            with patch.object(self.m,'OUT',out),patch.object(self.m,'PROJECT',root),patch.object(self.m,'CONFIG',cfg),patch.object(self.m,'base_factors',side_effect=lambda end:out/f'representations/{end}/original.npz'):
                receipt=self.m.export_generator(2025,'nsde')
                self.assertTrue(receipt['complete'])
                self.assertIn('xi',dict(np.load(out/'inputs/2025'/receipt['state_sidecar']['file'])))
                status['identity']['head_widths']=[7]
                (directory/'market.json').write_text(json.dumps(status),encoding='utf8')
                with self.assertRaises(ValueError):self.m.export_generator(2025,'nsde')


if __name__=='__main__':unittest.main(verbosity=2)
