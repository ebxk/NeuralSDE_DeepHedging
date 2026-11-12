import importlib.util
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
from unittest.mock import patch
import json


class CleanBaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path=Path(__file__).with_name('prepare_clean_base.py');cls.exists=path.exists()
        if cls.exists:
            spec=importlib.util.spec_from_file_location('cleanbase',path)
            cls.m=importlib.util.module_from_spec(spec);spec.loader.exec_module(cls.m)

    def setUp(self):self.assertTrue(self.exists,'common clean baseline builder must exist')

    def test_fixed_inputs_do_not_read_old_calls_or_states(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'old.npz';bad=np.array([object()],dtype=object)
            np.savez(p,A=np.eye(20),b=np.zeros(20),grid_t=np.repeat([14,30,60,90],5)/365,
                     grid_k=np.tile([.9,.95,1.,1.05,1.1],4),weights=np.ones(20),calls=bad,xi=bad,basis=bad)
            self.assertEqual(set(self.m.fixed_constraints(p)),{'A','b','grid_t','grid_k'})
            np.testing.assert_array_equal(self.m.frozen_weights(p),np.ones(20))

    def test_common_quote_bounds_and_financial_repair(self):
        from volatility_surfaces import bs_call
        ts=np.repeat([14,30,60,90],5)/365;ks=np.tile([.9,.95,1.,1.05,1.1],4)
        q=pd.DataFrame(dict(date='2020-01-02',ttm=ts,k=ks,price=bs_call(ts,ks,.2),holdout=False,spot=2.))
        fixed={'grid_t':ts,'grid_k':ks,'A':np.r_[np.eye(20),-np.eye(20)],'b':np.r_[np.maximum(1-ks,0),-np.ones(20)]}
        result=self.m.rebuild_day(q,fixed)
        np.testing.assert_allclose(result['calls'],q.price,atol=1e-10)
        self.assertGreaterEqual(np.min(fixed['A']@result['calls']-fixed['b']),-1e-7)
        bad=q.copy();bad.loc[0,'price']=-.1
        with self.assertRaises(ValueError):self.m.rebuild_day(bad,fixed)

    def test_encoder_mask_excludes_future_and_one_shrink_only(self):
        data={'dates':np.array(['2019-01-01','2019-01-02','2025-01-01']),
              'calls':np.arange(60,dtype=float).reshape(3,20),'A':np.eye(20),'b':np.full(20,-1e6)}
        seen=[]
        def decoder(d,mask,weights):
            seen.append((mask.copy(),weights.copy()))
            center=d['calls'][mask].mean(0)
            return dict(center=center,basis=np.zeros((2,20)),xi=np.zeros((3,2)),W=np.eye(2),bounds=np.full(2,-1.))
        a,_=self.m.encode_frozen_basis(data,np.ones(20),2019,decoder)
        data['calls'][-1]=999999
        b,_=self.m.encode_frozen_basis(data,np.ones(20),2019,decoder)
        np.testing.assert_array_equal(a['center'],b['center'])
        np.testing.assert_array_equal(seen[0][0],[True,True,False])
        attempts=[]
        def retry(d,mask,w):
            attempts.append(w.copy())
            if len(attempts)==1:raise ValueError('thin geometry')
            return decoder(d,mask,w)
        _,shrunk=self.m.encode_frozen_basis(data,np.full(20,2.),2019,retry)
        self.assertTrue(shrunk);self.assertEqual(len(attempts),2)
        np.testing.assert_array_equal(attempts[1],np.full(20,1.5))

    def test_encoder_child_does_not_inherit_other_python_home(self):
        with tempfile.TemporaryDirectory() as d:
            base=Path(d)/'original.npz';base.with_suffix('.json').write_text(json.dumps({'complete':True}),encoding='utf8')
            with patch.object(self.m,'base_factors',return_value=base),patch.object(self.m.subprocess,'run') as child,patch.dict('os.environ',{'PYTHONHOME':'wrong-3.13-home','PYTHONPATH':'wrong-3.13-path'}):
                self.m.run_encoder(2025)
            env=child.call_args.kwargs['env']
            self.assertNotIn('PYTHONHOME',env);self.assertNotIn('PYTHONPATH',env)
            self.assertIn('--encoder-worker',child.call_args.args[0])


if __name__=='__main__':unittest.main(verbosity=2)
