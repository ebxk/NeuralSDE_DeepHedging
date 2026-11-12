import importlib.util
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
from unittest.mock import patch
from types import SimpleNamespace


def routing_head_not_executed(*args):
    return [tf.keras.layers.Dense(64),tf.keras.layers.Dense(64),tf.keras.layers.Dense(7)]


class ReferenceMarketTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path=Path(__file__).with_name('reference_market.py');cls.exists=path.exists()
        if cls.exists:
            spec=importlib.util.spec_from_file_location('reference',path)
            cls.m=importlib.util.module_from_spec(spec);spec.loader.exec_module(cls.m)

    def setUp(self):self.assertTrue(self.exists,'fresh nonlinear NSDE reference module must exist')

    def test_original_dense64_architecture_is_required(self):
        source=(Path(__file__).resolve().parents[1]/'创新组合实验/market.py').read_text(encoding='utf-8-sig')
        source=source[source.index('def fit_member('):source.index('\ndef simulate(')]
        self.assertEqual(self.m.validate_original_head(source),[64,64,7])
        with self.assertRaises(ValueError):self.m.validate_original_head(source.replace('Dense(64,activation="tanh")','Dense(7)',1))

    def test_conditions_reader_does_not_load_cached_old_states(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'common.npz'
            np.savez(path,time=np.zeros((3,2),dtype='f'),warm=np.ones((3,20),dtype='f'),
                     strike=np.ones(3),premium=np.ones(3),gbm=np.array([object()],dtype=object),
                     real_xi=np.array([object()],dtype=object),real_cov=np.array([object()],dtype=object))
            conditions=self.m.load_shared_conditions(path)
            self.assertEqual(set(conditions),{'time','warm','strike','premium'})

    def test_closed_form_uses_first90percent_legal_training_only(self):
        dates=np.array(['2019-01-01','2019-01-02','2019-01-03','2019-01-04','2025-01-01'])
        rep={'dates':dates,'spot':np.exp(np.array([0.,.01,.04,.13,10.]))}
        cal=pd.Series(np.ones(5),index=dates)
        fitted=self.m.closed_spot_parameters(rep,cal,[2019,2020,2025],[])
        self.assertEqual(fitted['fit_transitions'],2)
        self.assertAlmostEqual(fitted['log_mu'],(.01+.03)/(2/365))
        self.assertLess(fitted['last_fit_date'],'2020')

    def test_financial_domain_and_same_two_four_shocks(self):
        rep={'xi':np.zeros((5,2)),'W':np.r_[np.eye(2),-np.eye(2)],'bounds':np.full(4,-1.)}
        self.assertEqual(self.m.check_factor_domain(rep),1.)
        rep['xi'][-1]=[2,0]
        with self.assertRaises(ValueError):self.m.check_factor_domain(rep)
        fine,coarse=self.m.paired_shocks(3,5,105)
        np.testing.assert_array_equal(fine,np.random.default_rng(105).normal(size=(3,4,4,3)))
        np.testing.assert_allclose(coarse,fine.reshape(3,4,2,2,3).sum(3)/np.sqrt(2))

    def test_reference_reads_new_encoder_before_any_market_fit(self):
        setters=SimpleNamespace(set_intra_op_parallelism_threads=lambda n:None,set_inter_op_parallelism_threads=lambda n:None)
        legacy=Path(__file__).resolve().parents[1]/'创新组合实验/market.py'
        market=SimpleNamespace(__file__=str(legacy),tf=SimpleNamespace(config=SimpleNamespace(threading=setters)),fit_member=routing_head_not_executed)
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);new=root/'representations/2025/original.npz';new.parent.mkdir(parents=True)
            np.savez(new,marker=np.array([1]))
            with patch.object(self.m,'OUT',root/'out'),patch.object(self.m,'PROJECT',root),patch.object(self.m,'base_factors',return_value=new),patch.dict('sys.modules',{'market':market}),patch.object(self.m.np,'load',wraps=np.load) as load:
                with self.assertRaises(FileNotFoundError):self.m.run(2025) # no fold config, stops before any fitting
                self.assertEqual(Path(load.call_args.args[0]),new)


if __name__=='__main__':unittest.main(verbosity=2)
