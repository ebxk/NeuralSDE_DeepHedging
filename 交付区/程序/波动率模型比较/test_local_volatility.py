import importlib.util
from pathlib import Path
import unittest
import numpy as np


class LocalVolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path=Path(__file__).with_name('local_volatility.py');cls.exists=path.exists()
        if cls.exists:
            spec=importlib.util.spec_from_file_location('localvol',path)
            cls.m=importlib.util.module_from_spec(spec);spec.loader.exec_module(cls.m)

    def setUp(self):
        self.assertTrue(self.exists,'local volatility module must exist')

    @staticmethod
    def flat(sigma=.2):
        return {'kind':'essvi','times':[.05,.2,.4],
                'parameters':[[sigma*sigma*t,0.,0.] for t in [.05,.2,.4]]}

    def test_dupire_bs_limit_and_no_silent_clip(self):
        table=self.m.build_table(self.flat())
        np.testing.assert_allclose(table['local_variance'],.04,atol=2e-6)
        self.assertTrue(table['diagnostics']['valid'])
        self.assertEqual(table['diagnostics']['clipped_points'],0)
        self.assertGreater(table['diagnostics']['stable_branch_fraction'],0)
        self.assertEqual(table['diagnostics']['days90_120']['nodes'],31)
        self.assertAlmostEqual(table['diagnostics']['days90_120']['variance_mean'],.04,places=5)

    def test_conditional_bs_price_delta_and_cache_identity(self):
        table=self.m.build_table(self.flat())
        result=self.m.conditional_prices(table,.04,1.05,.15,[.9,1.,1.1])
        from scipy.special import ndtr
        k=np.array([.9,1.,1.1]);s=1.05;t=.15
        d1=np.log(s/k)/(.2*np.sqrt(t))+.2*np.sqrt(t)/2
        expected=s*ndtr(d1)-k*ndtr(d1-.2*np.sqrt(t))
        np.testing.assert_allclose(result['prices'],expected,atol=1e-3)
        np.testing.assert_allclose(result['delta'],ndtr(d1),atol=.003)
        h=1e-4
        up=self.m.conditional_prices(table,.04,s+h,.15,k)['prices']
        down=self.m.conditional_prices(table,.04,s-h,.15,k)['prices']
        np.testing.assert_allclose(result['delta'],(up-down)/(2*h),atol=.003)
        again=self.m.conditional_prices(self.m.build_table(self.flat()),.04,s,.15,k)
        self.assertEqual(result['cache_key'],again['cache_key'])
        other=self.m.conditional_prices(self.m.build_table(self.flat(.3)),.04,s,.15,k)
        self.assertNotEqual(result['cache_key'],other['cache_key'])
        self.assertTrue(np.all(other['prices']>result['prices']))

    def test_negative_calendar_variance_rejected(self):
        bad=self.flat();bad['parameters'][1][0]=.0001
        with self.assertRaises(ValueError):
            self.m.build_table(bad)

    def test_simulation_moments_and_shared_shocks(self):
        table=self.m.build_table(self.flat());rng=np.random.default_rng(5)
        z=rng.standard_normal((10000,30,2));z=np.concatenate([z,-z])
        times=np.linspace(0,30/365,31)
        p=self.m.simulate_p(table,1.,times,z,spot_drift=.03)
        self.assertTrue(p['diagnostics']['valid'])
        self.assertAlmostEqual(np.mean(p['spot'][:,-1]),np.exp(.03*30/365),delta=.001)
        self.assertEqual(p['spot'].shape,(20000,31))
        np.testing.assert_array_equal(p['spot'][:2],self.m.simulate_p(table,1.,times,z[:2],.03)['spot'])

    def test_no_future_curve_and_horizon_guard(self):
        table=self.m.build_table(self.flat())
        self.assertNotIn('future_surface',table)
        with self.assertRaises(ValueError):
            self.m.conditional_prices(table,.3,1.,.1,[1.])

    def test_expiry_exact_intrinsic_and_curved_origin_recovery(self):
        surface={'kind':'essvi','times':[.05,.15,.4],
                 'parameters':[[.004,-.6,.05],[.012,-.4,.08],[.032,-.3,.12]]}
        table=self.m.build_table(surface)
        expiry=self.m.conditional_prices(table,.1,1.003,0,[1.,1.003,1.01])
        np.testing.assert_allclose(expiry['prices'],[.003,0,0],atol=1e-12)
        from volatility_surfaces import price_surface
        strikes=np.array([.9,1.,1.1])
        for tenor in [.05,.15,.3]:
            result=self.m.conditional_prices(table,0,1.,tenor,strikes)
            np.testing.assert_allclose(result['prices'],price_surface(surface,tenor,strikes),atol=1e-3)


if __name__=='__main__':
    unittest.main(verbosity=2)
