"""Small synthetic checks; no original quotes or full experiment required."""
import importlib.util
import json
from pathlib import Path
import unittest
import numpy as np
from unittest.mock import patch
from scipy.optimize import OptimizeResult


class SurfaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).with_name('volatility_surfaces.py')
        cls.module_exists = path.exists()
        if cls.module_exists:
            spec = importlib.util.spec_from_file_location('surfaces', path)
            cls.v = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(cls.v)

    def setUp(self):
        self.assertTrue(self.module_exists, 'surface module must exist')

    def test_bs_inverse_broadcast_and_expiry(self):
        t = np.array([0., .1, 1.])
        k = np.array([.9, 1., 1.1])
        p = self.v.bs_call(t, k, .3)
        self.assertAlmostEqual(p[0], .1)
        np.testing.assert_allclose(self.v.implied_vol(t[1:], k[1:], p[1:]), .3, atol=1e-9)
        self.assertTrue(np.isnan(self.v.implied_vol(.1, 1., 1.1)))

    def test_synthetic_fits_and_serialization(self):
        t, k = np.meshgrid([.05, .1, .3], np.linspace(.85, 1.15, 9), indexing='ij')
        p = self.v.bs_call(t, k, .25)
        for kind in ['svi', 'ssvi', 'essvi']:
            with self.subTest(kind=kind):
                model = self.v.fit_surface(kind, t.ravel(), k.ravel(), p.ravel())
                json.dumps(model, allow_nan=False)
                self.assertTrue(model['diagnostics']['valid'], model['diagnostics'])
                self.assertLess(model['diagnostics']['objective'], 1e-7)
                self.assertEqual(model['diagnostics']['starts'], 5)
                np.testing.assert_allclose(self.v.price_surface(model, 0, [.9, 1.1]), [.1, 0], atol=1e-12)
                s = self.v.summary(model)
                self.assertTrue(s['atm_valid'])
                self.assertTrue(s['skew_valid'])
                self.assertAlmostEqual(s['atm_iv_30d'], .25, delta=.015)

    def test_essvi_interpolates_product_not_rho(self):
        model = {'kind': 'essvi', 'times': [.1, .3], 'parameters': [[.01, -.8, .05], [.04, -.2, .1]]}
        w = self.v.total_variance(model, .2, np.array([.8, 1., 1.2]))
        theta, psi, rho = .025, .075, -.03 / .075
        x = np.log([.8, 1., 1.2])
        expected = .5 * (theta + rho * psi * x + np.sqrt((psi*x+theta*rho)**2+theta**2*(1-rho*rho)))
        np.testing.assert_allclose(w, expected)

    def test_invalid_inputs_rejected(self):
        for prices in [[-.1], [1.1], [np.nan]]:
            with self.assertRaises(ValueError):
                self.v.fit_surface('ssvi', [.1], [1.], prices)
        with self.assertRaises(ValueError):
            self.v.fit_surface('ssvi', [0], [1.], [.1])

    def test_grid_detects_calendar_and_butterfly_failures(self):
        calendar = {'kind': 'essvi', 'times': [.1, .2],
                    'parameters': [[.04, 0., .01], [.01, 0., .01]]}
        check = self.v._grid_check(calendar, [1.])
        self.assertFalse(check['valid'])
        self.assertLess(check['margins']['calendar'], 0)
        butterfly = {'kind': 'svi', 'times': [.1],
                     'parameters': [[.001, 1.9, .9, 0., .005]]}
        check = self.v._grid_check(butterfly, [1.])
        self.assertFalse(check['valid'])
        self.assertLess(check['margins']['strike_convex'], 0)

    def test_curved_essvi_fit_and_repeatability(self):
        source = {'kind': 'essvi', 'times': [.05, .15, .4],
                  'parameters': [[.004, -.6, .05], [.012, -.4, .08], [.032, -.3, .12]]}
        t, k = np.meshgrid(source['times'], np.linspace(.8, 1.2, 11), indexing='ij')
        p = self.v.price_surface(source, t, k)
        fit = self.v.fit_surface('essvi', t, k, p)
        again = self.v.fit_surface('essvi', t, k, p)
        self.assertTrue(fit['diagnostics']['valid'], fit['diagnostics'])
        self.assertLess(fit['diagnostics']['objective'], 1e-8)
        self.assertEqual(fit['parameters'], again['parameters'])
        self.assertFalse(fit['diagnostics']['global_arbitrage_certified'])

    def test_failed_optimizer_cannot_displace_successful_financial_candidate(self):
        source={'kind':'essvi','times':[.05,.1],
                'parameters':[[.004,0.,1e-6],[.008,0.,1e-6]]}
        t,k=np.meshgrid(source['times'],[.95,1.,1.05],indexing='ij')
        prices=self.v.price_surface(source,t,k)
        first=np.array(source['parameters']).ravel();good=first.copy();good[[0,3]]*=1.01
        calls=0
        def optimizer(fun,x,**kwargs):
            nonlocal calls
            z=first if calls==0 else good;success=calls!=0;calls+=1
            return OptimizeResult(x=z.copy(),fun=fun(z),success=success,status=0 if success else 9,
                                  message='successful' if success else 'iteration limit',nit=12)
        with patch.object(self.v,'minimize',side_effect=optimizer):
            result=self.v.fit_surface('essvi',t,k,prices)
        self.assertTrue(result['diagnostics']['valid'])
        np.testing.assert_array_equal(result['parameters'],good.reshape(2,3))
        self.assertFalse(result['diagnostics']['attempts'][0]['qualified'])
        self.assertTrue(all('grid_valid' in row for row in result['diagnostics']['attempts']))

    def test_grid_failing_lower_loss_candidate_cannot_displace_valid_surface(self):
        # Both parameter margins pass the declared 1e-7 tolerance; a small
        # negative theta calendar difference still creates genuine call arb.
        bad=np.array([[1e-6,0.,1e-8],[.95e-6,0.,1e-8]])
        good=np.array([[1e-6,0.,1e-8],[1e-6,0.,1e-8]])
        source={'kind':'essvi','times':[.05,.1],'parameters':bad.tolist()}
        self.assertGreaterEqual(np.min(self.v._constraints(source)),-1e-7)
        self.assertFalse(self.v._grid_check(source,[1.])['valid'])
        t=np.array([.05,.1]);k=np.ones(2);prices=self.v.price_surface(source,t,k);calls=0
        def optimizer(fun,x,**kwargs):
            nonlocal calls
            z=bad.ravel() if calls==0 else good.ravel();calls+=1
            return OptimizeResult(x=z.copy(),fun=fun(z),success=True,status=0,message='successful',nit=4)
        with patch.object(self.v,'minimize',side_effect=optimizer):
            result=self.v.fit_surface('essvi',t,k,prices)
        self.assertTrue(result['diagnostics']['valid'])
        np.testing.assert_array_equal(result['parameters'],good)
        self.assertFalse(result['diagnostics']['attempts'][0]['grid_valid'])

    def test_no_qualified_candidate_remains_explicitly_invalid(self):
        parameters=np.array([[.004,0.,1e-6],[.008,0.,1e-6]])
        source={'kind':'essvi','times':[.05,.1],'parameters':parameters.tolist()}
        t=np.array([.05,.1]);k=np.ones(2);prices=self.v.price_surface(source,t,k)
        def optimizer(fun,x,**kwargs):
            return OptimizeResult(x=parameters.ravel().copy(),fun=fun(parameters.ravel()),success=False,
                                  status=9,message='iteration limit',nit=350)
        with patch.object(self.v,'minimize',side_effect=optimizer):
            result=self.v.fit_surface('essvi',t,k,prices)
        self.assertFalse(result['diagnostics']['valid'])
        self.assertEqual(result['diagnostics']['qualified_candidates'],0)

    def test_real_essvi_20150914_selector_regression(self):
        quote_file=Path(__file__).resolve().parents[3]/'数据/处理后/波动率模型比较/data/quotes.csv'
        if not quote_file.exists():self.skipTest('local real-quote counterexample fixture unavailable')
        import pandas as pd
        quotes=pd.read_csv(quote_file);q=quotes[quotes.date=='2015-09-14']
        self.assertGreater(len(q),0)
        result=self.v.fit_surface('essvi',q.ttm.to_numpy(),q.k.to_numpy(),q.price.to_numpy())
        self.assertTrue(result['diagnostics']['valid'],result['diagnostics'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
