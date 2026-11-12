import importlib.util
from pathlib import Path
import unittest
import numpy as np
from unittest.mock import patch
from scipy.optimize import OptimizeResult


class StochasticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).with_name('stochastic_volatility.py')
        cls.exists = path.exists()
        if cls.exists:
            spec = importlib.util.spec_from_file_location('stochastic', path)
            cls.m = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(cls.m)

    def setUp(self):
        self.assertTrue(self.exists, 'stochastic module must exist')

    def test_constant_vol_limits_and_expiry(self):
        k = np.array([.8, 1., 1.2])
        expected = self.m.bs_call(.5, k, .2)
        np.testing.assert_allclose(self.m.price_heston(.5,k,.04,2.,.04,0.,-.5), expected, atol=1e-10)
        np.testing.assert_allclose(self.m.price_sabr(.5,k,.2,0.,-.5), expected, atol=1e-10)
        np.testing.assert_allclose(self.m.price_heston(0,k,.04,2.,.04,.3,-.5), np.maximum(1-k,0))

    def test_heston_quadrature_against_independent_quad(self):
        from scipy.integrate import quad
        t, k, v, kap, theta, xi, rho = .7, 1.1, .04, 1.5, .05, .4, -.6
        def cf(u):
            b = kap-rho*xi*1j*u
            d = np.sqrt(b*b+xi*xi*(u*u+1j*u))
            g = (b-d)/(b+d)
            c = kap*theta/xi**2*((b-d)*t-2*np.log((1-g*np.exp(-d*t))/(1-g)))
            dd = (b-d)/xi**2*(1-np.exp(-d*t))/(1-g*np.exp(-d*t))
            return np.exp(c+dd*v)
        p1 = .5+quad(lambda u: np.real(np.exp(-1j*u*np.log(k))*cf(u-1j)/(1j*u)), 1e-8, 1000, epsabs=1e-10)[0]/np.pi
        p2 = .5+quad(lambda u: np.real(np.exp(-1j*u*np.log(k))*cf(u)/(1j*u)), 1e-8, 1000, epsabs=1e-10)[0]/np.pi
        self.assertAlmostEqual(float(self.m.price_heston(t,k,v,kap,theta,xi,rho)),p1-k*p2,places=7)

    def test_heston_real_short_low_variance_counterexample(self):
        model={'kappa':.459383058744558,'theta':.13825085092212847,
               'volvol':.5325879972389376,'rho':-.045780834961336114}
        k=np.linspace(.5,1.5,161);v=.002877310695567458
        calls=self.m.price_heston(1/365,k,v,**model)
        slopes=np.diff(calls)/np.diff(k)
        self.assertGreaterEqual(np.min(calls-np.maximum(1-k,0)),-1e-7)
        self.assertGreaterEqual(np.min(-slopes),-1e-6)
        self.assertGreaterEqual(np.min(np.diff(slopes)),-2e-6)
        self.assertGreater(self.m.pricing_diagnostics()['fallback_quotes'],0)

    def test_prices_convex_and_derivatives(self):
        for kind, model, state in [('heston', {'kappa':2.,'theta':.04,'volvol':.3,'rho':-.5}, .04),
                                   ('sabr', {'nu':.3,'rho':-.4}, .2)]:
            k = np.linspace(.7,1.3,81)
            p = self.m.price_model(kind,model,.25,k,state)
            self.assertTrue(np.all(p >= np.maximum(1-k,0)-1e-7))
            self.assertTrue(np.all(p <= 1+1e-7))
            self.assertGreater(np.min(np.diff(p,2)), -1e-7)
            d = self.m.q_derivatives(kind,model,.25,1.05,state)
            h=1e-5
            cp=(1+h)*self.m.price_model(kind,model,.25,1.05/(1+h),state)
            cm=(1-h)*self.m.price_model(kind,model,.25,1.05/(1-h),state)
            self.assertAlmostEqual(d['delta'], (cp-cm)/(2*h), places=5)
            self.assertGreater(d['gamma'],0)

    def test_filter_only_supplied_current_quotes(self):
        model={'nu':.3,'rho':-.4}
        k=np.array([.9,1.,1.1]);t=np.array([.1,.2,.3])
        prices=self.m.price_sabr(t,k,.24,**model)
        state=self.m.filter_state('sabr',model,t,k,prices,previous=.2)
        self.assertAlmostEqual(state,.24,places=5)
        # Repeating on exactly the supplied prefix cannot read later quotes.
        future=prices.copy();future[-1]=.9
        prefix=self.m.filter_state('sabr',model,t[:2],k[:2],prices[:2])
        repeated=self.m.filter_state('sabr',model,t[:2],k[:2],future[:2])
        self.assertEqual(prefix,repeated)

    def test_p_transitions_and_shared_shocks(self):
        rng=np.random.default_rng(7);n=5000;dt=np.full(n,1/252)
        a=np.full(n,.2);z=rng.standard_normal(n)
        nxt=a*np.exp((.1-.5*.3**2)*dt+.3*np.sqrt(dt)*z)
        p=self.m.estimate_p_transitions('sabr',a,nxt,dt)
        self.assertAlmostEqual(p['diagnostics']['state_volvol_empirical'],.3,delta=.015)
        shocks=np.zeros((3,2));shocks[:,0]=[.1,-.2,.3]
        a=self.m.simulate_p('sabr',{'rho':-.5,'nu':.3},p,1.,.2,1/252,shocks)
        b=self.m.simulate_p('sabr',{'rho':-.5,'nu':.3},p,1.,.2,1/252,shocks)
        np.testing.assert_array_equal(a['spot'],b['spot'])
        self.assertTrue(np.all(a['state']>0))
        with self.assertRaises(ValueError):
            self.m.estimate_p_transitions('sabr',[.2],[.21],[0])
        with self.assertRaises(ValueError):
            self.m.price_sabr(.1,1.,.2,.3,.2)

    def test_p_simulation_enforces_frozen_q_diffusion(self):
        model={'nu':.3,'rho':-.5}
        drift={'state_drift':.1,'spot_drift':.03,'state_volvol':8.,'rho':.99}
        altered=dict(drift,state_volvol=.01,rho=-.99)
        z=np.array([[.2,.5],[-.3,.1]])
        a=self.m.simulate_p('sabr',model,drift,1.,.2,.01,z)
        b=self.m.simulate_p('sabr',model,altered,1.,.2,.01,z)
        np.testing.assert_array_equal(a['state'],b['state'])
        d=self.m.q_derivatives('sabr',model,.1,1.,.2)
        self.assertAlmostEqual(d['lmv'],d['delta']-.5*.3*d['state_derivative'])

    def test_sabr_estimated_log_drift_with_frozen_q_nu(self):
        x=np.full(4,.2);dt=np.full(4,.01)
        log_returns=np.array([-.1,.05,.03,.024])
        p=self.m.estimate_p_transitions('sabr',x,x*np.exp(log_returns),dt)
        expected=float(log_returns.sum()/dt.sum())
        self.assertAlmostEqual(p['state_log_drift'],expected)
        model={'nu':.3,'rho':-.5}
        paths=self.m.simulate_p('sabr',model,p,1.,.2,.01,np.zeros((1,2)))
        self.assertAlmostEqual(np.log(paths['state'][-1]/.2),expected*.01)

    def test_global_fit_fixed_states(self):
        import json
        t,k=np.meshgrid([.1,.4],np.linspace(.85,1.15,7),indexing='ij')
        states=np.full(t.shape,.2)
        prices=self.m.price_sabr(t,k,states,.3,-.4)
        model=self.m.fit_global('sabr',t,k,prices,states)
        json.dumps(model,allow_nan=False)
        self.assertLess(model['diagnostics']['objective'],1e-9)
        self.assertTrue(model['diagnostics']['valid'])

    def test_q_validation_reports_invalid_hagan_region(self):
        check=self.m.check_q_grid('sabr',{'nu':2.,'rho':-.95},2.)
        self.assertFalse(check['valid'])
        self.assertIn('nonpositive',check['error'])

    def test_heston_global_fit_and_q_simulation_moments(self):
        model={'kappa':2.,'theta':.04,'volvol':.3,'rho':-.5}
        t,k=np.meshgrid([.1,.4],np.linspace(.9,1.1,5),indexing='ij')
        states=np.full(t.shape,.04)
        prices=self.m.price_heston(t,k,states,**model)
        fitted=self.m.fit_global('heston',t,k,prices,states)
        self.assertTrue(fitted['diagnostics']['valid'],fitted['diagnostics'])
        self.assertLess(fitted['diagnostics']['objective'],1e-8)
        rng=np.random.default_rng(200)
        shocks=rng.standard_normal((20000,64,2))
        p={'state_kappa':2.,'state_theta':.04,'state_volvol':.3,'spot_drift':0.,'rho':-.5}
        paths=self.m.simulate_p('heston',model,p,1.,.04,.25/64,shocks)
        self.assertAlmostEqual(np.mean(paths['spot'][:,-1]),1.,delta=.004)
        self.assertAlmostEqual(np.mean(paths['state'][:,-1]),.04,delta=.001)
        mc=np.mean(np.maximum(paths['spot'][:,-1]-1,0))
        analytic=float(self.m.price_heston(.25,1.,.04,**model))
        self.assertAlmostEqual(mc,analytic,delta=.0015)

    def test_global_selector_uses_all_three_state_grids_before_lowest_loss(self):
        states=np.array([1.8,2.,2.2]);t=np.full(3,.01);k=np.array([.95,1.,1.05])
        bad=np.array([2.,-.95]);good=np.array([.3,-.4])
        self.assertFalse(self.m.check_q_grid('sabr',{'nu':bad[0],'rho':bad[1]},2.)['valid'])
        self.assertTrue(all(self.m.check_q_grid('sabr',{'nu':good[0],'rho':good[1]},float(s))['valid'] for s in states))
        prices=self.m.price_sabr(t,k,states,*bad);count=0
        def optimizer(fun,x,**kwargs):
            nonlocal count
            z=bad if count==0 else good;count+=1
            return OptimizeResult(x=z.copy(),fun=fun(z),success=True,status=0,message='successful',nit=8)
        with patch.object(self.m,'minimize',side_effect=optimizer):
            result=self.m.fit_global('sabr',t,k,prices,states)
        self.assertTrue(result['diagnostics']['valid'])
        self.assertEqual(result['nu'],good[0]);self.assertEqual(result['rho'],good[1])
        attempts=result['diagnostics']['attempts']
        self.assertFalse(attempts[0]['qualified']);self.assertFalse(attempts[0]['grid_valid'])
        self.assertTrue(all(len(row['grid'])==3 for row in attempts))
        self.assertLess(attempts[0]['objective'],result['diagnostics']['objective'])

    def test_global_selector_no_qualified_candidate_stays_invalid(self):
        state=np.array([1.8,2.,2.2]);t=np.full(3,.01);k=np.ones(3);z=np.array([2.,-.95])
        prices=self.m.price_sabr(t,k,state,*z)
        def optimizer(fun,x,**kwargs):
            return OptimizeResult(x=z.copy(),fun=fun(z),success=True,status=0,message='successful',nit=2)
        with patch.object(self.m,'minimize',side_effect=optimizer):
            result=self.m.fit_global('sabr',t,k,prices,state)
        self.assertFalse(result['diagnostics']['valid'])
        self.assertEqual(result['diagnostics']['qualified_candidates'],0)

    def test_global_selector_failed_optimizer_cannot_beat_valid_success(self):
        state=np.full(3,.2);t=np.array([.05,.1,.2]);k=np.array([.95,1.,1.05])
        bad=np.array([.3,-.4]);good=np.array([.31,-.4]);prices=self.m.price_sabr(t,k,state,*bad);count=0
        def optimizer(fun,x,**kwargs):
            nonlocal count
            success=count>0;z=good if success else bad;count+=1
            return OptimizeResult(x=z.copy(),fun=fun(z),success=success,status=0 if success else 1,
                                  message='successful' if success else 'iteration limit',nit=20)
        with patch.object(self.m,'minimize',side_effect=optimizer):
            result=self.m.fit_global('sabr',t,k,prices,state)
        self.assertTrue(result['diagnostics']['valid']);self.assertEqual(result['nu'],good[0])
        self.assertEqual(result['diagnostics']['qualified_candidates'],4)


if __name__=='__main__':
    unittest.main(verbosity=2)
