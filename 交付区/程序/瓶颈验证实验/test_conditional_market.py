import unittest
import numpy as np
from conditional_market import calibrated, energy_score, crps


class ConditionalTests(unittest.TestCase):
    def test_right_scaling_preserves_boundary_and_arithmetic_drift(self):
        def original(x):
            return np.tile([.03,.1,.2],(len(x),1)), np.tile(
                np.array([[.2,.1,0],[0,0,0],[.1,0,.3]]),(len(x),1,1))
        x=np.zeros((2,2))
        mu, sigma=original(x)
        new_mu,new_sigma=calibrated(original,[.8,1,1.25])(x)
        np.testing.assert_allclose(new_sigma[:,1],0)
        np.testing.assert_allclose(new_mu[:,0]+.5*(new_sigma[:,0]**2).sum(1),
                                   mu[:,0]+.5*(sigma[:,0]**2).sum(1))
        np.testing.assert_allclose(calibrated(original,[1,1,1])(x)[0],mu)

    def test_proper_scores_point_forecast(self):
        self.assertEqual(crps(np.array([1.,1.]),3.),2.)
        self.assertEqual(energy_score(np.ones((2,2)),np.array([1.,3.])),2.)

if __name__=='__main__':
    unittest.main()
