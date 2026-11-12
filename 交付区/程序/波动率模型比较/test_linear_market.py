import unittest
import numpy as np
from linear_market import coefficient_head_source, shared_summary

class LinearMarketTests(unittest.TestCase):
    def test_only_head_changes(self):
        original='x\n    model = tf.keras.Sequential([tf.keras.layers.Input((2,)),\n        tf.keras.layers.Dense(64,activation="tanh"), tf.keras.layers.Dense(64,activation="tanh"),\n        tf.keras.layers.Dense(7)])\ny'
        new=coefficient_head_source(original)
        self.assertNotIn('Dense(64',new)
        self.assertIn('Dense(7)',new)
        self.assertTrue(new.startswith('x\n'))
        self.assertTrue(new.endswith('\ny'))
        with self.assertRaises(ValueError): coefficient_head_source('unrecognized source')

    def test_grid_summary_matches_flat_bs_surface(self):
        from volatility_surfaces import bs_call
        ts=np.array([14,30,60,90])/365;ks=np.array([.9,.95,1,1.05,1.1])
        prices=bs_call(ts[:,None],ks[None,:],.2)
        out=shared_summary(prices,ts,ks)
        np.testing.assert_allclose(out[:2],[.2,0],atol=1e-8)
        np.testing.assert_array_equal(out[2:],[1,1])

if __name__=='__main__':unittest.main()
