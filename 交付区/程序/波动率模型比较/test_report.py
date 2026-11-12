import unittest
import numpy as np
from report_common import paired_intervals

class ReportTests(unittest.TestCase):
    def test_entry_on_ex_date_does_not_count_as_holding_across_dividend(self):
        from evaluate_common import crosses_ex_date
        events=[('2024-12-02','2024-11-29','2024-12-03',.05,10200)]
        self.assertFalse(crosses_ex_date('2024-12-02','2024-12-25',events))
        self.assertTrue(crosses_ex_date('2024-11-29','2024-12-25',events))
        self.assertTrue(crosses_ex_date('2024-11-29','2024-12-02',events))
        self.assertFalse(crosses_ex_date('2024-11-01','2024-11-28',events))

    def test_surface_cost_includes_rejected_attempts_and_respects_date_cutoff(self):
        from numerical_report import surface_cost
        rows=[dict(day='2020-01-02',status='complete',seconds=2.),
              dict(day='2021-01-02',status='numerical_rejected',seconds=8.),
              dict(day='2022-01-02',status='complete',seconds=100.)]
        result=surface_cost(rows,2021)
        self.assertEqual(result['dates'],2);self.assertEqual(result['qualified_dates'],1)
        self.assertEqual(result['rejected_dates'],1);self.assertEqual(result['sum_fit_seconds'],10.)
        self.assertEqual(result['median_fit_seconds'],5.)

    def test_direct_tail_uses_same_fractional_probability_mass_as_DH(self):
        import pandas as pd
        from direct_dynamic import _groups
        from report_common import risks
        x=-np.arange(41,dtype=float)/100
        frame=pd.DataFrame(dict(end=2025,model='fixture',hedge='delta',execution='same_close',bp=5,
            pnl=x,normalized_cost=0.,normalized_turnover=1.,decision_clip_count=0,pending_fill_clip_count=0))
        row=_groups(frame).iloc[0];rmse,cvar=risks(x)
        self.assertAlmostEqual(row.cvar,cvar);self.assertAlmostEqual(row.cvar05,-cvar)
        self.assertAlmostEqual(row.rmse,rmse)
        self.assertNotAlmostEqual(row.cvar,np.mean(-x[x<=np.quantile(x,.05)]))

    def test_identical_strategies_have_zero_paired_difference(self):
        p=np.array([[.01,-.02,.015,-.03],[.02,-.03,.025,-.02],[.015,-.025,.02,-.025]])
        result=paired_intervals(p,p,np.array(['2024-01','2024-02','2024-03','2024-04']),3,100)
        np.testing.assert_allclose([result[k] for k in ('rmse_lower','rmse_upper','cvar_lower','cvar_upper')],0)

if __name__=='__main__':unittest.main()
