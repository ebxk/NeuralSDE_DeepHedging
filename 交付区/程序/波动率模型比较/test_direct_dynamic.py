import importlib.util
from pathlib import Path
import unittest
import numpy as np
import tempfile
import json
from unittest.mock import patch


class DirectTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path=Path(__file__).with_name('direct_dynamic.py');cls.exists=path.exists()
        if cls.exists:
            spec=importlib.util.spec_from_file_location('direct',path)
            cls.m=importlib.util.module_from_spec(spec);spec.loader.exec_module(cls.m)

    def setUp(self):self.assertTrue(self.exists,'direct dynamic module must exist')

    @staticmethod
    def rows():
        return [dict(date=f'2020-01-0{i+1}',expiry='2020-01-03',spot=s,strike=100,unit=1,option_close=1.)
                for i,s in enumerate([100.,110.,120.])]

    def test_bs_next_close_hand_settlement_and_expiry_cancellation(self):
        rows=self.rows();sigma=.2
        def action(row,i,previous):
            t=(np.datetime64(row['expiry'])-np.datetime64(row['date'])).astype(float)/365
            return self.m.bs_delta(t,float(row['strike'])/float(row['spot']),sigma)
        first=action(rows[0],0,0);result=self.m.rollout(rows,action,.001,delayed=True,events=[])
        self.assertEqual(result['book'][0]['shares'],0)
        self.assertAlmostEqual(result['book'][1]['shares'],first)
        expected=1+first*(120-110)-20-.001*first*(110+120)
        self.assertAlmostEqual(result['book'][-1]['wealth'],expected)
        self.assertGreater(result['diagnostics']['expiry_pending_cancelled_shares'],0)
        self.assertEqual(result['book'][-1]['shares'],0)

    def test_dividend_record_unit_adjustment_and_absolute_pending_hand(self):
        rows=[dict(date=d,expiry='2020-12-01',spot=s,strike=k,unit=u,option_close=.1)
              for d,s,k,u in [('2020-11-27',2.,2.,100),('2020-11-30',1.949,1.96,102),('2020-12-01',2.1,1.96,102)]]
        events=[('2020-11-30','2020-11-27','2020-12-01','0.051',102)]
        current=self.m.rollout(rows,lambda row,i,p:.5,0,events=events)
        delayed=self.m.rollout(rows,lambda row,i,p:.5,0,delayed=True,events=events)
        self.assertAlmostEqual(current['book'][-1]['wealth'],3.421)
        self.assertAlmostEqual(current['book'][-1]['paid'],2.55)
        self.assertAlmostEqual(delayed['book'][1]['shares'],50.)
        self.assertAlmostEqual(delayed['book'][-1]['wealth'],3.27)
        self.assertEqual(delayed['book'][-1]['paid'],0)
        self.assertEqual(delayed['diagnostics']['expiry_pending_cancelled_shares'],51.)

    def test_heston_sabr_delta_lmv_and_bounds(self):
        from stochastic_volatility import q_derivatives
        for kind,model,state in [('heston',{'kappa':2.,'theta':.04,'volvol':.3,'rho':-.5},.04),
                                 ('sabr',{'nu':.3,'rho':-.4},.2)]:
            actual=self.m.stochastic_hedges(kind,model,.2,1.05,state)
            reference=q_derivatives(kind,model,.2,1.05,state)
            self.assertAlmostEqual(actual['delta'],reference['delta'],places=8)
            self.assertAlmostEqual(actual['lmv'],reference['lmv'],places=8)
        result=self.m.rollout(self.rows(),lambda row,i,p:1.2,0,events=[])
        self.assertEqual(result['diagnostics']['decision_clip_count'],2)

    def test_state_past_only_stale_limit(self):
        calendar={f'2020-01-0{i}':i for i in range(1,10)}
        states={'2020-01-01':.04,'2020-01-08':.09}
        value,metadata=self.m.select_state('2020-01-04',states,calendar)
        self.assertEqual(value,.04);self.assertEqual(metadata['stale_days'],3)
        value,metadata=self.m.select_state('2020-01-07',states,calendar)
        self.assertIsNone(value);self.assertEqual(metadata['fallback_source'],'RV20_BS')

    def test_numpy32_callback_preserves_python_float_financial_ledger(self):
        from contract_ledger import ledger
        rows=self.rows()
        for row in rows:row['unit']=10000
        action=np.float32(.234567)
        result=self.m.rollout(rows,lambda row,i,previous:action,.0005,events=[])
        expected=ledger(rows,[float(action)]*2,1.,.0005,events=[])
        self.assertTrue(all(type(x) is float for x in result['effective_ratios']))
        self.assertEqual(result['book'][-1]['wealth'],expected[-1]['wealth'])

    def test_small_real_evaluation_outputs_and_missing_benchmark(self):
        import pandas as pd
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);out=root/'out';old=root/'工作区/数据/处理后/创新组合实验/2025';old.mkdir(parents=True)
            records=[]
            for episode in [11,23]:
                for row in self.rows():
                    records.append(dict(**row,episode=episode,split_end=2025,term='15-30',moneyness='ATM'))
            pd.DataFrame(records).to_csv(old/'episode_days.csv',index=False)
            raw=root/'工作区/数据/原始数据/50ETF';raw.mkdir(parents=True)
            dates=pd.date_range('2019-12-01','2020-01-03').strftime('%Y-%m-%d')
            pd.DataFrame({'交易日期':dates[::-1],'收盘价(元)':np.full(len(dates),100.)}).to_csv(
                raw/'510050_华夏上证50ETF.csv',index=False,encoding='gb18030')
            directory=out/'markets/2025/sabr';directory.mkdir(parents=True)
            q={'nu':0.,'rho':-.4,'diagnostics':{'valid':True}}
            (directory/'global.json').write_text(json.dumps({'model':q}),encoding='utf8')
            (directory/'status.json').write_text(json.dumps({'complete':True}),encoding='utf8')
            pd.DataFrame({'date':['2020-01-01','2020-01-02'],'state':[.2,.2]}).to_csv(directory/'states.csv',index=False)
            with patch.object(self.m,'OUT',out),patch.object(self.m,'PROJECT',root):
                receipt=self.m.run_model(2025,'sabr')
                self.assertTrue(receipt['complete']);self.assertEqual(receipt['paths'],2)
                pnl=pd.read_csv(out/'direct_dynamic/2025/sabr/pnl.csv')
                self.assertEqual(len(pnl),16)
                self.assertEqual(set(pnl.episode),{11,23})
                ledger=pd.read_csv(out/'direct_dynamic/2025/sabr/daily_ledger.csv.gz')
                self.assertEqual(len(ledger),48)
                self.assertTrue(np.isfinite(ledger.wealth).all())
                q['diagnostics']['valid']=False
                (directory/'global.json').write_text(json.dumps({'model':q}),encoding='utf8')
                missing=self.m.run_model(2025,'sabr')
                self.assertFalse(missing['complete']);self.assertEqual(missing['status'],'benchmark_missing')
                surface={'kind':'essvi','times':[.05,.2,.4],
                    'parameters':[[.04*t,0,0] for t in [.05,.2,.4]],'diagnostics':{'valid':True}}
                for day in ['2020-01-01','2020-01-02']:
                    dest=out/f'surfaces/essvi/{day}';dest.mkdir(parents=True)
                    (dest/'full.json').write_text(json.dumps({'status':'complete','model':surface}),encoding='utf8')
                import generator_export
                with patch.object(generator_export,'OUT',out):
                    lv=self.m.run_model(2025,'localvol')
                self.assertTrue(lv['complete']);self.assertEqual(lv['path_ids_sha256'],receipt['path_ids_sha256'])
                book=pd.read_csv(out/'direct_dynamic/2025/localvol/daily_ledger.csv.gz')
                first=book[(book.episode==11)&(book.bp==0)].iloc[0]
                self.assertAlmostEqual(first.raw_decision,self.m.bs_delta(2/365,1.,.2),delta=.01)


if __name__=='__main__':unittest.main(verbosity=2)
