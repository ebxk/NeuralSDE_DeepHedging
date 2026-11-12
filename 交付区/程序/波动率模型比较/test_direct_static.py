import importlib.util
from pathlib import Path
import tempfile
import json
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd


class StaticDirectTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path=Path(__file__).with_name('direct_hedges.py')
        spec=importlib.util.spec_from_file_location('static_direct',path)
        cls.m=importlib.util.module_from_spec(spec);spec.loader.exec_module(cls.m)

    def test_pending_absolute_shares_adjustment_and_expiry_hand(self):
        rows=[dict(date=d,expiry='2020-12-01',spot=s,strike=k,unit=u,option_close=.1)
              for d,s,k,u in [('2020-11-27',2.,2.,100),('2020-11-30',1.949,1.96,102),('2020-12-01',2.1,1.96,102)]]
        events=[('2020-11-30','2020-11-27','2020-12-01','.051',102)]
        result=self.m.execute_path(rows,[np.float32(.5),np.float32(.5)],0,'next_close',events)
        self.assertEqual(result['book'][1]['shares'],50.)
        self.assertEqual(result['diagnostics']['expiry_pending_cancelled_shares'],51.)
        self.assertAlmostEqual(result['book'][-1]['wealth'],3.27)

    def test_first_evaluation_day_can_use_previous_full_calendar_curve(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d);path=out/'surfaces/essvi/2020-01-30';path.mkdir(parents=True)
            model={'kind':'essvi','times':[.1],'parameters':[[.004,0,0]],'diagnostics':{'valid':True}}
            (path/'full.json').write_text(json.dumps({'status':'complete','model':model}),encoding='utf8')
            days=pd.date_range('2020-01-27',periods=10).strftime('%Y-%m-%d').to_numpy()
            with patch.object(self.m,'OUT',out):
                lookup=self.m.surface_lookup('essvi',['2020-02-01'],days,lambda p:p)
            self.assertEqual(lookup['2020-02-01']['source_date'],'2020-01-30')
            self.assertEqual(lookup['2020-02-01']['stale_days'],2)

    def test_rv_exdate_total_return_and_no_future_dependency(self):
        days=pd.date_range('2020-01-01',periods=31).strftime('%Y-%m-%d')
        close=pd.Series(np.r_[np.full(24,2.),np.full(7,1.9)],index=days)
        events=[('2020-01-25','2020-01-24','2020-01-27','.1',10100)]
        rv=self.m.rv20_series(close,events)
        self.assertAlmostEqual(rv.loc['2020-01-25'],0.,places=12)
        close.loc['2020-01-26':]=1000
        changed=self.m.rv20_series(close,events)
        self.assertEqual(rv.loc['2020-01-25'],changed.loc['2020-01-25'])

    def test_all_static_scenarios_have_same_paths(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);out=root/'out';out.mkdir();old=root/'工作区/数据/处理后/创新组合实验/2025';old.mkdir(parents=True)
            (out/'surface_stage.json').write_text(json.dumps({'stage_complete':True}),encoding='utf8')
            rows=[]
            for episode in [1,9]:
                for i,spot in enumerate([2.,2.02,2.03]):
                    rows.append(dict(episode=episode,split_end=2025,date=f'2020-01-0{i+1}',expiry='2020-01-03',
                                     spot=spot,strike=2.,unit=10000,option_close=.05,term='15-30',moneyness='ATM'))
            pd.DataFrame(rows).to_csv(old/'episode_days.csv',index=False)
            raw=root/'工作区/数据/原始数据/50ETF';raw.mkdir(parents=True)
            dates=pd.date_range('2019-12-01','2020-01-03').strftime('%Y-%m-%d')
            pd.DataFrame({'交易日期':dates[::-1],'收盘价(元)':np.full(len(dates),2.)}).to_csv(
                raw/'510050_华夏上证50ETF.csv',index=False,encoding='gb18030')
            for family in ['svi','ssvi','essvi']:
                for day in ['2020-01-01','2020-01-02']:
                    path=out/f'surfaces/{family}/{day}';path.mkdir(parents=True)
                    if family=='svi':model={'kind':'svi','times':[.05,.2],'parameters':[[.002,0.,0.,0.,.2],[.008,0.,0.,0.,.2]]}
                    elif family=='ssvi':model={'kind':'ssvi','times':[.05,.2],'parameters':[.002,.008],'rho':0.,'eta':0.}
                    else:model={'kind':'essvi','times':[.05,.2],'parameters':[[.002,0.,0.],[.008,0.,0.]]}
                    model['diagnostics']={'valid':True}
                    (path/'full.json').write_text(json.dumps({'status':'complete','model':model}),encoding='utf8')
            with patch.object(self.m,'OUT',out),patch.object(self.m,'PROJECT',root):receipt=self.m.run(2025)
            self.assertTrue(receipt['complete'])
            frame=pd.read_csv(out/'direct/2025/path_pnl.csv')
            self.assertEqual(len(frame),56)
            self.assertEqual(len(frame.groupby(['model','delta','execution','cost_bp'])),28)
            for _,group in frame.groupby(['model','delta','execution','cost_bp']):
                self.assertEqual(set(group.episode),{1,9})
            self.assertEqual(len(pd.read_csv(out/'direct/2025/daily_ledger.csv')),168)


if __name__=='__main__':unittest.main(verbosity=2)
