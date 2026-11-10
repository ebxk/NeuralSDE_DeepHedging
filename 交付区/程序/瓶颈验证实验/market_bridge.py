"""A promoted route must retain every rolling year, including adverse diagnostics."""
import argparse
import json
from pathlib import Path
import conditional_market as market

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--project',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--end',type=int,required=True);p.add_argument('--export',action='store_true')
    p.add_argument('--reference-only',action='store_true')
    args=p.parse_args()
    market.run(args)
    # Global route promotion was frozen before these historical years are read.
    if args.end!=2025:
        folder=args.out/'markets'/str(args.end)
        report=json.loads((folder/'calibration.json').read_text(encoding='utf8'))
        fit,geom,rep,common,cfg,cal,base,identity=market.load_frozen(args.project,args.end)
        if not report['prediction_gate']:
            market.export(args.project,folder,base,fit,geom,rep,common,cfg,report)
