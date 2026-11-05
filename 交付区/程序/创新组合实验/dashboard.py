"""Read saved study outputs through a localhost-only browser interface."""
import argparse
import csv
import gzip
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import json
from pathlib import Path
from urllib.parse import parse_qs,urlparse


PAGE = """<!doctype html>
<html lang="zh-CN"><meta charset="utf-8"><title>50ETF 对冲组合实验</title>
<style>
body{font:16px system-ui,sans-serif;margin:2rem auto;max-width:1250px;padding:0 1rem;color:#1d2732}
label{display:inline-block;margin:.25rem .7rem .5rem 0} select,button{padding:.35rem}
table{border-collapse:collapse;width:100%;margin:1rem 0} th,td{padding:.4rem;border-bottom:1px solid #ddd;text-align:right}
th:first-child,td:first-child{text-align:left} img{width:min(100%,750px)} section{margin:1rem 0}
#detail{max-height:26rem;overflow:auto} .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr))}
</style>
<h1>50ETF 对冲组合实验</h1>
<p>2024–2025 为已观察的历史评价。损益和成本按初始名义金额归一化。</p>
<section id="filters"></section><section id="metrics"></section>
<div class="grid"><img src="/risk_tradeoff.png" alt="RMSE 与尾部损失"><img src="/rolling_stability.png" alt="跨年结果"></div>
<h2>逐日路径</h2>
<label>策略 <select id="strategy"></select></label><label>合约路径 <select id="episode"></select></label>
<button id="show">查看逐日记录</button>
<canvas id="chart" width="850" height="240" aria-label="ETF 份数与门控权重"></canvas>
<div id="detail"></div>
<script>
const el=id=>document.getElementById(id);
const get=async url=>{const r=await fetch(url);if(!r.ok)throw Error(await r.text());return r.json()};
const option=(value,label)=>new Option(label??value,value);
const escape=s=>String(s).replace(/[&<>]/g,x=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[x]));
const table=(headers,rows)=>'<table><thead><tr>'+headers.map(x=>'<th>'+escape(x)+'</th>').join('')+'</tr></thead><tbody>'+rows.map(r=>'<tr>'+r.map(x=>'<td>'+escape(x)+'</td>').join('')+'</tr>').join('')+'</tbody></table>';
const pct=x=>(100*Number(x)).toFixed(4)+'%';
(async()=>{
  const [metrics,index]=await Promise.all([get('/metrics'),get('/index')]);
  const fields={end:'评价截止',cost:'成本 (bp)',lam:'损失权重',group:'分组'};
  const defaults={end:'2025',cost:'5',lam:'0',group:'all'};
  for(const [key,label] of Object.entries(fields)){
    const node=document.createElement('label');node.textContent=label+' ';
    const select=document.createElement('select');select.id=key;
    for(const v of [...new Set(metrics.map(r=>r[key]))].sort())select.add(option(v));
    select.value=defaults[key];node.append(select);el('filters').append(node);
    select.onchange=render;
  }
  const title=r=>[r.combo||r.policy,r.rep,r.market].join(' / ');
  function render(){
    const rows=metrics.filter(r=>Object.keys(fields).every(k=>r[k]===el(k).value||(k==='lam'&&r[k]==='-1')));
    el('metrics').innerHTML=table(['策略','路径数','RMSE','CVaR95','平均损益','交易成本','换手','训练秒'],
      rows.map(r=>[title(r),r.n,pct(r.rmse),pct(r.cvar95),pct(r.mean_pnl),pct(r.trade_cost),pct(r.turnover),Number(r.seconds).toFixed(1)]));
    el('strategy').replaceChildren(...index.filter(r=>String(r.end)===el('end').value&&String(r.cost)===el('cost').value&&(String(r.lam)===el('lam').value||r.lam===-1))
      .map(r=>option(r.id,[title(r),r.seed].join(' / '))));
    fillEpisodes();
  }
  async function fillEpisodes(){
    const id=el('strategy').value;
    if(!id)return;
    const data=await get('/episodes?id='+encodeURIComponent(id));
    if(id===el('strategy').value)el('episode').replaceChildren(...data.map(r=>option(r.episode,r.episode+' / '+r.contract+' / '+r.expiry)));
  }
  el('strategy').onchange=fillEpisodes;
  el('show').onclick=async()=>{
    const data=await get('/path?id='+encodeURIComponent(el('strategy').value)+'&episode='+encodeURIComponent(el('episode').value));
    const fields=['date','spot','strike','unit','shares','traded','cost','accrued','paid','wealth','gate_weight','factor1','factor2'];
    el('detail').innerHTML=table(fields,data.map(r=>fields.map(k=>r[k]??'')));
    const canvas=el('chart'), ctx=canvas.getContext('2d');
    ctx.clearRect(0,0,canvas.width,canvas.height);
    for(const [name,color,max] of [['shares','#086a87',Math.max(1,...data.map(r=>Number(r.unit)))],['gate_weight','#a13c25',1]]){
      const series=data.map((r,i)=>[i,r[name]]).filter(([,v])=>v!=='');
      ctx.beginPath();ctx.strokeStyle=color;
      series.forEach(([i,v],j)=>{const x=35+i*780/Math.max(1,data.length-1),y=215-180*Number(v)/max;if(j)ctx.lineTo(x,y);else ctx.moveTo(x,y)});
      ctx.stroke();ctx.fillStyle=color;ctx.fillText(name,name==='shares'?35:130,15);
    }
  };
  render();
})().catch(e=>{el('metrics').textContent=e.message});
</script></html>"""


def serve(root,port):
    directory=Path(root)/"evaluation"
    allowed={r["id"] for r in json.loads((directory/"paths_index.json").read_text(encoding="utf8"))}
    class Handler(BaseHTTPRequestHandler):
        def send(self,data,kind):
            self.send_response(200)
            self.send_header("Content-Type",kind)
            self.send_header("Content-Length",str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def send_json(self,rows):
            self.send(json.dumps(rows).encode(),"application/json")

        def do_GET(self):
            url=urlparse(self.path)
            params=parse_qs(url.query)
            if url.path=="/":
                return self.send(PAGE.encode(),"text/html; charset=utf-8")
            if url.path in {"/risk_tradeoff.png","/rolling_stability.png"}:
                return self.send((directory/url.path[1:]).read_bytes(),"image/png")
            if url.path=="/metrics":
                with (directory/"summary.csv").open(encoding="utf8",newline="") as f:
                    return self.send_json(list(csv.DictReader(f)))
            if url.path=="/index":
                return self.send((directory/"paths_index.json").read_bytes(),"application/json")
            key=params.get("id",[""])[0]
            if key not in allowed:
                return self.send_error(404)
            if url.path=="/episodes":
                p=directory/"paths"/f"{key}.json"
                rows=json.loads(p.read_text(encoding="utf8"))["rows"]
                return self.send_json([{k:r[k] for k in ("episode","contract","expiry")} for r in rows])
            if url.path=="/path":
                episode=params.get("episode",[""])[0]
                with gzip.open(directory/"daily"/f"{key}.csv.gz","rt",encoding="utf8",newline="") as f:
                    return self.send_json([r for r in csv.DictReader(f) if r["episode"]==episode])
            self.send_error(404)
    print(f"http://127.0.0.1:{port}/",flush=True)
    ThreadingHTTPServer(("127.0.0.1",port),Handler).serve_forever()


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("study_data")
    p.add_argument("--port",type=int,default=8765)
    a=p.parse_args()
    serve(a.study_data,a.port)

