"""F13b_구성하한 — 판 뒤집기 검토(docs/판_뒤집기_검토.md, 11/11 검토 에이전트 작성, 읽기 전용 분석).
구성 하한(253곳 한 묶음 벌금 대비 지금 구성·참조 구성·지점 균형 구성)"""
import numpy as np, pandas as pd
R="C:/Users/109so/OneDrive/Desktop/학교/compete/oibc/"
TZ="Asia/Seoul"
sites=pd.read_csv(R+"01_participant_dataset/sites.csv"); sites["site_id"]=sites.site_id.astype(str)
sid=sites.site_id.tolist(); cap=sites.capacity_kw.to_numpy(); tot=cap.sum()
comp=pd.read_csv(R+"outputs/compositions/E12-10cx2.csv"); comp["site_id"]=comp.site_id.astype(str)
xc=(comp.set_index("site_id").group.reindex(sid)=="G1").to_numpy()
xr=(sites.current_group=="A").to_numpy()
# station-stratified balanced split: within each wa_station, sort by capacity desc, alternate greedy to the lighter group
def strat(key, seed=0):
    rng=np.random.default_rng(seed)
    x=np.zeros(len(cap),bool); c=[0.0,0.0]
    order=sites.assign(r=rng.random(len(sites))).sort_values([key,"capacity_kw","r"],ascending=[True,False,True]).index
    # per station greedy
    for st,grp in sites.loc[order].groupby(key,sort=False):
        loc=[0.0,0.0]
        for i in grp.index:
            # prefer group with less capacity within station; tie -> global lighter
            g = 0 if (loc[0]<loc[1] or (loc[0]==loc[1] and c[0]<=c[1])) else 1
            x[i]= (g==0); loc[g]+=cap[i]; c[g]+=cap[i]
    return x
a=pd.read_parquet(R+"outputs/work/base.parquet",columns=["site_id","ts","energy"])
a["site_id"]=a.site_id.astype(str); a["ts"]=pd.to_datetime(a.ts).dt.tz_convert(TZ).dt.as_unit("ns")
smp=pd.read_csv(R+"01_participant_dataset/train/smp.csv"); smp["ts"]=pd.to_datetime(smp.ts).dt.tz_convert(TZ).dt.as_unit("ns")
h=smp.ts.dt.hour; smp["w"]=smp.smp*np.where((h>=11)&(h<=15),3,1); W=smp.set_index("ts").w
name="E12-10ctrx2"
cands={"ref":xr,"E12-10cx2":xc}
for key in ["wa_station","wb_station"]:
    for s in range(3):
        x=strat(key,s); f=cap[x].sum()/tot
        cands[f"strat_{key}_{s}"]=x
for k,x in cands.items(): print(k, f"G1 frac {cap[x].sum()/tot:.3f}", end="; ")
print()
for suf,cut in {"":"2024-06-01","_cut2023":"2023-06-01","_cut2022":"2022-06-01"}.items():
    s=pd.Timestamp(cut+" 01:00",tz=TZ); e=s+pd.DateOffset(years=1)
    aa=a[(a.ts>=s)&(a.ts<e)&a.energy.notna()]
    p=pd.read_parquet(R+f"outputs/preds/{name}{suf}.parquet"); p["site_id"]=p.site_id.astype(str)
    p["ts"]=pd.to_datetime(p.ts).dt.tz_convert(TZ).dt.as_unit("ns")
    m=aa.merge(p,on=["site_id","ts"]); m["e"]=m.pred_kwh-m.energy
    E=m.pivot(index="ts",columns="site_id",values="e").reindex(columns=sid).fillna(0.0)
    w=W.reindex(E.index).to_numpy(); Ev=E.to_numpy()
    lb=w@np.abs(Ev.sum(1))
    out=[]
    for k,x in cands.items():
        pen=w@(np.abs(Ev[:,x].sum(1))+np.abs(Ev[:,~x].sum(1)))
        out.append(f"{k} +{(pen/lb-1)*100:.2f}%")
    print(cut, " | ".join(out))
