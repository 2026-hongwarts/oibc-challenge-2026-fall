"""F13c_휴일계절 — 판 뒤집기 검토(docs/판_뒤집기_검토.md, 11/11 검토 에이전트 작성, 읽기 전용 분석).
봄·가을 휴일 한낮 예측÷실적(추세 보정 뒤)"""
import numpy as np, pandas as pd, re
R="C:/Users/109so/OneDrive/Desktop/학교/compete/oibc/"
TZ="Asia/Seoul"
src=open(R+"src_tools/features_v4.py",encoding="utf-8").read()
HOL=set(re.findall(r'"(\d{4}-\d{2}-\d{2})"', src.split("HOLIDAYS")[1].split("dtype")[0]))
sites=pd.read_csv(R+"01_participant_dataset/sites.csv"); sites["site_id"]=sites.site_id.astype(str)
comp=pd.read_csv(R+"outputs/compositions/E12-10cx2.csv"); comp["site_id"]=comp.site_id.astype(str)
cmap=comp.set_index("site_id").group
a=pd.read_parquet(R+"outputs/work/base.parquet",columns=["site_id","ts","energy"])
a["site_id"]=a.site_id.astype(str); a["ts"]=pd.to_datetime(a.ts).dt.tz_convert(TZ).dt.as_unit("ns")
smp=pd.read_csv(R+"01_participant_dataset/train/smp.csv"); smp["ts"]=pd.to_datetime(smp.ts).dt.tz_convert(TZ).dt.as_unit("ns")
h=smp.ts.dt.hour; smp["w"]=smp.smp*np.where((h>=11)&(h<=15),3,1); W=smp.set_index("ts")
SEAS={12:"W",1:"W",2:"W",3:"Sp",4:"Sp",5:"Sp",6:"Su",7:"Su",8:"Su",9:"Au",10:"Au",11:"Au"}
name="E12-10ctrx2"
for suf,cut in {"":"2024-06-01","_cut2023":"2023-06-01","_cut2022":"2022-06-01"}.items():
    s=pd.Timestamp(cut+" 01:00",tz=TZ); e=s+pd.DateOffset(years=1)
    aa=a[(a.ts>=s)&(a.ts<e)&a.energy.notna()]
    p=pd.read_parquet(R+f"outputs/preds/{name}{suf}.parquet"); p["site_id"]=p.site_id.astype(str)
    p["ts"]=pd.to_datetime(p.ts).dt.tz_convert(TZ).dt.as_unit("ns")
    m=aa.merge(p,on=["site_id","ts"]); m["g"]=m.site_id.map(cmap)
    G=m.groupby(["g","ts"]).agg(P=("pred_kwh","sum"),A=("energy","sum")).reset_index()
    G=G.join(W[["w","smp"]],on="ts")
    d=(G.ts-pd.Timedelta(hours=1)).dt.tz_localize(None).dt.normalize()
    G["off"]=(d.dt.dayofweek>=5)|d.dt.strftime("%Y-%m-%d").isin(HOL)
    G["se"]=G.ts.dt.month.map(SEAS); G["hr"]=G.ts.dt.hour
    G["pen"]=0.1*G.w*(G.P-G.A).abs()
    tot=G.pen.sum()
    mid=G[(G.hr>=10)&(G.hr<=16)]
    t=mid.groupby(["se","off"]).agg(P=("P","sum"),A=("A","sum"),pen=("pen","sum"))
    t["ratio"]=t.P/t.A; t["pen_share"]=t.pen/tot
    # in-sample best multiplier per cell
    best=[]
    for (se,off),grp in mid.groupby(["se","off"]):
        cs=np.arange(0.85,1.051,0.005)
        pens=[(0.1*grp.w*(grp.P*c-grp.A).abs()).sum() for c in cs]
        i=int(np.argmin(pens)); best.append((se,off,cs[i],(pens[i]-grp.pen.sum())/tot*100))
    t["best_c"]=[b[2] for b in best]; t["gain%_of_total"]=[b[3] for b in best]
    print(f"\n== {cut} total pen {tot/1e6:.1f}M"); print(t[["ratio","pen_share","best_c","gain%_of_total"]].round(3).to_string())
    print(" monthly pred/actual 10-16h:", (mid.groupby(mid.ts.dt.month).P.sum()/mid.groupby(mid.ts.dt.month).A.sum()).round(3).to_dict())
