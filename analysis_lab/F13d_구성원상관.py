"""F13d_구성원상관 — 판 뒤집기 검토(docs/판_뒤집기_검토.md, 11/11 검토 에이전트 작성, 읽기 전용 분석).
구성원 간 한낮 묶음 오차 상관, 가장 좋은 단일 모델"""
import numpy as np, pandas as pd
R="C:/Users/109so/OneDrive/Desktop/학교/compete/oibc/"
TZ="Asia/Seoul"
comp=pd.read_csv(R+"outputs/compositions/E12-10cx2.csv"); comp["site_id"]=comp.site_id.astype(str)
cmap=comp.set_index("site_id").group
a=pd.read_parquet(R+"outputs/work/base.parquet",columns=["site_id","ts","energy"])
a["site_id"]=a.site_id.astype(str); a["ts"]=pd.to_datetime(a.ts).dt.tz_convert(TZ).dt.as_unit("ns")
smp=pd.read_csv(R+"01_participant_dataset/train/smp.csv"); smp["ts"]=pd.to_datetime(smp.ts).dt.tz_convert(TZ).dt.as_unit("ns")
h=smp.ts.dt.hour; smp["w"]=smp.smp*np.where((h>=11)&(h<=15),3,1); W=smp.set_index("ts").w
mem=["E11-12x2","E11-11x2","G5bT1x2","S3x2","V07T1x2","E12-10cx2"]
for suf,cut in {"":"2024-06-01","_cut2023":"2023-06-01","_cut2022":"2022-06-01"}.items():
    s=pd.Timestamp(cut+" 01:00",tz=TZ); e=s+pd.DateOffset(years=1)
    aa=a[(a.ts>=s)&(a.ts<e)&a.energy.notna()].copy(); aa["g"]=aa.site_id.map(cmap)
    GA=aa.groupby(["g","ts"]).energy.sum()
    errs={}
    for n in mem:
        p=pd.read_parquet(R+f"outputs/preds/{n}{suf}.parquet",columns=["site_id","ts","pred_kwh"]); p["site_id"]=p.site_id.astype(str)
        p["ts"]=pd.to_datetime(p.ts).dt.tz_convert(TZ).dt.as_unit("ns")
        m=aa[["site_id","ts","g"]].merge(p,on=["site_id","ts"],how="left")
        GP=m.groupby(["g","ts"]).pred_kwh.sum()
        errs[n]=(GP-GA)
    E=pd.DataFrame(errs); w=W.reindex(E.index.get_level_values("ts")).to_numpy()
    pen={n:0.1*float((w*E[n].abs()).sum()) for n in mem}
    ens=pen["E12-10cx2"]
    print(f"\n== {cut}: group penalty (M) ", {k:round(v/1e6,1) for k,v in pen.items()})
    best=min(v for k,v in pen.items() if k!="E12-10cx2")
    print(f" ensemble vs best member {(ens/best-1)*100:+.2f}%  vs mean member {(ens/np.mean([v for k,v in pen.items() if k!='E12-10cx2'])-1)*100:+.2f}%")
    pk=(E.index.get_level_values("ts").hour>=11)&(E.index.get_level_values("ts").hour<=15)
    print(" peak-hour group error correlation:\n", E[pk][mem[:-1]].corr().round(3).to_string())
