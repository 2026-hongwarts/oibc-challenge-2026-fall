"""F13a_벌금분해 — 판 뒤집기 검토(docs/판_뒤집기_검토.md, 11/11 검토 에이전트 작성, 읽기 전용 분석).
벌금 분해(11~15시 몫, 하루 총량 vs 시각 배분, 하루 오차 자기상관, 달별 치우침)"""
import numpy as np, pandas as pd
R="C:/Users/109so/OneDrive/Desktop/학교/compete/oibc/"
TZ="Asia/Seoul"
sites=pd.read_csv(R+"01_participant_dataset/sites.csv")
sites["site_id"]=sites.site_id.astype(str)
comp=pd.read_csv(R+"outputs/compositions/E12-10cx2.csv"); comp["site_id"]=comp.site_id.astype(str)
cmap=comp.set_index("site_id").group
refmap=sites.set_index("site_id").current_group.map({"A":"G1","B":"G2"})
a=pd.read_parquet(R+"outputs/work/base.parquet",columns=["site_id","ts","energy","elev","capacity_kw"])
a["site_id"]=a.site_id.astype(str)
a["ts"]=pd.to_datetime(a.ts).dt.tz_convert(TZ).dt.as_unit("ns")
smp=pd.read_csv(R+"01_participant_dataset/train/smp.csv"); smp["ts"]=pd.to_datetime(smp.ts).dt.tz_convert(TZ).dt.as_unit("ns")
h=smp.ts.dt.hour; smp["k"]=np.where((h>=11)&(h<=15),3,1); smp["w"]=smp.smp*smp.k
W=smp.set_index("ts").w
name="E12-10ctrx2"
folds={"":"2024-06-01","_cut2023":"2023-06-01","_cut2022":"2022-06-01"}
for suf,cut in folds.items():
    s=pd.Timestamp(cut+" 01:00",tz=TZ); e=s+pd.DateOffset(years=1)
    aa=a[(a.ts>=s)&(a.ts<e)&a.energy.notna()]
    first=a[a.energy.notna()].groupby("site_id").ts.min()
    seen=set(first[first < s - pd.DateOffset(days=1)].index)  # crude: had data before window
    p=pd.read_parquet(R+f"outputs/preds/{name}{suf}.parquet"); p["site_id"]=p.site_id.astype(str)
    p["ts"]=pd.to_datetime(p.ts).dt.tz_convert(TZ).dt.as_unit("ns")
    m=aa.merge(p,on=["site_id","ts"],how="left")
    m["pred_kwh"]=m.pred_kwh.fillna(0)
    m["e"]=m.pred_kwh-m.energy
    m["g"]=m.site_id.map(cmap); m["gr"]=m.site_id.map(refmap)
    m["seen"]=m.site_id.isin(seen)
    w=W.reindex(m.ts.unique())
    def pen(gcol, df=m):
        G=df.groupby([gcol,"ts"]).e.sum().unstack(gcol).fillna(0)
        ww=W.reindex(G.index).to_numpy()
        return 0.1*float((ww[:,None]*np.abs(G.to_numpy())).sum()), G, ww
    pc,Gc,ww=pen("g"); pr,_,_=pen("gr")
    tot=m.groupby("ts").e.sum(); wt=W.reindex(tot.index).to_numpy()
    plb=0.1*float((wt*np.abs(tot.to_numpy())).sum())
    e1,e2=Gc["G1"].to_numpy(),Gc["G2"].to_numpy()
    opp=(np.sign(e1)!=np.sign(e2))
    print(f"\n== fold {cut} ({name}{suf}) sites {m.site_id.nunique()} seen {len(seen & set(m.site_id))}")
    print(f" penalty comp E12-10cx2 {pc/1e6:,.1f}M  ref {pr/1e6:,.1f}M  single-portfolio LB {plb/1e6:,.1f}M  comp/LB-1 = {(pc/plb-1)*100:.2f}%  ref/LB-1 = {(pr/plb-1)*100:.2f}%")
    print(f" opposite-sign hours (w-weighted share) {np.sum(ww*opp)/np.sum(ww):.3f}; loss from opp {0.1*np.sum(ww*opp*2*np.minimum(np.abs(e1),np.abs(e2)))/1e6:,.1f}M")
    # peak share
    hh=Gc.index.hour; pk=(hh>=11)&(hh<=15)
    print(f" peak(11-15) share of penalty {0.1*np.sum((ww*(np.abs(e1)+np.abs(e2)))[pk])/pc:.3f}")
    # daily oracle: scale group forecast per day to match daily actual
    m["day"]=(m.ts-pd.Timedelta(hours=1)).dt.tz_localize(None).dt.normalize()
    gd=m.groupby(["g","day"]).agg(P=("pred_kwh","sum"),A=("energy","sum"))
    m=m.join((gd.A/gd.P.replace(0,np.nan)).rename("sc"),on=["g","day"])
    m["e_d"]=m.pred_kwh*m.sc.fillna(1)-m.energy
    Gd=m.groupby(["g","ts"]).e_d.sum().unstack("g").fillna(0)
    pd_=0.1*float((W.reindex(Gd.index).to_numpy()[:,None]*np.abs(Gd.to_numpy())).sum())
    print(f" daily-total oracle (perfect daily group energy, keep hourly shape): {pd_/1e6:,.1f}M = {(pd_/pc-1)*100:.1f}% vs comp")
    # daily error stats: relative daily group error, lag-1 autocorrelation
    de=(gd.P-gd.A)/gd.A.replace(0,np.nan)
    for g in ["G1","G2"]:
        x=de.loc[g].dropna()
        print(f"  {g} daily rel err: mean {x.mean():+.3f} MAD {x.abs().mean():.3f} lag1 autocorr {x.autocorr(1):.3f}")
    # seen-only oracle per-site x hour x month bias removal (in-sample upper bound)
    m["hr"]=m.ts.dt.hour; m["mo"]=m.ts.dt.month
    b=m.groupby(["site_id","hr","mo"]).e.transform("mean")
    m["e_b"]=m.e-b
    Gb=m.groupby(["g","ts"]).e_b.sum().unstack("g").fillna(0)
    pb=0.1*float((W.reindex(Gb.index).to_numpy()[:,None]*np.abs(Gb.to_numpy())).sum())
    print(f" in-sample oracle site x hour x month mean-bias removal: {(pb/pc-1)*100:.1f}%")
    b2=m.groupby(["g","hr","mo"]).e.transform("mean")  # group-level hour x month
    m["e_b2"]=m.e - m.groupby(["site_id","hr","mo"]).e.transform("mean")*0  # placeholder
    Gg=m.groupby(["g","ts"]).e.sum().unstack("g").fillna(0)
    Gg2=Gg.copy()
    idx=Gg.index; key=pd.DataFrame({"hr":idx.hour,"mo":idx.month},index=idx)
    for g in ["G1","G2"]:
        Gg2[g]=Gg[g]-Gg[g].groupby([key.hr,key.mo]).transform("median")
    pg=0.1*float((W.reindex(Gg2.index).to_numpy()[:,None]*np.abs(Gg2.to_numpy())).sum())
    print(f" in-sample oracle group x hour x month median-bias removal: {(pg/pc-1)*100:.1f}%")
    # unseen sites share
    Gs=m.groupby(["g","seen","ts"]).e.sum()
    print(f" seen rows share {m.seen.mean():.3f}")
