# smc_dashboard.py
# -*- coding: utf-8 -*-
"""
SMC Intraday — BTC / ETH / XAUUSD / XAUEUR / EURUSD (text)
Ядро подтверждений: SFP/BOS/FVG (3/3 = готово; 2/3 = ждём). Журнал (SQLite, ID).
Telegram-алерты при росте числа подтверждений. Источник: yfinance.
"""

from __future__ import annotations
import time, sqlite3, json, hashlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np, pandas as pd, streamlit as st, yfinance as yf, requests

# ---------------- CFG ----------------
ASSETS = ["BTCUSDT","ETHUSDT","XAUUSD","XAUEUR","EURUSD"]
YF_TICKER_CANDIDATES = {
    "BTCUSDT":["BTC-USD"], "ETHUSDT":["ETH-USD"],
    "XAUUSD":["XAUUSD=X","GC=F"], "XAUEUR":["XAUEUR=X"], "EURUSD":["EURUSD=X"]
}
TF_FALLBACKS = {"5m":[("5m","60d"),("15m","60d"),("60m","730d")],
                "15m":[("15m","60d"),("60m","730d")],
                "1h":[("60m","730d"),("1d","730d")]}
HTF_OF = {"5m":"15m","15m":"60m","1h":"1d"}
DB = "smc_journal.sqlite"

# окна валидности ядра по TF (в барах)
CORE_WINDOWS = {
    "5m":{"sfp":120,"bos":150,"fvg":100},
    "15m":{"sfp":160,"bos":200,"fvg":120},
    "1h":{"sfp":220,"bos":260,"fvg":160},
}

# ---------------- INDICATORS ----------------
ema = lambda x,n: x.ewm(span=n,adjust=False).mean()

def rsi(x,n=14):
    d=x.diff(); up=(d.clip(lower=0)).ewm(alpha=1/n,adjust=False).mean()
    dn=(-d.clip(upper=0)).ewm(alpha=1/n,adjust=False).mean()
    rs=up/(dn.replace(0,np.nan)); y=100-100/(1+rs)
    return y.fillna(method="bfill").fillna(50)

def macd(x):
    f=ema(x,12); s=ema(x,26); m=f-s; return m,ema(m,9),m-ema(m,9)

def atr(df,n=14):
    c=df.close
    tr=pd.concat([(df.high-df.low),(df.high-c.shift()).abs(),(df.low-c.shift()).abs()],axis=1).max(axis=1)
    return tr.ewm(alpha=1/n,adjust=False).mean()

def obv(df):
    s=np.sign(df.close.diff().fillna(0.0)); return (s*df.volume).cumsum()

def adx(df,n=14):
    up=df.high.diff(); dn=-df.low.diff()
    plus=np.where((up>dn)&(up>0),up,0.0); minus=np.where((dn>up)&(dn>0),dn,0.0)
    tr=pd.concat([(df.high-df.low),(df.high-df.close.shift()).abs(),(df.low-df.close.shift()).abs()],axis=1).max(axis=1)
    a=tr.ewm(alpha=1/n,adjust=False).mean()
    p=100*pd.Series(plus,index=df.index).ewm(alpha=1/n,adjust=False).mean()/a
    m=100*pd.Series(minus,index=df.index).ewm(alpha=1/n,adjust=False).mean()/a
    d=100*(p.subtract(m).abs()/(p+m).replace(0,np.nan))
    return d.ewm(alpha=1/n,adjust=False).mean().fillna(20)

def vwap_series(df):
    tp=(df.high+df.low+df.close)/3.0; vol=df.volume.replace(0,np.nan).fillna(0.0)
    num=(tp*vol).cumsum(); den=vol.cumsum().replace(0,np.nan)
    return (num/den).fillna(method="bfill").fillna(df.close)

def slope(series,last_n=80):
    n=min(len(series),last_n)
    if n<8: return 0.0
    y=series.tail(n).values; x=np.arange(n,dtype=float); return float(np.polyfit(x,y,1)[0])

# ---------------- SMC PRIMS ----------------
def swings(df,L=3,R=3):
    hi,lo=df.high.values,df.low.values; n=len(df)
    SH=np.zeros(n,bool); SL=np.zeros(n,bool)
    for i in range(L,n-R):
        if hi[i]==hi[i-L:i+R+1].max(): SH[i]=True
        if lo[i]==lo[i-L:i+R+1].min(): SL[i]=True
    return pd.Series(SH,df.index),pd.Series(SL,df.index)

def last_swing_levels(df,SH,SL):
    sh_idx=SH[SH].index; sl_idx=SL[SL].index
    sh=float(df.loc[sh_idx[-1],"high"]) if len(sh_idx) else None
    sl=float(df.loc[sl_idx[-1],"low"])  if len(sl_idx) else None
    return sh,sl

def bos(df,SH,SL,look=200,confirm_mult=0.30):
    recent=df.iloc[-look:]
    sh_idx=recent[SH.loc[recent.index]].index; sl_idx=recent[SL.loc[recent.index]].index
    last_sh=recent.loc[sh_idx[-1]] if len(sh_idx) else None
    last_sl=recent.loc[sl_idx[-1]] if len(sl_idx) else None
    a=float(atr(df).iloc[-1]) or 1e-6
    if last_sh is not None:
        lvl=last_sh.high; post=recent[recent.index>last_sh.name]
        brk=post[post.close>lvl+confirm_mult*a]
        if len(brk): return "up",brk.index[0],lvl
    if last_sl is not None:
        lvl=last_sl.low; post=recent[recent.index>last_sl.name]
        brk=post[post.close<lvl-confirm_mult*a]
        if len(brk): return "down",brk.index[0],lvl
    return None,None,None

def sweeps(df,SH,SL,win=180):
    res={"high":[],"low":[]}; rec=df.iloc[-win:]
    for t in rec[SH.loc[rec.index]].index:
        level=df.loc[t,"high"]; post=rec[rec.index>t]
        if len(post[(post.high>level)&(post.close<level)]): res["high"].append((t,level))
    for t in rec[SL.loc[rec.index]].index:
        level=df.loc[t,"low"]; post=rec[rec.index>t]
        if len(post[(post.low<level)&(post.close>level)]): res["low"].append((t,level))
    return res

def fvg(df,look=140):
    out={"bull":[],"bear":[]}; hi,lo,idx=df.high.values,df.low.values,df.index
    n=len(df); s=max(2,n-look)
    for i in range(s,n):
        if i-2>=0 and lo[i]>hi[i-2]: out["bull"].append((idx[i],hi[i-2],lo[i]))
        if i-2>=0 and hi[i]<lo[i-2]: out["bear"].append((idx[i],hi[i],lo[i-2]))
    return out

def simple_ob(df,dir_,t,back=70):
    res={"demand":None,"supply":None}
    if dir_ is None or t is None: return res
    before=df[df.index<t].iloc[-back:]
    if dir_=="up":
        reds=before[before.close<before.open]
        if len(reds):
            last=reds.iloc[-1]
            res["demand"]=(last.name,float(min(last.open,last.close)),float(max(last.open,last.close)))
    if dir_=="down":
        greens=before[before.close>before.open]
        if len(greens):
            last=greens.iloc[-1]
            res["supply"]=(last.name,float(min(last.open,last.close)),float(max(last.open,last.close)))
    return res

def volume_profile(df,bins=40):
    lo=float(df.low.min()); hi=float(df.high.max())
    if hi<=lo: hi=lo+1e-6
    edges=np.linspace(lo,hi,bins+1); prices=df.close.values; vols=df.volume.values.astype(float)
    if np.nansum(vols)<=1e-12: vols=np.ones_like(vols,float)
    vol=np.zeros(bins,float); idx=np.clip(np.digitize(prices,edges)-1,0,bins-1)
    for i,v in zip(idx,vols): vol[i]+=float(v)
    total=max(vol.sum(),1.0); poc_i=int(vol.argmax()); poc=(edges[poc_i]+edges[poc_i+1])/2
    area=[poc_i]; L=poc_i-1; R=poc_i+1; acc=vol[poc_i]
    while acc<0.7*total and (L>=0 or R<bins):
        if R>=bins or (L>=0 and vol[L]>=vol[R]): area.append(L); acc+=vol[L]; L-=1
        else: area.append(R); acc+=vol[R]; R+=1
    val=edges[max(min(area),0)]; vah=edges[min(max(area)+1,bins)]
    return {"edges":edges,"volume":vol,"poc":float(poc),"val":float(val),"vah":float(vah)}

# ---------------- CONTEXT ----------------
def score_bias(df):
    c=df.close; s=0; r=float(rsi(c).iloc[-1]); h=float(macd(c)[2].iloc[-1])
    if r>55: s+=1
    elif r<45: s-=1
    if h>0: s+=1
    elif h<0: s-=1
    return "long" if s>=1 else ("short" if s<=-1 else "none")

def regime_daily(df_d):
    e50=ema(df_d.close,50).iloc[-1]; e200=ema(df_d.close,200).iloc[-1]
    if np.isnan(e50) or np.isnan(e200): return "none"
    return "long" if e50>e200 else "short" if e50<e200 else "none"

def market_regime(df,vp):
    ad=float(adx(df).iloc[-1]); p=float(df.close.iloc[-1])
    outside=(p>vp["vah"]) or (p<vp["val"])
    return "trend" if (ad>=22 or outside) else "range"

# ---------------- STRICT VALIDATION ----------------
def validate_sfp(df,SH,SL,atr_val,tf):
    w=CORE_WINDOWS[tf]["sfp"]; res={"up":None,"down":None,"notes":[]}
    rec=df.iloc[-w:]
    # last highs/lows
    highs=[(t,float(df.loc[t,"high"])) for t in rec[SH.loc[rec.index]].index]
    lows=[(t,float(df.loc[t,"low"])) for t in rec[SL.loc[rec.index]].index]
    if highs:
        t,lvl=highs[-1]
        post=rec[rec.index>t]
        hit=post[(post.high>lvl) & (post.close<lvl)]
        if len(hit) and (float(hit.high.iloc[-1])-lvl)>=0.15*atr_val:
            res["down"]=(t,lvl); res["notes"].append("SFP high валиден")
    if lows:
        t,lvl=lows[-1]
        post=rec[rec.index>t]
        hit=post[(post.low<lvl) & (post.close>lvl)]
        if len(hit) and (lvl-float(hit.low.iloc[-1]))>=0.15*atr_val:
            res["up"]=(t,lvl); res["notes"].append("SFP low валиден")
    return res

def validate_bos(df,SH,SL,atr_val,tf):
    w=CORE_WINDOWS[tf]["bos"]; dir_,t,lvl=bos(df,SH,SL,look=w,confirm_mult=0.30)
    notes=[]
    if dir_ is None: notes.append("BOS нет")
    else: notes.append(f"BOS {dir_}")
    return dir_,t,lvl,notes

def validate_fvg(direction,gaps,price,vp,atr_val,tf,bos_time):
    w=CORE_WINDOWS[tf]["fvg"]; notes=[]
    arr=gaps["bull"] if direction=="up" else gaps["bear"]
    if not arr: return None,["FVG отсутствует"]
    arr=list(reversed(arr))
    for t,lo,hi in arr:
        if bos_time and t<bos_time: continue
        mid=(lo+hi)/2
        depth=abs(price-mid)/max(atr_val,1e-9)
        if depth<0.25: continue
        side=(price>vp["poc"] and direction=="up") or (price<vp["poc"] and direction=="down") or abs(price-vp["poc"])<=0.6*atr_val
        if not side: continue
        notes.append("FVG валиден"); return (t,lo,hi),notes
    return None,["валидного FVG нет"]

# ---------------- MODELS ----------------
@dataclass
class Scenario:
    name:str; bias:str; etype:str; trigger:str; entry:float; sl:float; tp1:float; tp2:Optional[float]
    rr:str; confirms:int; confirm_list:List[str]; explain_short:str; stop_reason:str; tp_reason:str
    status:str

def rr_targets(entry,sl,bias,min_rr=2.0):
    risk=abs(entry-sl) or 1e-9
    tp1 = entry + (min_rr*risk if bias=="long" else -min_rr*risk)
    tp2 = entry + (3.0*risk if bias=="long" else -3.0*risk)
    return tp1,tp2,f"1:{int(min_rr)}/1:3"

def propose(df,tf,htf_bias,d_bias,regime,vp,obv_slope_val) -> List[Scenario]:
    c=float(df.close.iloc[-1]); at=float(atr(df).iloc[-1]); at=max(at,1e-9)
    SH,SL=swings(df); sh_lvl,sl_lvl=last_swing_levels(df,SH,SL)
    vw=float(vwap_series(df).iloc[-1]); ema20=ema(df.close,20); ema_up=slope(ema20)>0; ema_dn=slope(ema20)<0
    dir_bos,t_bos,lvl_bos,bos_notes=validate_bos(df,SH,SL,at,tf)
    swp=validate_sfp(df,SH,SL,at,tf)
    gaps=fvg(df)
    core_conf=[],[]
    S:List[Scenario]=[]

    def generic(bias,entry):
        out=[]
        if (bias=="long" and obv_slope_val>0) or (bias=="short" and obv_slope_val<0): out.append("OBV в сторону")
        if (bias=="long" and ema_up) or (bias=="short" and ema_dn): out.append("тренд EMA20")
        if (bias=="long" and c>vp["poc"]) or (bias=="short" and c<vp["poc"]): out.append("сторона POC")
        if abs(entry-vw)<=0.6*at: out.append("рядом VWAP")
        return out

    def struct_tp(entry,bias):
        if bias=="long":
            highs=[float(df.loc[i,"high"]) for i in SH[SH].index if float(df.loc[i,"high"])>entry]
            tgt=min(highs) if highs else vp["poc"]
        else:
            lows=[float(df.loc[i,"low"]) for i in SL[SL].index if float(df.loc[i,"low"])<entry]
            tgt=max(lows) if lows else vp["poc"]
        return tgt,("свинг" if tgt!=vp["poc"] else "POC")

    # core building helpers
    def core_list_for(bias):
        core=[]
        # SFP
        if bias=="long" and swp["up"]: core.append("SFP")
        if bias=="short" and swp["down"]: core.append("SFP")
        # BOS
        if (bias=="long" and dir_bos=="up") or (bias=="short" and dir_bos=="down"): core.append("BOS")
        # FVG
        fv,notes=validate_fvg("up" if bias=="long" else "down",gaps,price=c,vp=vp,atr_val=at,tf=tf,bos_time=t_bos)
        if fv: core.append("FVG")
        return core, fv

    # LONG: SFP/BOS/FVG
    coreL, fL = core_list_for("long")
    if len(coreL)>=2:
        entry=None; sl=None; trig=[]; expl=[]
        if fL: 
            _,lo,hi=fL; mid=(lo+hi)/2; entry=mid+0.0*at; trig.append(f"0.5 FVG {mid:.5f}"); expl.append("имбаланс")
        if swp["up"]: sl=swp["up"][1]-0.7*at; expl.append("свип low"); trig.append("после SFP")
        if dir_bos=="up": expl.append("BOS↑")
        if entry and sl:
            tp1,tp2,rr=rr_targets(entry,sl,"long"); tps,treason=struct_tp(entry,"long"); tp1=tps
            conf=coreL+generic("long",entry)
            status="ok" if len(coreL)>=3 else "await"
            S.append(Scenario("Core LONG", "long", "limit", ", ".join(trig), float(entry), float(sl), float(tp1), float(tp2), rr, len(coreL), conf, "ядро LONG", "под SFP −0.7×ATR", treason, status))

    # SHORT: SFP/BOS/FVG
    coreS, fS = core_list_for("short")
    if len(coreS)>=2:
        entry=None; sl=None; trig=[]; expl=[]
        if fS:
            _,lo,hi=fS; mid=(lo+hi)/2; entry=mid-0.0*at; trig.append(f"0.5 FVG {mid:.5f}"); expl.append("имбаланс")
        if swp["down"]: sl=swp["down"][1]+0.7*at; expl.append("свип high"); trig.append("после SFP")
        if dir_bos=="down": expl.append("BOS↓")
        if entry and sl:
            tp1,tp2,rr=rr_targets(entry,sl,"short"); tps,treason=struct_tp(entry,"short"); tp1=tps
            conf=coreS+generic("short",entry)
            status="ok" if len(coreS)>=3 else "await"
            S.append(Scenario("Core SHORT", "short", "limit", ", ".join(trig), float(entry), float(sl), float(tp1), float(tp2), rr, len(coreS), conf, "ядро SHORT", "над SFP +0.7×ATR", treason, status))

    # Range VA reversion как дополнительная
    if regime=="range" and float(adx(df).iloc[-1])<22:
        if abs(c-vp["val"])<=max(0.6*at,0.1*(vp["vah"]-vp["val"])):
            e=vp["val"]+0.1*at; sl=vp["val"]-0.8*at; tp1,tp2,rr=rr_targets(e,sl,"long")
            S.append(Scenario("Value Area Reversion","long","limit","от VAL к POC",float(e),float(sl),float(tp1),float(tp2),rr,1,["VAL","range"],"VA→POC","под VAL −0.8×ATR","POC","await"))
        if abs(c-vp["vah"])<=max(0.6*at,0.1*(vp["vah"]-vp["val"])):
            e=vp["vah"]-0.1*at; sl=vp["vah"]+0.8*at; tp1,tp2,rr=rr_targets(e,sl,"short")
            S.append(Scenario("Value Area Reversion","short","limit","от VAH к POC",float(e),float(sl),float(tp1),float(tp2),rr,1,["VAH","range"],"VA→POC","над VAH +0.8×ATR","POC","await"))

    # сортировка: сначала готовые ядра, затем ожидание
    S.sort(key=lambda x:(-(x.confirms>=3 and x.status=="ok"), -x.confirms))
    return S

# ---------------- PROBS ----------------
def scenario_probabilities(scen,htf_bias,d_bias,obv_slope_val,price,vp,atr_val,regime,cap=0.9,floor=0.05,temp=1.12):
    if not scen: return {"Wait (no-trade)":100.0},{"long":0.0,"short":0.0}
    scores,labels=[],[]
    for s in scen:
        if s.name.startswith("Wait"): continue
        sc = 1.4*s.confirms + (1.2 if s.bias==htf_bias else 0) + (0.8 if s.bias==d_bias else 0)
        sc += 0.6 if ((obv_slope_val>0 and s.bias=="long") or (obv_slope_val<0 and s.bias=="short")) else -0.2
        if (s.name.startswith("Core") and regime=="trend") or (s.name.startswith("Value Area") and regime=="range"): sc += 1.0
        dist=abs(s.entry-price)/max(atr_val,1e-6); sc += (-1.0 if dist>2.0 else (-0.5 if dist>1.5 else 0))
        scores.append(sc); labels.append((s.name,s.bias))
    if not scores: return {"Wait (no-trade)":100.0},{"long":0.0,"short":0.0}
    scores=np.array(scores)/temp; ex=np.exp(scores-scores.max()); p=np.clip(ex/ex.sum(),floor,cap); p=p/p.sum()
    out,agg={},{"long":0.0,"short":0.0}
    for (lbl,bias),pp in zip(labels,p): val=float(np.round(pp*100.0,2)); out[f"{lbl} ({bias})"]=val; agg[bias]+=val
    return dict(sorted(out.items(),key=lambda x:x[1],reverse=True)),{k:round(v,2) for k,v in agg.items()}

# ---------------- DATA ----------------
@st.cache_data(show_spinner=False,ttl=60)
def yf_ohlc_first_success(asset_key,tf,limit=800):
    cands=YF_TICKER_CANDIDATES.get(asset_key,[asset_key]); tries=TF_FALLBACKS.get(tf,TF_FALLBACKS["15m"]); last_err=None
    for tkr in cands:
        for interval,period in tries:
            try:
                df=yf.download(tkr,interval=interval,period=period,auto_adjust=False,progress=False)
                if df.empty: last_err=f"{tkr}@{interval}/{period}: пусто"; continue
                if isinstance(df.columns,pd.MultiIndex): df.columns=df.columns.get_level_values(0)
                df=df.rename(columns={"Open":"open","High":"high","Low":"low","Close":"close","Adj Close":"adj_close","Volume":"volume"})
                need=["open","high","low","close","volume"]
                for c in need:
                    if c not in df.columns: df[c]=0.0 if c=="volume" else np.nan
                    df[c]=pd.to_numeric(df[c],errors="coerce")
                try:
                    df.index=pd.to_datetime(df.index,utc=True) if df.index.tz is None else df.index.tz_convert("UTC")
                except Exception:
                    df.index=pd.to_datetime(df.index,utc=True)
                out=df[need].dropna().tail(limit)
                if out.empty: last_err=f"{tkr}@{interval}/{period}: после очистки нет"; continue
                return out,interval,period
            except Exception as e:
                last_err=f"{tkr}@{interval}/{period}: {e}"; continue
    raise RuntimeError(f"yfinance: нет данных для {asset_key}. {last_err}")

# ---------------- DB ----------------
def db_init():
    con=sqlite3.connect(DB); cur=con.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS trades (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts TEXT, asset TEXT, tf TEXT, name TEXT, bias TEXT,
        entry REAL, sl REAL, tp1 REAL, tp2 REAL,
        confirms INTEGER, reasons TEXT, status TEXT,
        result TEXT, result_ts TEXT)""")
    con.commit(); con.close()

def db_insert(row:dict) -> int:
    con=sqlite3.connect(DB); cur=con.cursor()
    cur.execute("""INSERT INTO trades (ts,asset,tf,name,bias,entry,sl,tp1,tp2,confirms,reasons,status,result,result_ts)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (row["ts"],row["asset"],row["tf"],row["name"],row["bias"],row["entry"],row["sl"],row["tp1"],row["tp2"],
                 row["confirms"],json.dumps(row["reasons"],ensure_ascii=False),row["status"],row.get("result","open"),row.get("result_ts","")))
    con.commit(); rid=cur.lastrowid; con.close(); return rid

def db_fetch(limit=200):
    con=sqlite3.connect(DB); cur=con.cursor()
    cur.execute("""SELECT id,ts,asset,tf,name,bias,entry,sl,tp1,tp2,confirms,reasons,status,result,result_ts
                   FROM trades ORDER BY id DESC LIMIT ?""",(limit,))
    rows=cur.fetchall(); con.close()
    cols=["id","ts","asset","tf","name","bias","entry","sl","tp1","tp2","confirms","reasons","status","result","result_ts"]
    return [dict(zip(cols,r)) for r in rows]

def db_mark_result(trade_id:int, result:str):
    con=sqlite3.connect(DB); cur=con.cursor()
    cur.execute("UPDATE trades SET result=?, result_ts=? WHERE id=?",(result,time.strftime("%Y-%m-%d %H:%M:%S"),trade_id))
    con.commit(); con.close()

# ---------------- UI UTILS ----------------
def infer_decimals(df,asset):
    x=df.close.tail(300).diff().abs(); step=float(np.nanmin(x[x>0])) if np.any(x>0) else 0.0
    if step>0: p=max(2,min(6,int(np.ceil(-np.log10(step))+1)))
    else: p={"EURUSD":5,"XAUEUR":5,"XAUUSD":2,"BTCUSDT":2,"ETHUSDT":2}.get(asset,4)
    return p

def fmt_price(x,d):
    s=f"{x:.{d}f}"; a,b=s.split(".") if "." in s else (s,"")
    a=f"{int(float(a)):,}".replace(","," ")
    return f"{a}.{b}" if b else a

# ---------------- UI ----------------
st.set_page_config(page_title="SMC Intraday (text)", layout="wide")
st.title("SMC Intraday — BTC / ETH / XAUUSD / XAUEUR / EURUSD (text)")

colA,colB,colC,colD,colE=st.columns([1.2,1,1,1,1])
with colA: asset=st.selectbox("Актив",ASSETS,index=4 if "EURUSD" in ASSETS else 0)
with colB: tf=st.selectbox("TF",["5m","15m","1h"],index=0)
with colC: min_risk_pct=st.slider("Мин. риск (%ATR)",5,60,25,step=5)
with colD: min_tp1_atr=st.slider("Мин. TP1 (×ATR)",1.0,3.0,1.5,step=0.25)
with colE: min_confirms=st.slider("Мин. ядро-подтв.",2,3,3,step=1)

colF,colG=st.columns([1,1])
with colF: refresh_mode=st.selectbox("Обновление",["Выключено","30s","1m","2m","5m"],index=0)
with colG:
    tg_token=st.text_input("Telegram bot token (опц.)", value="", type="password")
    tg_chat=st.text_input("Telegram chat id (опц.)", value="")

if st.button("🔄 Обновить"): st.cache_data.clear(); st.experimental_rerun()
INTERVALS={"30s":30,"1m":60,"2m":120,"5m":300}
if "next_refresh_ts" not in st.session_state: st.session_state.next_refresh_ts=time.time()+10**9
if refresh_mode!="Выключено":
    it=INTERVALS[refresh_mode]; now=time.time()
    if now>=st.session_state.next_refresh_ts:
        st.session_state.next_refresh_ts=now+it; st.cache_data.clear(); st.experimental_rerun()
else:
    st.session_state.next_refresh_ts=time.time()+10**9

beginner_mode=st.checkbox("Простой режим",value=True)
with st.expander("📘 Словарь/правила"):
    st.markdown("- Ядро подтверждений: **SFP, BOS, FVG**. Готово при **3/3**. При **2/3** сценарий показывается как «ждём 3».")
    st.markdown("- Вход по **0.5 FVG**, стоп за свип-уровень ±ATR, TP1 = ближайший **свинг** или POC.")
    st.markdown("- Алерты в Telegram — при **росте** числа ядерных подтверждений.")

st.caption("Сценарии с риском ниже %ATR и с TP1 меньше заданного множителя ATR скрываются. Вероятности относительные (softmax).")

# ---------------- MAIN ----------------
db_init()
if "notify_state" not in st.session_state: st.session_state.notify_state={}
summary=[]
try:
    df,tf_eff,_=yf_ohlc_first_success(asset,tf,limit=800)
    htf=HTF_OF[tf]; df_h,_,_=yf_ohlc_first_success(asset,htf,limit=400)
    df_d,_,_=yf_ohlc_first_success(asset,"1d",limit=600)

    price=float(df.close.iloc[-1]); vp=volume_profile(df); reg=market_regime(df,vp); atr_v=float(atr(df).iloc[-1])
    obv_s=slope(obv(df),160); htf_bias=score_bias(df_h); d_bias=regime_daily(df_d)
    scen_all=propose(df,tf,htf_bias,d_bias,reg,vp,obv_s)

    min_risk=atr_v*(min_risk_pct/100.0); scen=[]; awaiting=[]
    for s in scen_all:
        risk_ok=abs(s.entry-s.sl)>=min_risk
        tp1_ok=(abs(s.tp1-s.entry)/max(atr_v,1e-6))>=min_tp1_atr
        if risk_ok and tp1_ok:
            if s.confirms>=min_confirms and s.status=="ok": scen.append(s)
            else: awaiting.append(s)
    if not scen: scen=awaiting or scen_all

    probs,balance=scenario_probabilities(scen,htf_bias,d_bias,obv_s,price,vp,atr_v,reg)
    d=infer_decimals(df,asset)

    st.markdown(f"## {asset} ({tf}) — цена: {fmt_price(price,d)}")
    ltf_b=score_bias(df); poc_state="выше VAH" if price>vp["vah"] else ("ниже VAL" if price<vp["val"] else "внутри value area")
    st.markdown(f"**Контекст:** LTF={ltf_b.upper()}, HTF={htf_bias.upper()}, Daily={d_bias.upper()} • Режим: {reg.upper()} (ADX≈{float(adx(df).iloc[-1]):.1f}) • POC {fmt_price(vp['poc'],d)}, VAL {fmt_price(vp['val'],d)}, VAH {fmt_price(vp['vah'],d)} → цена {poc_state}.")

    # Главная карточка + алерты по росту подтверждений
    if scen:
        top_key=list(probs.keys())[0] if probs else f"{scen[0].name} ({scen[0].bias})"
        main=next((s for s in scen if f"{s.name} ({s.bias})"==top_key),scen[0])
        rr=round(abs(main.tp1-main.entry)/max(abs(main.entry-main.sl),1e-9),2)
        txt = "ГОТОВО (3/3)" if (main.confirms>=3 and main.status=="ok") else f"{min(main.confirms,3)}/3 — ждём"
        st.markdown(f"### {'LONG' if main.bias=='long' else 'SHORT'} — {txt}")
        st.markdown(f"- **Подтв. ядра:** {main.confirms}/3 — {', '.join([x for x in main.confirm_list if x in ('SFP','BOS','FVG')])}  \n"
                    f"- **Доп. контекст:** {', '.join([x for x in main.confirm_list if x not in ('SFP','BOS','FVG')]) or '—'}  \n"
                    f"- **Вход:** {fmt_price(main.entry,d)} • **Стоп:** {fmt_price(main.sl,d)} (риск≈{fmt_price(abs(main.entry-main.sl),d)}, ATR≈{fmt_price(atr_v,d)})  \n"
                    f"- **Цели:** TP1 {fmt_price(main.tp1,d)} ({main.tp_reason}), TP2 {fmt_price(main.tp2,d) if main.tp2 else '—'} • **R:R≈{rr}**")

        row={"ts":str(df.index[-1]),"asset":asset,"tf":tf,"name":main.name,"bias":main.bias,"entry":main.entry,"sl":main.sl,"tp1":main.tp1,"tp2":main.tp2,"confirms":main.confirms,"reasons":main.confirm_list,"status":main.status}
        trade_id=db_insert(row)

        key=(asset,tf,main.name,main.bias)
        last=st.session_state.notify_state.get(key,0)
        if main.confirms>last and tg_token and tg_chat:
            try:
                msg=f"{asset} {tf}: {main.name} {main.bias} — подтверждения {last}→{main.confirms} (/3). Entry {main.entry:.5f}, SL {main.sl:.5f}, id {trade_id}"
                requests.get(f"https://api.telegram.org/bot{tg_token}/sendMessage", params={"chat_id":tg_chat,"text":msg})
            except Exception:
                pass
        st.session_state.notify_state[key]=main.confirms
    else:
        st.info("Нет идей по текущим фильтрам.")

    # Таблица
    rows=[]
    for s in scen:
        key=f"{s.name} ({s.bias})"; stx=("ok" if (s.confirms>=3 and s.status=="ok") else "2/3 ждём" if s.confirms==2 else f"{s.confirms}/3")
        rows.append({"Сценарий":key,"Тип":s.etype,"Подтв.(ядро)":f"{s.confirms}/3 ({stx})","Вход":fmt_price(s.entry,d),"Стоп":fmt_price(s.sl,d),"TP1":fmt_price(s.tp1,d),"TP2":fmt_price(s.tp2,d) if s.tp2 else "—","R:R":round(abs(s.tp1-s.entry)/max(abs(s.entry-s.sl),1e-9),2),"Prob%":round(probs.get(key,0.0),2)})
    if rows: st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)

    # Журналы
    st.markdown("### Журналы (SQLite)")
    jrows=db_fetch(200)
    if jrows:
        dfj=pd.DataFrame([{**r,**{"reasons":', '.join(json.loads(r["reasons"]) if isinstance(r["reasons"],str) else r["reasons"])}} for r in jrows])
        st.dataframe(dfj, use_container_width=True, hide_index=True)
        with st.form("mark_result"):
            tid=st.number_input("ID сделки",min_value=1,step=1)
            res=st.selectbox("Результат",["tp","sl","open"],index=2)
            submitted=st.form_submit_button("Отметить")
            if submitted:
                try: db_mark_result(int(tid),res); st.success("Сохранено"); st.experimental_rerun()
                except Exception as e: st.error(str(e))

    summary.append(f"{asset} {tf} → режим {reg}; идей {len(scen)}; ядро min {min_confirms}")
    st.divider()
except Exception as e:
    st.error(f"{asset}: {e}")

st.subheader("Зведення")
for line in summary: st.write("•", line)
