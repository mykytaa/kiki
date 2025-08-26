# smc_dashboard.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import time, sqlite3, json
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np, pandas as pd, streamlit as st, yfinance as yf

# ====== Config / Symbols / TF
ASSETS=["BTCUSDT","ETHUSDT","XAUUSD","XAUEUR","EURUSD"]
YF_TICKER_CANDIDATES={"BTCUSDT":["BTC-USD"],"ETHUSDT":["ETH-USD"],"XAUUSD":["XAUUSD=X","GC=F"],"XAUEUR":["XAUEUR=X"],"EURUSD":["EURUSD=X"]}
TF_FALLBACKS={"5m":[("5m","60d"),("15m","60d"),("60m","730d")],"15m":[("15m","60d"),("60m","730d")],"1h":[("60m","730d"),("1d","730d")]}
HTF_OF={"5m":"15m","15m":"60m","1h":"1d"}
DB="smc_journal.sqlite"

# ====== Indicators / Utils
ema=lambda x,n:x.ewm(span=n,adjust=False).mean()
def rsi(x,n=14): d=x.diff(); up=(d.clip(lower=0)).ewm(alpha=1/n,adjust=False).mean(); dn=(-d.clip(upper=0)).ewm(alpha=1/n,adjust=False).mean(); rs=up/(dn.replace(0,np.nan)); y=100-100/(1+rs); return y.fillna(method="bfill").fillna(50)
def macd(x): f=ema(x,12); s=ema(x,26); m=f-s; return m,ema(m,9),m-ema(m,9)
def atr(df,n=14): c=df.close; tr=pd.concat([(df.high-df.low),(df.high-c.shift()).abs(),(df.low-c.shift()).abs()],axis=1).max(axis=1); return tr.ewm(alpha=1/n,adjust=False).mean()
def obv(df): s=np.sign(df.close.diff().fillna(0.0)); return (s*df.volume).cumsum()
def adx(df,n=14): up=df.high.diff(); dn=-df.low.diff(); plus=np.where((up>dn)&(up>0),up,0.0); minus=np.where((dn>up)&(dn>0),dn,0.0); tr=pd.concat([(df.high-df.low),(df.high-df.close.shift()).abs(),(df.low-df.close.shift()).abs()],axis=1).max(axis=1); a=tr.ewm(alpha=1/n,adjust=False).mean(); p=100*pd.Series(plus,index=df.index).ewm(alpha=1/n,adjust=False).mean()/a; m=100*pd.Series(minus,index=df.index).ewm(alpha=1/n,adjust=False).mean()/a; d=100*(p.subtract(m).abs()/(p+m).replace(0,np.nan)); return d.ewm(alpha=1/n,adjust=False).mean().fillna(20)
def vwap_series(df): tp=(df.high+df.low+df.close)/3.0; vol=df.volume.replace(0,np.nan).fillna(0.0); num=(tp*vol).cumsum(); den=vol.cumsum().replace(0,np.nan); return (num/den).fillna(method="bfill").fillna(df.close)
def lin_slope(s,last_n=80):
    n = min(len(s), last_n)
    if n < 8:
        return 0.0
    y = s.tail(n).values
    x = np.arange(n, dtype=float)
    return float(np.polyfit(x, y, 1)[0])

def swings(df,L=3,R=3):
 hi,lo=df.high.values,df.low.values; n=len(df); SH=np.zeros(n,bool); SL=np.zeros(n,bool)
 for i in range(L,n-R):
  if hi[i]==hi[i-L:i+R+1].max(): SH[i]=True
  if lo[i]==lo[i-L:i+R+1].min(): SL[i]=True
 return pd.Series(SH,df.index),pd.Series(SL,df.index)

def bos(df,SH,SL,look=200,confirm_mult=0.30):
 rec=df.iloc[-look:]; sh_idx=rec[SH.loc[rec.index]].index; sl_idx=rec[SL.loc[rec.index]].index
 last_sh=rec.loc[sh_idx[-1]] if len(sh_idx) else None; last_sl=rec.loc[sl_idx[-1]] if len(sl_idx) else None
 a=float(atr(df).iloc[-1]) or 1e-6
 if last_sh is not None:
  lvl=last_sh.high; post=rec[rec.index>last_sh.name]; brk=post[post.close>lvl+confirm_mult*a]
  if len(brk): return "up",brk.index[0],lvl
 if last_sl is not None:
  lvl=last_sl.low; post=rec[rec.index>last_sl.name]; brk=post[post.close<lvl-confirm_mult*a]
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
 out={"bull":[],"bear":[]}; hi,lo,idx=df.high.values,df.low.values,df.index; n=len(df); s=max(2,n-look)
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
   last=reds.iloc[-1]; res["demand"]=(last.name,float(min(last.open,last.close)),float(max(last.open,last.close)))
 if dir_=="down":
  greens=before[before.close>before.open]
  if len(greens):
   last=greens.iloc[-1]; res["supply"]=(last.name,float(min(last.open,last.close)),float(max(last.open,last.close)))
 return res

def volume_profile(df,bins=40):
 lo=float(df.low.min()); hi=float(df.high.max()); 
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

def last_swing_levels(df,SH,SL):
 sh_idx=SH[SH].index; sl_idx=SL[SL].index
 sh=float(df.loc[sh_idx[-1],"high"]) if len(sh_idx) else None
 sl=float(df.loc[sl_idx[-1],"low"])  if len(sl_idx) else None
 return sh,sl

def score_bias(df):
 c=df.close; s=0; r=float(rsi(c).iloc[-1]); h=float(macd(c)[2].iloc[-1])
 if r>55:s+=1
 elif r<45:s-=1
 if h>0:s+=1
 elif h<0:s-=1
 return "long" if s>=1 else ("short" if s<=-1 else "none")

def regime_daily(df_d):
 e50=ema(df_d.close,50).iloc[-1]; e200=ema(df_d.close,200).iloc[-1]
 if np.isnan(e50) or np.isnan(e200): return "none"
 return "long" if e50>e200 else "short" if e50<e200 else "none"

def market_regime(df,vp):
 ad=float(adx(df).iloc[-1]); p=float(df.close.iloc[-1]); outside=(p>vp["vah"]) or (p<vp["val"])
 return "trend" if (ad>=22 or outside) else "range"

def validate_fvg(direction, gaps,*,price,vp,atr_val,bos_time,max_age=60,min_depth_atr=0.25):
 arr=gaps["bull"] if direction=="up" else gaps["bear"]
 if not arr: return None,["FVG отсутствует"]
 for t,lo,hi in reversed(arr):
  if bos_time and t<bos_time: continue
  mid=(lo+hi)/2; depth=abs(price-mid)/max(atr_val,1e-9)
  if depth<min_depth_atr: continue
  side_ok=(price>vp["poc"] and direction=="up") or (price<vp["poc"] and direction=="down") or abs(price-vp["poc"])<=0.6*atr_val
  if not side_ok: continue
  return (t,lo,hi),["FVG валиден"]
 return None,["FVG не прошёл валидацию"]

# ====== Scenario model
@dataclass
class Scenario:
 name:str; bias:str; etype:str; trigger:str; entry:float; sl:float; tp1:float; tp2:Optional[float]; rr:str
 confirms:int; confirm_list:List[str]; explain_short:str; stop_reason:str; tp_reason:str; status:str

def rr_targets(entry,sl,bias,min_rr=2.0):
 risk=abs(entry-sl) or 1e-9
 return (entry+min_rr*risk if bias=="long" else entry-min_rr*risk,
         entry+3.0*risk if bias=="long" else entry-3.0*risk,
         f"1:{int(min_rr)}/1:3")

def nearest_struct_target(df,SH,SL,entry,direction):
 if direction=="long":
  highs=[float(df.loc[i,"high"]) for i in SH[SH].index if float(df.loc[i,"high"])>entry]; return min(highs) if highs else None
 lows=[float(df.loc[i,"low"]) for i in SL[SL].index if float(df.loc[i,"low"])<entry]; return max(lows) if lows else None

def propose(df, htf_bias, d_bias, regime, vp, obv_slope_val):
 c=float(df.close.iloc[-1]); at=float(atr(df).iloc[-1]); at=max(at,1e-9)
 SH,SL=swings(df); dir_bos,t_bos,_=bos(df,SH,SL); gaps=fvg(df); swp=sweeps(df,SH,SL); ob=simple_ob(df,dir_bos,t_bos)
 sh_lvl,sl_lvl=last_swing_levels(df,SH,SL); vw=float(vwap_series(df).iloc[-1]); ema20_val=float(ema(df.close,20).iloc[-1])
 ema_up=lin_slope(ema(df.close,20))>0; ema_dn=lin_slope(ema(df.close,20))<0; above_poc=c>vp["poc"]

 def generic(bias,entry):
  out=[]
  if (bias=="long" and obv_slope_val>0) or (bias=="short" and obv_slope_val<0): out.append("OBV в сторону")
  if (bias=="long" and ema_up) or (bias=="short" and ema_dn): out.append("тренд EMA20")
  if (bias=="long" and above_poc) or (bias=="short" and not above_poc): out.append("сторона POC")
  if abs(entry-vw)<=0.6*at: out.append("рядом VWAP")
  if (bias==htf_bias) or (bias==d_bias): out.append("HTF/Daily")
  return out

 def struct_tp(entry,bias):
  tgt=nearest_struct_target(df,SH,SL,entry,"long" if bias=="long" else "short")
  return (tgt,"свинг") if tgt is not None else (vp["poc"],"POC")

 S:List[Scenario]=[]
 add=lambda **k: S.append(Scenario(**k))

 # SFP→BOS→FVG (вход от 0.5 FVG)
 if (swp["high"] and dir_bos=="down") or (swp["low"] and dir_bos=="up"):
  direction="down" if dir_bos=="down" else "up"
  f,_=validate_fvg("up" if direction=="up" else "down",gaps,price=c,vp=vp,atr_val=at,bos_time=t_bos)
  if f:
   _,lo,hi=f; mid=(lo+hi)/2; e=mid
   sl=(swp["low"][-1][1]-0.7*at) if direction=="up" else (swp["high"][-1][1]+0.7*at)
   bias="long" if direction=="up" else "short"; tp1,tp2,rr=rr_targets(e,sl,bias); tp1s,tp_reason=struct_tp(e,bias); tp1=tp1s
   base=["SFP","BOS","FVG валиден"]; conf=base+generic(bias,e)
   status="ok" if len(set([x.split()[0] for x in base]))>=3 else "await"
   add(name="SFP→BOS→FVG",bias=bias,etype="limit",trigger=f"mid FVG {mid:.5f}",entry=float(e),sl=float(sl),tp1=float(tp1),tp2=float(tp2),rr=rr,confirms=len(conf),confirm_list=conf,explain_short="срыв→пробой→имбаланс",stop_reason="за SFP±0.7×ATR",tp_reason=tp_reason,status=status)

 # BOS→OB Retest (+валидный FVG)
 if dir_bos=="up" and ob.get("demand"):
  _,lo,hi=ob["demand"]; e=hi; sl=lo-0.6*at; f,_=validate_fvg("up",gaps,price=c,vp=vp,atr_val=at,bos_time=t_bos)
  tp1,tp2,rr=rr_targets(e,sl,"long"); tp1s,tp_reason=struct_tp(e,"long"); tp1=tp1s
  base=["BOS↑","OB ретест"]+(["FVG валиден"] if f else []); conf=base+generic("long",e)
  status="ok" if len(set([x.split()[0] for x in base]))>=3 else "await"
  add(name="BOS→OB Retest",bias="long",etype="limit",trigger=f"OB {lo:.5f}-{hi:.5f}",entry=float(e),sl=float(sl),tp1=float(tp1),tp2=float(tp2),rr=rr,confirms=len(conf),confirm_list=conf,explain_short="пробой и ретест спроса",stop_reason="за OB −0.6×ATR",tp_reason=tp_reason,status=status)
 if dir_bos=="down" and ob.get("supply"):
  _,lo,hi=ob["supply"]; e=lo; sl=hi+0.6*at; f,_=validate_fvg("down",gaps,price=c,vp=vp,atr_val=at,bos_time=t_bos)
  tp1,tp2,rr=rr_targets(e,sl,"short"); tp1s,tp_reason=struct_tp(e,"short"); tp1=tp1s
  base=["BOS↓","OB ретест"]+(["FVG валиден"] if f else []); conf=base+generic("short",e)
  status="ok" if len(set([x.split()[0] for x in base]))>=3 else "await"
  add(name="BOS→OB Retest",bias="short",etype="limit",trigger=f"OB {lo:.5f}-{hi:.5f}",entry=float(e),sl=float(sl),tp1=float(tp1),tp2=float(tp2),rr=rr,confirms=len(conf),confirm_list=conf,explain_short="пробой и ретест предложения",stop_reason="за OB +0.6×ATR",tp_reason=tp_reason,status=status)

 # Breaker
 if swp["high"] and dir_bos=="down":
  _,lvl=swp["high"][-1]; e=lvl-0.1*at; sl=lvl+0.7*at; tp1,tp2,rr=rr_targets(e,sl,"short")
  base=["SFP","BOS↓"]; conf=base+generic("short",e); status="ok" if len(set([x.split()[0] for x in base]))>=3 else "await"
  add(name="Breaker",bias="short",etype="stop",trigger=f"возврат под {lvl:.5f}",entry=float(e),sl=float(sl),tp1=float(tp1),tp2=float(tp2),rr=rr,confirms=len(conf),confirm_list=conf,explain_short="срыв high и возврат",stop_reason="над свип-уровнем +0.7×ATR",tp_reason="свинг/POC",status=status)
 if swp["low"] and dir_bos=="up":
  _,lvl=swp["low"][-1]; e=lvl+0.1*at; sl=lvl-0.7*at; tp1,tp2,rr=rr_targets(e,sl,"long")
  base=["SFP","BOS↑"]; conf=base+generic("long",e); status="ok" if len(set([x.split()[0] for x in base]))>=3 else "await"
  add(name="Breaker",bias="long",etype="stop",trigger=f"возврат над {lvl:.5f}",entry=float(e),sl=float(sl),tp1=float(tp1),tp2=float(tp2),rr=rr,confirms=len(conf),confirm_list=conf,explain_short="срыв low и возврат",stop_reason="под свип-уровнем −0.7×ATR",tp_reason="свинг/POC",status=status)

 # Range reversion
 adx_val=float(adx(df).iloc[-1])
 if regime=="range" and adx_val<22:
  if abs(c-vp["val"])<=max(0.6*at,0.1*(vp["vah"]-vp["val"])):
   e=vp["val"]+0.1*at; sl=vp["val"]-0.8*at; tp1,tp2,rr=rr_targets(e,sl,"long")
   base=["VAL edge","ADX низкий"]; conf=base+generic("long",e); status="ok" if len(set([x.split()[0] for x in base]))>=3 else "await"
   add(name="Value Area Reversion",bias="long",etype="limit",trigger="от VAL к POC",entry=float(e),sl=float(sl),tp1=float(tp1),tp2=float(tp2),rr=rr,confirms=len(conf),confirm_list=conf,explain_short="от края к POC",stop_reason="под VAL −0.8×ATR",tp_reason="POC",status=status)
  if abs(c-vp["vah"])<=max(0.6*at,0.1*(vp["vah"]-vp["val"])):
   e=vp["vah"]-0.1*at; sl=vp["vah"]+0.8*at; tp1,tp2,rr=rr_targets(e,sl,"short")
   base=["VAH edge","ADX низкий"]; conf=base+generic("short",e); status="ok" if len(set([x.split()[0] for x in base]))>=3 else "await"
   add(name="Value Area Reversion",bias="short",etype="limit",trigger="от VAH к POC",entry=float(e),sl=float(sl),tp1=float(tp1),tp2=float(tp2),rr=rr,confirms=len(conf),confirm_list=conf,explain_short="от края к POC",stop_reason="над VAH +0.8×ATR",tp_reason="POC",status=status)

 # EMA pullback
 if regime=="trend" and c>ema20_val:
  e=ema20_val; sl=e-1.2*at; tp1,tp2,rr=rr_targets(e,sl,"long")
  base=["тренд вверх","EMA20"]; conf=base+generic("long",e); status="ok" if len(set([x.split()[0] for x in base]))>=3 else "await"
  add(name="EMA Pullback",bias="long",etype="limit",trigger=f"к EMA20 {e:.5f}",entry=float(e),sl=float(sl),tp1=float(tp1),tp2=float(tp2),rr=rr,confirms=len(conf),confirm_list=conf,explain_short="pullback к EMA/VWAP",stop_reason="ниже EMA −1.2×ATR",tp_reason="свинг/POC",status=status)
 if regime=="trend" and c<ema20_val:
  e=ema20_val; sl=e+1.2*at; tp1,tp2,rr=rr_targets(e,sl,"short")
  base=["тренд вниз","EMA20"]; conf=base+generic("short",e); status="ok" if len(set([x.split()[0] for x in base]))>=3 else "await"
  add(name="EMA Pullback",bias="short",etype="limit",trigger=f"к EMA20 {e:.5f}",entry=float(e),sl=float(sl),tp1=float(tp1),tp2=float(tp2),rr=rr,confirms=len(conf),confirm_list=conf,explain_short="pullback к EMA/VWAP",stop_reason="выше EMA +1.2×ATR",tp_reason="свинг/POC",status=status)

 # Structure breakout
 if sh_lvl is not None:
  base_sl=(sl_lvl if sl_lvl is not None else c-1.2*at); e=sh_lvl+0.2*at; sl=base_sl-0.4*at; tp1,tp2,rr=rr_targets(e,sl,"long")
  base=["swing↑","ретест"]; conf=base+generic("long",e); status="ok" if len(set([x.split()[0] for x in base]))>=3 else "await"
  add(name="Structure Breakout",bias="long",etype="stop",trigger=f"пробой {sh_lvl:.5f}",entry=float(e),sl=float(sl),tp1=float(tp1),tp2=float(tp2),rr=rr,confirms=len(conf),confirm_list=conf,explain_short="breakout",stop_reason="за swing −0.4×ATR",tp_reason="свинг/POC",status=status)
 if sl_lvl is not None:
  base_sl=(sh_lvl if sh_lvl is not None else c+1.2*at); e=sl_lvl-0.2*at; sl=base_sl+0.4*at; tp1,tp2,rr=rr_targets(e,sl,"short")
  base=["swing↓","ретест"]; conf=base+generic("short",e); status="ok" if len(set([x.split()[0] for x in base]))>=3 else "await"
  add(name="Structure Breakout",bias="short",etype="stop",trigger=f"пробой {sl_lvl:.5f}",entry=float(e),sl=float(sl),tp1=float(tp1),tp2=float(tp2),rr=rr,confirms=len(conf),confirm_list=conf,explain_short="breakdown",stop_reason="за swing +0.4×ATR",tp_reason="свинг/POC",status=status)

 U,seen=[],set()
 for s in sorted(S, key=lambda x: (-x.confirms, -1 if (x.status=="ok") else 0)):
  k=(s.name,s.bias)
  if k in seen: continue
  seen.add(k); U.append(s)
  if len(U)>=8: break
 if not U:
  c0=c; U.append(Scenario("Wait (no-trade)","none","—","нет",c0,c0,c0,c0,"—",0,[],"Пауза","—","—","await"))
 return U

def scenario_probabilities(scen,htf_bias,d_bias,obv_slope_val,price,vp,atr_val,regime,cap=0.9,floor=0.05,temp=1.12):
 if not scen: return {"Wait (no-trade)":100.0},{"long":0.0,"short":0.0}
 scores,labels=[],[]
 for s in scen:
  if s.name.startswith("Wait"): continue
  sc=0.6*s.confirms+(1.6 if s.bias==htf_bias else 0)+(1.0 if s.bias==d_bias else 0)
  sc+=0.8 if ((obv_slope_val>0 and s.bias=="long") or (obv_slope_val<0 and s.bias=="short")) else -0.2
  if (s.name.startswith(("SFP→BOS→FVG","BOS→OB","EMA","Structure")) and regime=="trend") or (s.name.startswith(("Value Area","Breaker")) and regime=="range"): sc+=1.0
  dist=abs(s.entry-price)/max(atr_val,1e-6); sc+=(-1.0 if dist>2.0 else (-0.5 if dist>1.5 else 0))
  scores.append(sc); labels.append((s.name,s.bias))
 scores=np.array(scores)/temp; ex=np.exp(scores-scores.max()); p=np.clip(ex/ex.sum(),floor,cap); p=p/p.sum()
 out,agg={},{"long":0.0,"short":0.0}
 for (lbl,bias),pp in zip(labels,p): val=float(np.round(pp*100.0,2)); out[f"{lbl} ({bias})"]=val; agg[bias]+=val
 return dict(sorted(out.items(),key=lambda x:x[1],reverse=True)),{k:round(v,2) for k,v in agg.items()}

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
    except Exception: df.index=pd.to_datetime(df.index,utc=True)
    out=df[need].dropna().tail(limit)
    if out.empty: last_err=f"{tkr}@{interval}/{period}: после очистки нет данных"; continue
    return out,interval,period
   except Exception as e: last_err=f"{tkr}@{interval}/{period}: {e}"; continue
 raise RuntimeError(f"yfinance: нет данных для {asset_key}. {last_err}")

# ====== SQLite Journal
def db_init():
 con=sqlite3.connect(DB); cur=con.cursor()
 cur.execute("""CREATE TABLE IF NOT EXISTS trades (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts TEXT, asset TEXT, tf TEXT,
  name TEXT, bias TEXT,
  entry REAL, sl REAL, tp1 REAL, tp2 REAL,
  confirms INTEGER, reasons TEXT,
  trade_type TEXT, status TEXT, result TEXT,
  notes TEXT, result_ts TEXT
 )"""); con.commit(); con.close()

def db_insert(row:dict)->int:
 con=sqlite3.connect(DB); cur=con.cursor()
 cur.execute("""INSERT INTO trades
 (ts,asset,tf,name,bias,entry,sl,tp1,tp2,confirms,reasons,trade_type,status,result,notes,result_ts)
 VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
 (row["ts"],row["asset"],row["tf"],row["name"],row["bias"],row["entry"],row["sl"],row["tp1"],row.get("tp2"),
  row["confirms"],json.dumps(row["reasons"],ensure_ascii=False),row.get("trade_type","plan"),
  row.get("status","open"),row.get("result",""),row.get("notes",""),row.get("result_ts","")))
 con.commit(); tid=cur.lastrowid; con.close(); return tid

def db_update(trade_id:int, updates:dict):
 con=sqlite3.connect(DB); cur=con.cursor()
 sets=[]; vals=[]
 for k,v in updates.items():
  sets.append(f"{k}=?")
  vals.append(v if k!="reasons" else json.dumps(v,ensure_ascii=False))
 vals.append(trade_id)
 cur.execute(f"UPDATE trades SET {', '.join(sets)} WHERE id=?", vals)
 con.commit(); con.close()

def db_fetch(limit=500):
 con=sqlite3.connect(DB); cur=con.cursor()
 cur.execute("SELECT id,ts,asset,tf,name,bias,entry,sl,tp1,tp2,confirms,reasons,trade_type,status,result,notes,result_ts FROM trades ORDER BY id DESC LIMIT ?",(limit,))
 rows=cur.fetchall(); con.close()
 cols=["id","ts","asset","tf","name","bias","entry","sl","tp1","tp2","confirms","reasons","trade_type","status","result","notes","result_ts"]
 out=[]; 
 for r in rows:
  d=dict(zip(cols,r))
  try: d["reasons"]=json.loads(d["reasons"]) if isinstance(d["reasons"],(str,bytes)) else d["reasons"]
  except Exception: pass
  out.append(d)
 return out

# ====== Outcome checker (first-touch TP1 vs SL after ts)
def _yf_interval(tf): return {"5m":"5m","15m":"15m","1h":"60m"}[tf]
def check_outcome_history(asset, tf, ts_iso, bias, entry, sl, tp1) -> Tuple[str,str]:
 try:
  df,_,_=yf_ohlc_first_success(asset, tf, limit=1200)
  df=df[df.index>=pd.to_datetime(ts_iso,utc=True)]
  if df.empty: return "",""
  for t,row in df.iterrows():
   if bias=="long":
    if row.low<=sl: return "loss", str(t)
    if row.high>=tp1: return "win", str(t)
   else:
    if row.high>=sl: return "loss", str(t)
    if row.low<=tp1: return "win", str(t)
  return "", ""
 except Exception:
  return "",""

# ====== Format helpers
def infer_decimals(df,asset):
 x=df.close.tail(300).diff().abs(); step=float(np.nanmin(x[x>0])) if np.any(x>0) else 0.0
 if step>0: p=max(2,min(6,int(np.ceil(-np.log10(step))+1)))
 else: p={"EURUSD":5,"XAUEUR":5,"XAUUSD":2,"BTCUSDT":2,"ETHUSDT":2}.get(asset,4)
 return p
def fmt_price(x,d):
 s=f"{x:.{d}f}"; a,b=s.split(".") if "." in s else (s,"")
 a=f"{int(float(a)):,}".replace(","," "); return f"{a}.{b}" if b else a

# ====== App
st.set_page_config(page_title="SMC Intraday (text)", layout="wide")
st.title("SMC Intraday — BTC / ETH / XAUUSD / XAUEUR / EURUSD (text)")
db_init()
tab_signals, tab_journal = st.tabs(["📊 Сигналы","📒 Журнал"])

with tab_signals:
 colA,colB,colC,colD,colE=st.columns([1.2,1,1,1,1])
 with colA: asset=st.selectbox("Актив",ASSETS,index=4 if "EURUSD" in ASSETS else 0, key="asset_sel")
 with colB: tf=st.selectbox("TF",["5m","15m","1h"],index=0, key="tf_sel")
 with colC: min_risk_pct=st.slider("Мин. риск (%ATR)",5,60,25,step=5)
 with colD: min_tp1_atr=st.slider("Мин. TP1 (×ATR)",1.0,3.0,1.5,step=0.25)
 with colE: min_confirms=st.slider("Мин. подтверждений",2,7,3,step=1)
 colF,colG=st.columns([1,1])
 with colF: refresh_mode=st.selectbox("Автообновление",["Выключено","30s","1m","2m","5m"],index=0)
 with colG: beginner_mode=st.checkbox("Простой режим",value=True)
 st.caption("«📝 В журнал» = ордер поставлен/вход выполнен (тип из сценария). 2/3 подтверждений показывается как «ожидаем 3».")

 INTERVALS={"30s":30,"1m":60,"2m":120,"5m":300}
 if "next_refresh_ts" not in st.session_state: st.session_state.next_refresh_ts=time.time()+10**9
 if st.button("🔄 Обновить сейчас"): st.cache_data.clear(); st.experimental_rerun()
 if refresh_mode!="Выключено":
  it=INTERVALS[refresh_mode]; now=time.time()
  if now>=st.session_state.next_refresh_ts:
   st.session_state.next_refresh_ts=now+it; st.cache_data.clear(); st.experimental_rerun()
 else: st.session_state.next_refresh_ts=time.time()+10**9

 try:
  df,tf_eff,_=yf_ohlc_first_success(asset,tf,limit=800)
  htf=HTF_OF[tf]; df_h,_,_=yf_ohlc_first_success(asset,htf,limit=400)
  df_d,_,_=yf_ohlc_first_success(asset,"1d",limit=600)
  price=float(df.close.iloc[-1]); vp=volume_profile(df); reg=market_regime(df,vp); atr_v=float(atr(df).iloc[-1]); obv_s=lin_slope(obv(df),160)
  htf_bias=score_bias(df_h); d_bias=regime_daily(df_d)
  scen_all=propose(df,htf_bias,d_bias,reg,vp,obv_s)
  min_risk=atr_v*(min_risk_pct/100.0)
  scen,awaiting=[],[]
  for s in scen_all:
   if s.name.startswith("Wait"): continue
   risk_ok=abs(s.entry-s.sl)>=min_risk; tp1_ok=(abs(s.tp1-s.entry)/max(atr_v,1e-6))>=min_tp1_atr
   if risk_ok and tp1_ok:
    if s.confirms>=min_confirms and s.status=="ok": scen.append(s)
    elif s.confirms==max(2,min_confirms-1) or s.status=="await": awaiting.append(s)
  if not scen: scen=awaiting or scen_all

  probs,balance=scenario_probabilities(scen,htf_bias,d_bias,obv_s,price,vp,atr_v,reg)
  d=infer_decimals(df,asset)
  st.markdown(f"### {asset} ({tf}) — цена: {fmt_price(price,d)}")
  ltf_b=score_bias(df); poc_state="выше VAH" if price>vp["vah"] else ("ниже VAL" if price<vp["val"] else "внутри value area")
  st.markdown(f"**Контекст:** LTF={ltf_b.upper()}, HTF={htf_bias.upper()}, Daily={d_bias.upper()} • Режим: {reg.upper()} (ADX≈{float(adx(df).iloc[-1]):.1f}) • POC {fmt_price(vp['poc'],d)}, VAL {fmt_price(vp['val'],d)}, VAH {fmt_price(vp['vah'],d)} → цена {poc_state}.")

  for i,s in enumerate(scen):
   if s.name.startswith("Wait"): continue
   key=f"{s.name} ({s.bias})"; rr=round(abs(s.tp1-s.entry)/max(abs(s.entry-s.sl),1e-9),2)
   status_txt="готов (3/3+)" if (s.confirms>=3 and s.status=="ok") else f"{min(s.confirms,3)}/3 — ждём 3"
   c1,c2,c3=st.columns([5,2,2])
   with c1:
    st.markdown(f"**{key}** · {status_txt} · Prob≈{probs.get(key,0.0):.1f}%  \n"
                f"Вход {fmt_price(s.entry,d)} · Стоп {fmt_price(s.sl,d)} · TP1 {fmt_price(s.tp1,d)} · R:R≈{rr}  \n"
                f"Подтв.: {', '.join(s.confirm_list) if s.confirm_list else '—'}  \n"
                f"Почему: {s.explain_short}")
   with c2:
    btn=st.button("📝 В журнал", key=f"add_{i}")
    if btn:
     row={"ts":str(df.index[-1]),"asset":asset,"tf":tf,"name":s.name,"bias":s.bias,"entry":float(s.entry),
          "sl":float(s.sl),"tp1":float(s.tp1),"tp2":float(s.tp2) if s.tp2 else None,"confirms":int(s.confirms),
          "reasons":s.confirm_list,"trade_type":("limit" if s.etype=="limit" else "stop"),"status":"open","result":"","notes":""}
     new_id=db_insert(row); st.success(f"Добавлено в журнал (ID {new_id}) — считаем, что ордер поставлен/вход выполнен.")
   with c3: st.metric("Подтв.", s.confirms)
   st.divider()

 except Exception as e:
  st.error(f"{asset}: {e}")

with tab_journal:
 st.subheader("Журнал сделок")
 data=db_fetch(500)
 if not data:
  st.info("Журнал пуст. Добавь записи из вкладки «Сигналы».")
 else:
  dfj=pd.DataFrame([{**r, "reasons":(", ".join(r["reasons"]) if isinstance(r.get("reasons"),list) else r.get("reasons"))} for r in data])
  st.dataframe(dfj, use_container_width=True, hide_index=True)
  st.markdown("#### Управление записью")
  ids=[r["id"] for r in data]; sel_id=st.selectbox("ID сделки", ids)
  sel=next(r for r in data if r["id"]==sel_id)

  col1,col2,col3,col4=st.columns(4)
  with col1: trade_type=st.selectbox("Тип", ["plan","market","limit","stop"], index=["plan","market","limit","stop"].index(sel.get("trade_type","plan")))
  with col2: status=st.selectbox("Статус", ["open","closed","cancelled"], index=["open","closed","cancelled"].index(sel.get("status","open")))
  with col3: result=st.selectbox("Результат", ["","win","loss","be"], index=["","win","loss","be"].index(sel.get("result","")))
  with col4: result_ts=st.text_input("Дата результата (опц.)", value=sel.get("result_ts",""))
  notes=st.text_area("Заметки", value=sel.get("notes",""), height=100)

  e1,e2,e3=st.columns(3)
  with e1: entry=st.number_input("Entry", value=float(sel.get("entry") or 0.0), format="%.8f")
  with e2: sl=st.number_input("SL", value=float(sel.get("sl") or 0.0), format="%.8f")
  with e3: tp1=st.number_input("TP1", value=float(sel.get("tp1") or 0.0), format="%.8f")
  tp2=st.number_input("TP2 (опц.)", value=float(sel.get("tp2") or 0.0), format="%.8f")

  b1,b2,b3,b4,b5=st.columns(5)
  if b1.button("💾 Сохранить"):
   db_update(sel_id, {"trade_type":trade_type,"status":status,"result":result,"notes":notes,"result_ts":result_ts,"entry":entry,"sl":sl,"tp1":tp1,"tp2":tp2})
   st.success("Сохранено.")

  if b2.button("✔️ TP"):
   db_update(sel_id, {"status":"closed","result":"win","result_ts":time.strftime("%Y-%m-%d %H:%M:%S")}); st.success("Отмечено TP (закрыто).")
  if b3.button("❌ SL"):
   db_update(sel_id, {"status":"closed","result":"loss","result_ts":time.strftime("%Y-%m-%d %H:%M:%S")}); st.warning("Отмечено SL (закрыто).")
  if b4.button("🔄 BE"):
   db_update(sel_id, {"status":"closed","result":"be","result_ts":time.strftime("%Y-%m-%d %H:%M:%S")}); st.info("Отмечено BE (закрыто).")
  if b5.button("◼️ Cancel"):
   db_update(sel_id, {"status":"cancelled","result":"","result_ts":time.strftime("%Y-%m-%d %H:%M:%S")}); st.info("Сделка отменена.")

  if st.button("🔍 Автопроверка TP/SL (история)"):
   res,ts_hit=check_outcome_history(sel["asset"], sel["tf"], sel["ts"], sel["bias"], float(entry), float(sl), float(tp1))
   if res:
    db_update(sel_id, {"status":"closed","result":res,"result_ts":ts_hit})
    st.success(f"Автопроверка: {res.upper()} @ {ts_hit}")
   else:
    st.info("По истории после времени сигнала TP/SL не были достигнуты (или данных мало).")
