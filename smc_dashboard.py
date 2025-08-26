# smc_dashboard.py
# -*- coding: utf-8 -*-

"""
SMC Intraday — BTC / ETH / XAUUSD / XAUEUR / EURUSD (text)
Сигналы с конфлюенсами SFP/BOS/iFVG/BPR, строгой валидацией, журналом сделок (SQLite),
уведомлениями (Telegram, анти-спам), карточками «2 из 3 (ожидаем 3)», вкладки: Сигналы / Журнал / Статистика / Правила.
Исключено по желанию: расширенная TP-логика, ротация источников, авто-децимали, экспорт/индексы/Walk-Forward/оптимизации/секьюрити.
"""
from __future__ import annotations
import time, sqlite3, hashlib, json, math, os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np, pandas as pd, streamlit as st, yfinance as yf, requests

# ========= Config (минимум)
ASSETS = ["EURUSD","BTCUSDT","ETHUSDT","XAUUSD","XAUEUR"]
YF_TICKER_CANDIDATES = {"BTCUSDT":["BTC-USD"],"ETHUSDT":["ETH-USD"],"XAUUSD":["XAUUSD=X","GC=F"],"XAUEUR":["XAUEUR=X"],"EURUSD":["EURUSD=X"]}
TF_FALLBACKS = {"5m":[("5m","60d"),("15m","60d"),("60m","730d")],"15m":[("15m","60d"),("60m","730d")],"1h":[("60m","730d"),("1d","730d")]}
HTF_OF = {"5m":"15m","15m":"60m","1h":"1d"}
DB = "smc_journal.sqlite"

# ========= Индикаторы/утилы
ema=lambda x,n:x.ewm(span=n,adjust=False).mean()
def rsi(x,n=14): d=x.diff(); up=(d.clip(lower=0)).ewm(alpha=1/n,adjust=False).mean(); dn=(-d.clip(upper=0)).ewm(alpha=1/n,adjust=False).mean(); rs=up/(dn.replace(0,np.nan)); y=100-100/(1+rs); return y.fillna(method="bfill").fillna(50)
def macd(x): f=ema(x,12); s=ema(x,26); m=f-s; return m,ema(m,9),m-ema(m,9)
def atr(df,n=14): c=df.close; tr=pd.concat([(df.high-df.low),(df.high-c.shift()).abs(),(df.low-c.shift()).abs()],axis=1).max(axis=1); return tr.ewm(alpha=1/n,adjust=False).mean()
def obv(df): s=np.sign(df.close.diff().fillna(0.0)); return (s*df.volume).cumsum()
def adx(df,n=14): up=df.high.diff(); dn=-df.low.diff(); plus=np.where((up>dn)&(up>0),up,0.0); minus=np.where((dn>up)&(dn>0),dn,0.0); tr=pd.concat([(df.high-df.low),(df.high-df.close.shift()).abs(),(df.low-df.close.shift()).abs()],axis=1).max(axis=1); a=tr.ewm(alpha=1/n,adjust=False).mean(); p=100*pd.Series(plus,index=df.index).ewm(alpha=1/n,adjust=False).mean()/a; m=100*pd.Series(minus,index=df.index).ewm(alpha=1/n,adjust=False).mean()/a; d=100*(p.subtract(m).abs()/(p+m).replace(0,np.nan)); return d.ewm(alpha=1/n,adjust=False).mean().fillna(20)
def slope_series(s,last_n=80):
    n = min(len(s), last_n)
    if n < 8:
        return 0.0
    y = s.tail(n).values
    x = np.arange(n, dtype=float)
    return float(np.polyfit(x, y, 1)[0])
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
def vwap_series(df): tp=(df.high+df.low+df.close)/3.0; vol=df.volume.replace(0,np.nan).fillna(0.0); num=(tp*vol).cumsum(); den=vol.cumsum().replace(0,np.nan); return (num/den).fillna(method="bfill").fillna(df.close)

# ========= SMC примитивы
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

# ========= Расширенные подтверждения: iFVG, BPR, дивергенции, строгая валидация
def is_impulse_bar(df,i,atr_val,impulse_mult=1.2):
 rng=float(df.high.iloc[i]-df.low.iloc[i]); return rng>=impulse_mult*float(atr_val)
def detect_iFVG(df,gaps,direction,atr_val,min_depth_atr=0.3,impulse_mult=1.2):
 arr=gaps["bull"] if direction=="up" else gaps["bear"]
 if not arr: return None
 t,lo,hi=list(arr)[-1]  # самый свежий
 j=df.index.get_indexer([t])[0]
 if j-1<0: return None
 if not is_impulse_bar(df,j,atr_val,impulse_mult): return None
 mid=(lo+hi)/2; depth=abs(float(df.close.iloc[-1])-mid)/max(float(atr_val),1e-9)
 if depth<min_depth_atr: return None
 return (t,lo,hi)
def detect_bpr(gaps):
 # BPR: рядом лежащие FVG разного типа перекрываются -> баланс/ретест
 bulls=gaps["bull"]; bears=gaps["bear"]
 if not bulls or not bears: return None
 tb,lb,hb=bulls[-1]; ts,hs,ls=bears[-1]
 inter_lo=max(lb,hs); inter_hi=min(hb,ls)
 if inter_hi>inter_lo: return (max(tb,ts),inter_lo,inter_hi)  # зона BPR
 return None
def strict_fvg_validate(direction,gaps,*,price,vp,atr_val,bos_time,max_age_bars=60,min_depth_atr=0.3,need_side=True):
 notes=[]; use=None
 arr=list(reversed(gaps["bull"] if direction=="up" else gaps["bear"]))
 for t,lo,hi in arr:
  if bos_time and t<bos_time: continue
  mid=(lo+hi)/2
  depth=abs(price-mid)/max(atr_val,1e-9)
  if depth<min_depth_atr: notes.append("глубина<мин"); continue
  side_ok=(price>vp["poc"] and direction=="up") or (price<vp["poc"] and direction=="down") or (not need_side)
  if not side_ok: notes.append("POC сторона не совпала"); continue
  use=(t,lo,hi); break
 if use: notes.append("FVG валиден")
 return use,notes
def obv_rsi_divergence(df,look=60):
 px=df.close.tail(look); o=obv(df).tail(look); r=rsi(df.close).tail(look)
 def _sl(s): 
  x=np.arange(len(s)); return float(np.polyfit(x,s.values,1)[0]) if len(s)>=10 else 0.0
 return {"price_slope":_sl(px),"obv_slope":_sl(o),"rsi_slope":_sl(r)}

# ========= Контекст/скоринг
def last_swings(df,SH,SL):
 sh_idx=SH[SH].index; sl_idx=SL[SL].index
 sh=float(df.loc[sh_idx[-1],"high"]) if len(sh_idx) else None
 sl=float(df.loc[sl_idx[-1],"low"]) if len(sl_idx) else None
 return sh,sl
def score_bias(df):
 c=df.close; s=0; r=float(rsi(c).iloc[-1]); h=float(macd(c)[2].iloc[-1])
 if r>55:s+=1
 elif r<45:s-=1
 if h>0:s+=1
 elif h<0:s-=1
 return "long" if s>=1 else ("short" if s<=-1 else "none")
def regime_daily(df_d):
    e50 = ema(df_d.close, 50).iloc[-1]
    e200 = ema(df_d.close, 200).iloc[-1]
    if np.isnan(e50) or np.isnan(e200):
        return "none"
    if e50 > e200:
        return "long"
    elif e50 < e200:
        return "short"
    else:
        return "none"
def market_regime(df,vp): ad=float(adx(df).iloc[-1]); p=float(df.close.iloc[-1]); outside=(p>vp["vah"]) or (p<vp["val"]); return "trend" if (ad>=22 or outside) else "range"

# ========= Правила подтверждений (в SQLite) + загрузка/сохранение
DEFAULT_RULES = {
 "windows":{"SFP":120,"BOS":120,"FVG":80,"iFVG":80,"BPR":80},
 "min_depth_atr":0.3,"impulse_mult":1.2,"need_poc_side":True,
 "weights":{"base":1.0,"obv_align":0.5,"ema_trend":0.5,"poc_side":0.4,"vwap_near":0.3,"divergence":0.4},
 "min_confirms":3
}
def db_init():
 con=sqlite3.connect(DB); cur=con.cursor()
 cur.execute("""CREATE TABLE IF NOT EXISTS trades(
  id INTEGER PRIMARY KEY, ts TEXT, asset TEXT, tf TEXT, name TEXT, bias TEXT,
  entry REAL, sl REAL, tp1 REAL, tp2 REAL, confirms INTEGER, reasons TEXT, status TEXT,
  journal_state TEXT, result TEXT, result_ts TEXT)""")
 cur.execute("""CREATE TABLE IF NOT EXISTS cfg (k TEXT PRIMARY KEY, v TEXT)""")
 cur.execute("""CREATE TABLE IF NOT EXISTS alerts (sig TEXT PRIMARY KEY, ts REAL)""")
 # сохранить дефолтные правила если пусто
 cur.execute("SELECT COUNT(*) FROM cfg"); n=cur.fetchone()[0]
 if n==0:
  cur.execute("INSERT INTO cfg(k,v) VALUES(?,?)",("rules",json.dumps(DEFAULT_RULES,ensure_ascii=False)))
 con.commit(); con.close()
def load_rules()->dict:
 con=sqlite3.connect(DB); cur=con.cursor(); cur.execute("SELECT v FROM cfg WHERE k='rules'"); row=cur.fetchone(); con.close()
 return json.loads(row[0]) if row else DEFAULT_RULES
def save_rules(r:dict):
 con=sqlite3.connect(DB); cur=con.cursor(); cur.execute("INSERT OR REPLACE INTO cfg(k,v) VALUES(?,?)",("rules",json.dumps(r,ensure_ascii=False))); con.commit(); con.close()

# ========= Данные
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

# ========= Журнал
def db_upsert_trade(row:dict):
 con=sqlite3.connect(DB); cur=con.cursor()
 cur.execute("""INSERT INTO trades(ts,asset,tf,name,bias,entry,sl,tp1,tp2,confirms,reasons,status,journal_state,result,result_ts)
 VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
  (row["ts"],row["asset"],row["tf"],row["name"],row["bias"],row["entry"],row["sl"],row.get("tp1"),row.get("tp2"),
   row["confirms"],json.dumps(row["reasons"],ensure_ascii=False),row["status"],row["journal_state"],row.get("result","open"),row.get("result_ts","")))
 con.commit(); con.close()
def db_list_trades(limit=300):
 con=sqlite3.connect(DB); cur=con.cursor()
 cur.execute("""SELECT id,ts,asset,tf,name,bias,entry,sl,tp1,tp2,confirms,reasons,status,journal_state,result,result_ts
               FROM trades ORDER BY id DESC LIMIT ?""",(limit,))
 rows=cur.fetchall(); con.close()
 cols=["id","ts","asset","tf","name","bias","entry","sl","tp1","tp2","confirms","reasons","status","journal_state","result","result_ts"]
 return [dict(zip(cols,r)) for r in rows]
def db_update_trade_status(trade_id:int,*,result:str=None,journal_state:str=None):
 con=sqlite3.connect(DB); cur=con.cursor()
 if result:
  cur.execute("UPDATE trades SET result=?, result_ts=? WHERE id=?",(result,str(pd.Timestamp.utcnow()),trade_id))
 if journal_state:
  cur.execute("UPDATE trades SET journal_state=? WHERE id=?",(journal_state,trade_id))
 con.commit(); con.close()

# ========= Анти-спам уведомления
def already_alerted(sig:str, cooldown_sec:int=600)->bool:
 now=time.time(); con=sqlite3.connect(DB); cur=con.cursor()
 cur.execute("SELECT ts FROM alerts WHERE sig=?",(sig,)); row=cur.fetchone()
 if row and now-row[0]<cooldown_sec: con.close(); return True
 cur.execute("INSERT OR REPLACE INTO alerts(sig,ts) VALUES(?,?)",(sig,now)); con.commit(); con.close(); return False

# ========= Сценарии
@dataclass
class Scenario:
 name:str; bias:str; etype:str; trigger:str; entry:float; sl:float; tp1:float; tp2:Optional[float]
 rr:str; confirms:int; confirm_list:List[str]; explain_short:str; stop_reason:str; tp_reason:str
 logic_path:List[str]; status:str; missing:List[str]

def rr_targets(entry,sl,bias,min_rr=2.0):
 risk=abs(entry-sl) or 1e-9
 return (entry+min_rr*risk if bias=="long" else entry-min_rr*risk,
         entry+3.0*risk if bias=="long" else entry-3.0*risk,
         f"1:{int(min_rr)}/1:3")

def propose(df, rules, htf_bias, d_bias, regime, vp, asset, tf) -> List[Scenario]:
 price=float(df.close.iloc[-1]); at=float(atr(df).iloc[-1]); at=max(at,1e-9)
 SH,SL=swings(df); dir_bos,t_bos,lvl_bos=bos(df,SH,SL); gaps=fvg(df); swp=sweeps(df,SH,SL)
 sh_lvl,sl_lvl=last_swings(df,SH,SL); ema20=float(ema(df.close,20).iloc[-1]); vw=float(vwap_series(df).iloc[-1])
 obv_rsi=obv_rsi_divergence(df,look=max(40,rules["windows"]["FVG"]))
 conf_weights=rules["weights"]; need_side=rules["need_poc_side"]

 def gen_conf(bias,entry):
  out=[]
  if (bias=="long" and obv_rsi["obv_slope"]>0) or (bias=="short" and obv_rsi["obv_slope"]<0): out.append("OBV в сторону")
  ema_tr = slope_series(ema(df.close,20),80)
  if (bias=="long" and ema_tr>0) or (bias=="short" and ema_tr<0): out.append("тренд EMA20")
  poc = (price>vp["poc"] and bias=="long") or (price<vp["poc"] and bias=="short")
  if poc: out.append("сторона POC")
  if abs(entry-vw)<=0.6*at: out.append("рядом VWAP")
  # дивергенция как ДОП-плюс (если price_slope и rsi_slope разнонаправлены)
  if (obv_rsi["price_slope"]>0 and obv_rsi["rsi_slope"]<0) or (obv_rsi["price_slope"]<0 and obv_rsi["rsi_slope"]>0):
   out.append("RSI дивергенция")
  return out

 scenarios:List[Scenario]=[]
 add=lambda **k: scenarios.append(Scenario(**k))

 # Помощники: iFVG/BPR/строгая FVG
 i_fvg_up  = detect_iFVG(df,gaps,"up",at,rules["min_depth_atr"],rules["impulse_mult"])
 i_fvg_dn  = detect_iFVG(df,gaps,"down",at,rules["min_depth_atr"],rules["impulse_mult"])
 bpr_zone  = detect_bpr(gaps)

 # 1) SFP → BOS → FVG (mid 0.5)
 if (swp["low"] and dir_bos=="up") or (swp["high"] and dir_bos=="down"):
  direction = "long" if dir_bos=="up" else "short"
  fvg_ok,notes = strict_fvg_validate("up" if direction=="long" else "down",gaps,
                                     price=price,vp=vp,atr_val=at,bos_time=t_bos,
                                     min_depth_atr=rules["min_depth_atr"],need_side=need_side)
  if fvg_ok:
   t,lo,hi=fvg_ok; mid=(lo+hi)/2; entry= mid+0.0 if direction=="long" else mid-0.0  # строго 0.5
   sl = (swp["low"][-1][1]-0.7*at) if direction=="long" else (swp["high"][-1][1]+0.7*at)
   tp1,tp2,rr=rr_targets(entry,sl,direction)
   base = ["SFP","BOS", "FVG валиден"]
   generic = gen_conf(direction,entry)
   confirms = base + generic
   missing=[]
   if "SFP" not in base: missing.append("SFP")
   if "BOS" not in base: missing.append("BOS")
   if "FVG валиден" not in base: missing.append("FVG")
   status="ok" if len(set(base))>=3 else "await"
   add(name="SFP→BOS→FVG",bias=direction,etype="limit",trigger=f"касание mid FVG {mid:.5f}",
       entry=float(entry),sl=float(sl),tp1=float(tp1),tp2=float(tp2),rr=rr,
       confirms=len(set(confirms)),confirm_list=confirms,explain_short="срыв→пробой→имбаланс (0.5)",
       stop_reason="за SFP ±0.7×ATR",tp_reason="структурная цель/POC",
       logic_path=["SFP","BOS","FVG"],status=status,missing=missing)

 # 2) BOS → OB Retest (с iFVG или BPR как доп)
 def ob_block(dir_):
  before=df[df.index<(t_bos if t_bos else df.index[-1])].iloc[-70:]
  if dir_=="up":
   reds=before[before.close<before.open]
   if len(reds):
    last=reds.iloc[-1]; return (float(min(last.open,last.close)),float(max(last.open,last.close)))
  else:
   greens=before[before.close>before.open]
   if len(greens):
    last=greens.iloc[-1]; return (float(min(last.open,last.close)),float(max(last.open,last.close)))
  return None
 if dir_bos=="up":
  ob=ob_block("up")
  if ob:
   lo,hi=ob; entry=hi; sl=lo-0.6*at; tp1,tp2,rr=rr_targets(entry,sl,"long")
   extra=[]
   if i_fvg_up: extra.append("iFVG")
   if bpr_zone: extra.append("BPR")
   base=["BOS↑","OB ретест"]+extra[:1]  # достаточно одного доп-типа для «разных»
   generic=gen_conf("long",entry); confirms=base+generic
   missing=[]
   for need in ["BOS","OB","FVG/BPR"]:
    if need=="BOS" and "BOS↑" not in base: missing.append("BOS")
    if need=="OB" and "OB ретест" not in base: missing.append("OB")
    if need=="FVG/BPR" and (("iFVG" not in base) and ("BPR" not in base)): missing.append("FVG/BPR")
   status="ok" if len(set(base))>=3 else "await"
   add(name="BOS→OB Retest",bias="long",etype="limit",trigger=f"OB {lo:.5f}-{hi:.5f}",
       entry=float(entry),sl=float(sl),tp1=float(tp1),tp2=float(tp2),rr=rr,
       confirms=len(set(confirms)),confirm_list=confirms,explain_short="пробой и ретест спроса (+iFVG/BPR)",
       stop_reason="за OB −0.6×ATR",tp_reason="структурная цель/POC",
       logic_path=["BOS","OB","iFVG/BPR"],status=status,missing=missing)
 if dir_bos=="down":
  ob=ob_block("down")
  if ob:
   lo,hi=ob; entry=lo; sl=hi+0.6*at; tp1,tp2,rr=rr_targets(entry,sl,"short")
   extra=[]
   if i_fvg_dn: extra.append("iFVG"); 
   if bpr_zone: extra.append("BPR")
   base=["BOS↓","OB ретест"]+extra[:1]
   generic=gen_conf("short",entry); confirms=base+generic
   missing=[]
   if "BOS↓" not in base: missing.append("BOS")
   if "OB ретест" not in base: missing.append("OB")
   if (("iFVG" not in base) and ("BPR" not in base)): missing.append("FVG/BPR")
   status="ok" if len(set(base))>=3 else "await"
   add(name="BOS→OB Retest",bias="short",etype="limit",trigger=f"OB {lo:.5f}-{hi:.5f}",
       entry=float(entry),sl=float(sl),tp1=float(tp1),tp2=float(tp2),rr=rr,
       confirms=len(set(confirms)),confirm_list=confirms,explain_short="пробой и ретест предложения (+iFVG/BPR)",
       stop_reason="за OB +0.6×ATR",tp_reason="структурная цель/POC",
       logic_path=["BOS","OB","iFVG/BPR"],status=status,missing=missing)

 # 3) Breaker (после свипа и возврата)
 if swp["high"] and dir_bos=="down":
  _,lv=swp["high"][-1]; entry=lv-0.1*at; sl=lv+0.7*at; tp1,tp2,rr=rr_targets(entry,sl,"short")
  base=["SFP high","BOS↓","return"]; generic=gen_conf("short",entry); confirms=base+generic
  missing=[]; status="ok" if len(set(base))>=3 else "await"
  add(name="Breaker",bias="short",etype="stop",trigger=f"возврат под {lv:.5f}",
      entry=float(entry),sl=float(sl),tp1=float(tp1),tp2=float(tp2),rr=rr,
      confirms=len(set(confirms)),confirm_list=confirms,explain_short="срыв high и возврат",
      stop_reason="над свип-уровнем +0.7×ATR",tp_reason="структурная цель/POC",
      logic_path=["SFP","BOS","ret"],status=status,missing=missing)
 if swp["low"] and dir_bos=="up":
  _,lv=swp["low"][-1]; entry=lv+0.1*at; sl=lv-0.7*at; tp1,tp2,rr=rr_targets(entry,sl,"long")
  base=["SFP low","BOS↑","return"]; generic=gen_conf("long",entry); confirms=base+generic
  missing=[]; status="ok" if len(set(base))>=3 else "await"
  add(name="Breaker",bias="long",etype="stop",trigger=f"возврат над {lv:.5f}",
      entry=float(entry),sl=float(sl),tp1=float(tp1),tp2=float(tp2),rr=rr,
      confirms=len(set(confirms)),confirm_list=confirms,explain_short="срыв low и возврат",
      stop_reason="под свип-уровнем −0.7×ATR",tp_reason="структурная цель/POC",
      logic_path=["SFP","BOS","ret"],status=status,missing=missing)

 # 4) Range reversion по VA краям
 if regime=="range" and float(adx(df).iloc[-1])<22:
  if abs(price-vp["val"])<=max(0.6*at,0.1*(vp["vah"]-vp["val"])):
   entry=vp["val"]+0.1*at; sl=vp["val"]-0.8*at; tp1,tp2,rr=rr_targets(entry,sl,"long")
   base=["VAL edge","range","POC target"]; generic=gen_conf("long",entry); confirms=base+generic
   missing=[]; status="ok" if len(set(base))>=3 else "await"
   add(name="Value Area Reversion",bias="long",etype="limit",trigger="от VAL к POC",
       entry=float(entry),sl=float(sl),tp1=float(tp1),tp2=float(tp2),rr=rr,
       confirms=len(set(confirms)),confirm_list=confirms,explain_short="от края к POC",
       stop_reason="под VAL −0.8×ATR",tp_reason="POC",logic_path=["VAL","range","POC"],status=status,missing=missing)
  if abs(price-vp["vah"])<=max(0.6*at,0.1*(vp["vah"]-vp["val"])):
   entry=vp["vah"]-0.1*at; sl=vp["vah"]+0.8*at; tp1,tp2,rr=rr_targets(entry,sl,"short")
   base=["VAH edge","range","POC target"]; generic=gen_conf("short",entry); confirms=base+generic
   missing=[]; status="ok" if len(set(base))>=3 else "await"
   add(name="Value Area Reversion",bias="short",etype="limit",trigger="от VAH к POC",
       entry=float(entry),sl=float(sl),tp1=float(tp1),tp2=float(tp2),rr=rr,
       confirms=len(set(confirms)),confirm_list=confirms,explain_short="от края к POC",
       stop_reason="над VAH +0.8×ATR",tp_reason="POC",logic_path=["VAH","range","POC"],status=status,missing=missing)

 # сортировка
 def _k(s:Scenario):
  sc=s.confirms
  if (s.name.startswith(("SFP→BOS→FVG","BOS→OB","Breaker")) and regime=="trend") or (s.name.startswith("Value Area") and regime=="range"): sc+=1
  if s.bias==htf_bias: sc+=0.5
  return -sc
 uniq=[]; seen=set()
 for s in sorted(scenarios,key=_k):
  key=(s.name,s.bias)
  if key in seen: continue
  seen.add(key); uniq.append(s)
  if len(uniq)>=8: break
 return uniq

# ========= Вероятности
def scenario_probabilities(scen,htf_bias,d_bias,price,vp,atr_val,regime,cap=0.9,floor=0.05,temp=1.1):
 if not scen: return {"Wait (no-trade)":100.0},{"long":0.0,"short":0.0}
 scores,labels=[],[]
 for s in scen:
  sc=0.7*s.confirms+(1.5 if s.bias==htf_bias else 0)+(1.0 if s.bias==d_bias else 0)
  if (s.name.startswith(("SFP→BOS→FVG","BOS→OB","Breaker")) and regime=="trend") or (s.name.startswith("Value Area") and regime=="range"): sc+=0.8
  dist=abs(s.entry-price)/max(atr_val,1e-6); sc+=(-1.0 if dist>2.0 else (-0.5 if dist>1.5 else 0))
  scores.append(sc); labels.append((s.name,s.bias))
 scores=np.array(scores)/temp; ex=np.exp(scores-scores.max()); p=np.clip(ex/ex.sum(),floor,cap); p=p/p.sum()
 out,agg={},{"long":0.0,"short":0.0}
 for (lbl,bias),pp in zip(labels,p): val=float(np.round(pp*100.0,2)); out[f"{lbl} ({bias})"]=val; agg[bias]+=val
 return dict(sorted(out.items(),key=lambda x:x[1],reverse=True)),{k:round(v,2) for k,v in agg.items()}

# ========= Streamlit UI
st.set_page_config(page_title="SMC Intraday (text)", layout="wide")
db_init(); rules=load_rules()

tab_signals, tab_journal, tab_stats, tab_rules = st.tabs(["📊 Сигналы","📒 Журнал","📈 Статистика","⚙️ Правила"])

with tab_signals:
 colA,colB,colC,colD,colE=st.columns([1.2,1,1,1,1])
 with colA: asset=st.selectbox("Актив",ASSETS,index=0)
 with colB: tf=st.selectbox("TF",["5m","15m","1h"],index=0)
 with colC: min_risk_pct=st.slider("Мин. риск (%ATR)",5,60,25,step=5)
 with colD: min_tp1_atr=st.slider("Мин. TP1 (×ATR)",1.0,3.0,1.5,step=0.25)
 with colE: min_confirms=st.slider("Мин. подтверждений",2,5,rules.get("min_confirms",3),step=1)
 colF,colG,colH=st.columns([1,1,1])
 with colF: refresh_mode=st.selectbox("Обновление",["Выключено","30s","1m","2m","5m"],index=0)
 with colG: tg_token=st.text_input("Telegram bot token", value="", type="password")
 with colH: tg_chat=st.text_input("Telegram chat id", value="")
 st.caption("2 из 3 — показываем как «ожидаем 3»; что именно не хватает — видно в карточке. FVG вход строго по 0.5 (mid).")

 if st.button("🔄 Обновить"): st.cache_data.clear(); st.experimental_rerun()
 INTERVALS={"30s":30,"1m":60,"2m":120,"5m":300}
 if "next_refresh_ts" not in st.session_state: st.session_state.next_refresh_ts=time.time()+10**9
 if refresh_mode!="Выключено":
  it=INTERVALS[refresh_mode]; now=time.time()
  if now>=st.session_state.next_refresh_ts:
   st.session_state.next_refresh_ts=now+it; st.cache_data.clear(); st.experimental_rerun()
 else: st.session_state.next_refresh_ts=time.time()+10**9

 try:
  df,tf_eff,_=yf_ohlc_first_success(asset,tf,limit=800); htf=HTF_OF[tf]; df_h,_,_=yf_ohlc_first_success(asset,htf,limit=400); df_d,_,_=yf_ohlc_first_success(asset,"1d",limit=600)
  price=float(df.close.iloc[-1]); vp=volume_profile(df); reg=market_regime(df,vp); atr_v=float(atr(df).iloc[-1])
  htf_bias=score_bias(df_h); d_bias=regime_daily(df_d)
  scen_all=propose(df,rules,htf_bias,d_bias,reg,vp,asset,tf)

  # фильтры + await
  min_risk=atr_v*(min_risk_pct/100.0); scen=[]; awaiting=[]
  for s in scen_all:
   risk_ok=abs(s.entry-s.sl)>=min_risk; tp1_ok=(abs(s.tp1-s.entry)/max(atr_v,1e-6))>=min_tp1_atr
   if risk_ok and tp1_ok:
    if s.confirms>=min_confirms and s.status=="ok": scen.append(s)
    else: awaiting.append(s)
  if not scen: scen=awaiting or scen_all

  probs,balance=scenario_probabilities(scen,htf_bias,d_bias,price,vp,atr_v,reg)
  st.markdown(f"### {asset} ({tf}) — цена: {price:.6f}")
  poc_state="выше VAH" if price>vp["vah"] else ("ниже VAL" if price<vp["val"] else "внутри value area")
  st.markdown(f"**Контекст:** LTF={score_bias(df).upper()}, HTF={htf_bias.upper()}, Daily={d_bias.upper()} • Режим: {reg.upper()} (ADX≈{float(adx(df).iloc[-1]):.1f}) • POC {vp['poc']:.5f}, VAL {vp['val']:.5f}, VAH {vp['vah']:.5f} → {poc_state}.")

  # Главная карточка
  if scen:
   top_key=list(probs.keys())[0] if probs else f"{scen[0].name} ({scen[0].bias})"; main=next((s for s in scen if f"{s.name} ({s.bias})"==top_key),scen[0])
   rr=round(abs(main.tp1-main.entry)/max(abs(main.entry-main.sl),1e-9),2)
   status_txt="готов (3/3+)" if (main.confirms>=3 and main.status=="ok") else f"{min(main.confirms,3)}/3 — ждём: {', '.join(main.missing) if main.missing else 'точку триггера'}"
   st.markdown(f"#### {'LONG' if main.bias=='long' else 'SHORT'} — {main.name} — {status_txt}")
   st.markdown(
    f"- **Подтв.:** {main.confirms} — {', '.join(main.confirm_list)}  \n"
    f"- **Вход:** {main.entry:.6f} • **Стоп:** {main.sl:.6f} ({main.stop_reason})  \n"
    f"- **Цели:** TP1 {main.tp1:.6f} ({main.tp_reason}), TP2 {main.tp2:.6f if main.tp2 else float('nan')} • **R:R≈{rr}**"
   )

  # Таблица сценариев + кнопки «в журнал»
  rows=[]
  for i,s in enumerate(scen,1):
   key=f"{s.name} ({s.bias})"; stx=("OK" if (s.confirms>=3 and s.status=="ok") else "2/3 ждём")
   cols=st.columns([3,1,1,1,1,1.2,1.2,1.2])
   with cols[0]:
    st.write(f"**{key}** — {stx}")
    st.caption(f"{s.trigger} • Подтв.: {', '.join(s.confirm_list)}")
   with cols[1]: st.write(f"{s.entry:.6f}")
   with cols[2]: st.write(f"{s.sl:.6f}")
   with cols[3]: st.write(f"{s.tp1:.6f}")
   with cols[4]: st.write(f"{round(abs(s.tp1-s.entry)/max(abs(s.entry-s.sl),1e-9),2)}R")
   with cols[5]:
    if st.button("В журнал: лимитка", key=f"j_addL_{i}"):
     db_upsert_trade({"ts":str(df.index[-1]),"asset":asset,"tf":tf,"name":s.name,"bias":s.bias,"entry":s.entry,"sl":s.sl,"tp1":s.tp1,"tp2":s.tp2,"confirms":s.confirms,"reasons":s.confirm_list,"status":s.status,"journal_state":"limit"})
     st.success("Добавлено как лимитка")
   with cols[6]:
    if st.button("В журнал: в рынке", key=f"j_addM_{i}"):
     db_upsert_trade({"ts":str(df.index[-1]),"asset":asset,"tf":tf,"name":s.name,"bias":s.bias,"entry":s.entry,"sl":s.sl,"tp1":s.tp1,"tp2":s.tp2,"confirms":s.confirms,"reasons":s.confirm_list,"status":s.status,"journal_state":"in_position"})
     st.success("Добавлено как в рынке")
   with cols[7]:
    st.write(f"Prob {probs.get(key,0):.1f}%")

  # Telegram: при появлении 2/3 или переходе в 3/3
  if tg_token and tg_chat and scen:
   for s in scen:
    level = "3of3" if (s.confirms>=3 and s.status=="ok") else ("2of3" if s.confirms>=2 else None)
    if not level: continue
    sig = f"{asset}|{tf}|{s.name}|{level}"
    if already_alerted(sig, cooldown_sec=300): continue
    try:
     text=f"{asset} {tf}: {s.name} {s.bias} — {'ГОТОВ 3/3' if level=='3of3' else '2/3, ждём 3'}\nEntry {s.entry:.6f} SL {s.sl:.6f} TP1 {s.tp1:.6f}"
     requests.get(f"https://api.telegram.org/bot{tg_token}/sendMessage", params={"chat_id":tg_chat,"text":text})
    except Exception: pass

 except Exception as e:
  st.error(f"{asset}: {e}")

with tab_journal:
 st.subheader("Журнал сделок (SQLite)")
 jrows=db_list_trades(300)
 if not jrows:
  st.info("Пока пусто. Добавляй из вкладки «Сигналы».")
 else:
  dfj=pd.DataFrame([{**r,**{"reasons":', '.join(json.loads(r["reasons"]) if isinstance(r["reasons"],str) else r["reasons"])}} for r in jrows])
  st.dataframe(dfj, use_container_width=True, hide_index=True)
  st.caption("Состояние: limit / in_position. Результат: open / tp / sl / cancel / mkt_close.")
  st.markdown("---")
  # Кнопки управления
  for r in jrows:
   c1,c2,c3,c4,c5=st.columns([3,1,1,1,1])
   with c1: st.write(f"#{r['id']} {r['ts']} • {r['asset']} {r['tf']} • {r['name']} {r['bias']} • entry {r['entry']:.6f} sl {r['sl']:.6f} tp1 {r['tp1']:.6f}")
   with c2:
    if st.button("TP", key=f"tp_{r['id']}"): db_update_trade_status(r['id'],result="tp"); st.experimental_rerun()
   with c3:
    if st.button("SL", key=f"sl_{r['id']}"): db_update_trade_status(r['id'],result="sl"); st.experimental_rerun()
   with c4:
    if st.button("Отмена", key=f"cn_{r['id']}"): db_update_trade_status(r['id'],result="cancel"); st.experimental_rerun()
   with c5:
    if st.button("Закрыть рыночн.", key=f"mk_{r['id']}"): db_update_trade_status(r['id'],result="mkt_close"); st.experimental_rerun()

with tab_stats:
 st.subheader("Краткая статистика по журналу")
 rows=db_list_trades(1000)
 if not rows:
  st.info("Нет данных для статистики.")
 else:
  dfp=pd.DataFrame(rows)
  total=len(dfp)
  win=int((dfp["result"]=="tp").sum()); lose=int((dfp["result"]=="sl").sum())
  winrate = (win/max(1,win+lose))*100
  st.write(f"Всего записей: **{total}**, TP: **{win}**, SL: **{lose}**, WinRate: **{winrate:.1f}%**")
  st.write("Разбивка по сетапам:")
  st.dataframe(dfp.groupby(["name","bias"])["result"].value_counts().unstack(fill_value=0), use_container_width=True)

with tab_rules:
 st.subheader("Правила подтверждений")
 rw=dict(rules)  # копия
 col1,col2,col3=st.columns(3)
 with col1:
  rw["windows"]["SFP"]=st.number_input("Окно SFP (баров)",1,400,int(rw["windows"]["SFP"]))
  rw["windows"]["BOS"]=st.number_input("Окно BOS (баров)",1,400,int(rw["windows"]["BOS"]))
 with col2:
  rw["windows"]["FVG"]=st.number_input("Окно FVG (баров)",1,400,int(rw["windows"]["FVG"]))
  rw["windows"]["iFVG"]=st.number_input("Окно iFVG (баров)",1,400,int(rw["windows"]["iFVG"]))
 with col3:
  rw["windows"]["BPR"]=st.number_input("Окно BPR (баров)",1,400,int(rw["windows"]["BPR"]))
  rw["min_depth_atr"]=st.number_input("Мин глубина по FVG (×ATR)",0.05,1.0,float(rw["min_depth_atr"]),step=0.05)
 st.checkbox("Требовать сторону POC",value=rw["need_poc_side"],key="need_poc_side_chk")
 rw["need_poc_side"]=st.session_state["need_poc_side_chk"]
 st.write("Веса (доп. бонусы):")
 cw=rw["weights"]
 cw["obv_align"]=st.slider("OBV в сторону",0.0,1.5,float(cw["obv_align"]),0.1)
 cw["ema_trend"]=st.slider("Тренд EMA20",0.0,1.5,float(cw["ema_trend"]),0.1)
 cw["poc_side"]=st.slider("Сторона POC",0.0,1.5,float(cw["poc_side"]),0.1)
 cw["vwap_near"]=st.slider("Рядом VWAP",0.0,1.5,float(cw["vwap_near"]),0.1)
 cw["divergence"]=st.slider("RSI дивергенция",0.0,1.5,float(cw["divergence"]),0.1)
 rw["min_confirms"]=st.slider("Мин. подтверждений по умолчанию",2,5,int(rw.get("min_confirms",3)))
 if st.button("💾 Сохранить правила"):
  save_rules(rw); st.success("Сохранено. Обнови вкладку Сигналы при необходимости.")
