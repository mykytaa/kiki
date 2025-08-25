# smc_dashboard.py
"""
SMC Intraday — BTCUSDT / ETHUSDT (text-only)
Качественный анализ: фракталы (L=3/R=3), BOS с подтверждением, FVG/OB,
VWAP/POC/Value Area, SFP (swing failure), EMA pullback, Breaker, BOS retest,
POC flip, Range reversion. Вероятности — по конфлюэнсам.
Beginner-карточка + подробное «market story» + справочник.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

# ========= Config =========
SYMBOLS = ["BTCUSDT", "ETHUSDT"]

# yfinance тикеры
TICKER_MAP = {"BTCUSDT": "BTC-USD", "ETHUSDT": "ETH-USD"}

# соответствие интервалов для yfinance
YF_INTERVAL = {"5m": ("5m", "5d"), "15m": ("15m", "60d"), "1h": ("60m", "365d")}
BINANCE_INTERVAL = {"5m": "5m", "15m": "15m", "1h": "1h"}  # для UI совместимости
HTF_MAP = {"5m": "15m", "15m": "1h", "1h": "4h"}  # 4h эмулируем через 60m (ниже берём 1h как суррогат)

LTF_LIMIT = 600
HTF_LIMIT = 400

# ========= Utils & indicators =========
def ema(x: pd.Series, n: int): return x.ewm(span=n, adjust=False).mean()

def rsi(x: pd.Series, n=14):
    d = x.diff()
    up = (d.clip(lower=0)).ewm(alpha=1/n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    rs = up / (dn.replace(0, np.nan))
    out = 100 - 100/(1+rs)
    return out.fillna(method="bfill").fillna(50)

# ==== TradingView helpers (Pine v5 export) ====

def _tv_symbol(symbol: str) -> str:
    # подберите мэппинг под ваш источник данных
    # обычный вариант для крипты на Binance:
    m = {"BTCUSDT": "BINANCE:BTCUSDT", "ETHUSDT": "BINANCE:ETHUSDT"}
    return m.get(symbol, symbol)

def _tv_interval(tf: str) -> str:
    return {"5m":"5","15m":"15","1h":"60"}.get(tf, "15")

def tv_chart_url(symbol: str, tf: str) -> str:
    return f"https://www.tradingview.com/chart/?symbol={_tv_symbol(symbol)}&interval={_tv_interval(tf)}"

def pine_for_scenario(symbol: str, tf: str, sc) -> str:
    """Генерим Pine v5 для отрисовки Entry/SL/TP1/TP2, стороны и R:R."""
    side = "LONG" if sc.bias == "long" else "SHORT"
    tvsym = _tv_symbol(symbol)
    tfn = _tv_interval(tf)
    return f"""//@version=5
indicator("SMC Idea — {symbol} {tf} — {sc.name} ({side})", overlay=true, timeframe="{tfn}", timeframe_gaps=true)

// ---- levels ----
var float entry = {sc.entry}
var float sl    = {sc.sl}
var float tp1   = {sc.tp1}
var float tp2   = {('na' if sc.tp2 is None else sc.tp2)}
var string side = "{side}"
var string name = "{sc.name}"

// ---- styling ----
col_entry = color.new(color.teal, 0)
col_sl    = color.new(color.red, 0)
col_tp1   = color.new(color.green, 0)
col_tp2   = color.new(color.green, 30)

// ---- plot ----
plot(entry, "ENTRY", col_entry, 2, plot.style_linebr)
plot(sl,    "SL",    col_sl,    2, plot.style_linebr)
plot(tp1,   "TP1",   col_tp1,   2, plot.style_linebr)
plot(tp2,   "TP2",   col_tp2,   2, plot.style_linebr)

// ---- labels (появятся у последней свечи) ----
if barstate.islast
    label.new(bar_index, entry, "ENTRY\\n" + str.tostring(entry, format.mintick), style=label.style_label_left, color=col_entry, textcolor=color.white)
    label.new(bar_index, sl,    "SL\\n" + str.tostring(sl, format.mintick),       style=label.style_label_left, color=col_sl, textcolor=color.white)
    label.new(bar_index, tp1,   "TP1\\n" + str.tostring(tp1, format.mintick),     style=label.style_label_left, color=col_tp1, textcolor=color.white)
    if not na(tp2)
        label.new(bar_index, tp2, "TP2\\n" + str.tostring(tp2, format.mintick),   style=label.style_label_left, color=col_tp2, textcolor=color.white)

// ---- info panel ----
rr = math.abs(tp1 - entry) / math.abs(entry - sl)
txt = name + " (" + side + ")\\n" +
      "Entry: " + str.tostring(entry, format.mintick) + "\\n" +
      "SL: "    + str.tostring(sl,    format.mintick) + "\\n" +
      "TP1: "   + str.tostring(tp1,   format.mintick) + "\\n" +
      "TP2: "   + (na(tp2) ? "—" : str.tostring(tp2, format.mintick)) + "\\n" +
      "R:R to TP1 ≈ " + str.tostring(rr, format.mintick)
var label panel = na
if barstate.islast
    panel := label.new(bar_index, high, txt, style=label.style_label_upper_left, textcolor=color.white, color=color.new(color.black, 0))
"""


def macd(x: pd.Series):
    f = ema(x, 12); s = ema(x, 26); m = f - s; sig = ema(m, 9)
    return m, sig, m - sig

def atr(df: pd.DataFrame, n=14):
    c = df["close"]
    tr = pd.concat(
        [(df["high"]-df["low"]),
         (df["high"]-c.shift()).abs(),
         (df["low"]-c.shift()).abs()],
        axis=1
    ).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False).mean()

def obv(df: pd.DataFrame) -> pd.Series:
    sign = np.sign(df["close"].diff().fillna(0.0))
    return (sign*df["volume"]).cumsum()

def adx(df: pd.DataFrame, n: int = 14) -> pd.Series:
    up = df["high"].diff(); dn = -df["low"].diff()
    plus_dm = np.where((up>dn) & (up>0), up, 0.0)
    minus_dm = np.where((dn>up) & (dn>0), dn, 0.0)
    tr = pd.concat(
        [(df["high"]-df["low"]),
         (df["high"]-df["close"].shift()).abs(),
         (df["low"]-df["close"].shift()).abs()],
        axis=1
    ).max(axis=1)
    atr_s = tr.ewm(alpha=1/n, adjust=False).mean()
    pdi = 100*pd.Series(plus_dm, index=df.index).ewm(alpha=1/n, adjust=False).mean()/atr_s
    mdi = 100*pd.Series(minus_dm, index=df.index).ewm(alpha=1/n, adjust=False).mean()/atr_s
    dx = 100*(pdi.subtract(mdi).abs() / (pdi+mdi).replace(0, np.nan))
    return dx.ewm(alpha=1/n, adjust=False).mean().fillna(20)

# ========= Volume Profile (approx) =========
def volume_profile(df: pd.DataFrame, bins: int = 40) -> Dict[str, float | np.ndarray]:
    lo = float(df["low"].min()); hi = float(df["high"].max())
    if hi<=lo: hi = lo+1e-6
    edges = np.linspace(lo, hi, bins+1); vol = np.zeros(bins)
    prices = df["close"].values; vols = df["volume"].values
    idx = np.clip(np.digitize(prices, edges)-1, 0, bins-1)
    for i, v in zip(idx, vols): vol[i]+=v
    total = max(vol.sum(),1.0)
    poc_i = int(vol.argmax()); poc = (edges[poc_i]+edges[poc_i+1])/2
    area=[poc_i]; L=poc_i-1; R=poc_i+1; acc=vol[poc_i]
    while acc<0.7*total and (L>=0 or R<bins):
        if R>=bins or (L>=0 and vol[L]>=vol[R]): area.append(L); acc+=vol[L]; L-=1
        else: area.append(R); acc+=vol[R]; R+=1
    val = edges[max(min(area),0)]; vah = edges[min(max(area)+1,bins)]
    return {"edges":edges,"volume":vol,"poc":float(poc),"val":float(val),"vah":float(vah)}

def liquidity_pools(df, SH, SL, win=260):
    rec = df.iloc[-win:]
    bsl = [(t, float(df.loc[t, "high"])) for t in rec[SH.loc[rec.index]].index]
    ssl = [(t, float(df.loc[t, "low"]))  for t in rec[SL.loc[rec.index]].index]
    return {"BSL": bsl, "SSL": ssl}

# ========= VWAP =========
def vwap_series(df: pd.DataFrame) -> pd.Series:
    tp  = (df["high"]+df["low"]+df["close"])/3.0
    vol = df["volume"].replace(0, np.nan).fillna(0.0)
    num = (tp*vol).cumsum(); den = vol.cumsum().replace(0, np.nan)
    return (num/den).fillna(method="bfill").fillna(df["close"])

# ========= SMC primitives (фракталы L=3/R=3) =========
def swings(df: pd.DataFrame, L=3, R=3):
    hi=df["high"].values; lo=df["low"].values; n=len(df)
    SH=np.zeros(n,bool); SL=np.zeros(n,bool)
    for i in range(L, n-R):
        if hi[i]==hi[i-L:i+R+1].max(): SH[i]=True
        if lo[i]==lo[i-L:i+R+1].min(): SL[i]=True
    return pd.Series(SH, df.index), pd.Series(SL, df.index)

# BOS с подтверждением: закрытие за уровнем ≥ confirm_mult*ATR
def bos(df, SH, SL, look=200, confirm_mult=0.30):
    recent = df.iloc[-look:]
    sh_idx = recent[SH.loc[recent.index]].index
    sl_idx = recent[SL.loc[recent.index]].index
    last_sh = recent.loc[sh_idx[-1]] if len(sh_idx) else None
    last_sl = recent.loc[sl_idx[-1]] if len(sl_idx) else None
    a = float(atr(df).iloc[-1]) or 1e-6
    if last_sh is not None:
        lvl = last_sh.high; post = recent[recent.index>last_sh.name]
        brk = post[post["close"] > lvl + confirm_mult*a]
        if len(brk): return "up", brk.index[0], lvl
    if last_sl is not None:
        lvl = last_sl.low; post = recent[recent.index>last_sl.name]
        brk = post[post["close"] < lvl - confirm_mult*a]
        if len(brk): return "down", brk.index[0], lvl
    return None, None, None

def sweeps(df, SH, SL, win=180):
    res={"high":[], "low":[]}; rec=df.iloc[-win:]
    for t in rec[SH.loc[rec.index]].index:
        level=df.loc[t,"high"]; post=rec[rec.index>t]
        if len(post[(post["high"]>level) & (post["close"]<level)]): res["high"].append((t, level))
    for t in rec[SL.loc[rec.index]].index:
        level=df.loc[t,"low"]; post=rec[rec.index>t]
        if len(post[(post["low"]<level) & (post["close"]>level)]): res["low"].append((t, level))
    return res

def fvg(df, look=140):
    out={"bull":[], "bear":[]}; hi=df["high"].values; lo=df["low"].values; idx=df.index
    n=len(df); s=max(2, n-look)
    for i in range(s,n):
        if i-2>=0 and lo[i]>hi[i-2]: out["bull"].append((idx[i], hi[i-2], lo[i]))
        if i-2>=0 and hi[i]<lo[i-2]: out["bear"].append((idx[i], hi[i], lo[i-2]))
    return out

def simple_ob(df, dir_, t, back=70):
    res={"demand":None,"supply":None}
    if dir_ is None or t is None: return res
    before=df[df.index<t].iloc[-back:]
    if dir_=="up":
        reds=before[before["close"]<before["open"]]
        if len(reds):
            last=reds.iloc[-1]
            res["demand"]=(last.name,float(min(last["open"],last["close"])),float(max(last["open"],last["close"])))
    if dir_=="down":
        greens=before[before["close"]>before["open"]]
        if len(greens):
            last=greens.iloc[-1]
            res["supply"]=(last.name,float(min(last["open"],last["close"])),float(max(last["open"],last["close"])))
    return res

def last_swing_levels(df: pd.DataFrame, SH: pd.Series, SL: pd.Series) -> Tuple[Optional[float], Optional[float]]:
    sh_idx = SH[SH].index; sl_idx = SL[SL].index
    sh_lvl = float(df.loc[sh_idx[-1],"high"]) if len(sh_idx) else None
    sl_lvl = float(df.loc[sl_idx[-1],"low"])  if len(sl_idx) else None
    return sh_lvl, sl_lvl

# ========= Bias & regime =========
def score_bias(df: pd.DataFrame) -> str:
    c=df["close"]; s=0
    r=float(rsi(c).iloc[-1]); h=float(macd(c)[2].iloc[-1])
    if r>55: s+=1
    elif r<45: s-=1
    if h>0: s+=1
    elif h<0: s-=1
    return "long" if s>=1 else ("short" if s<=-1 else "none")

def regime_daily(df_d: pd.DataFrame) -> str:
    e50=ema(df_d["close"],50).iloc[-1]; e200=ema(df_d["close"],200).iloc[-1]
    if np.isnan(e50) or np.isnan(e200): return "none"
    return "long" if e50>e200 else "short" if e50<e200 else "none"

def market_regime(df: pd.DataFrame, vp: Dict[str,float|np.ndarray]) -> str:
    ad=float(adx(df).iloc[-1]); price=float(df["close"].iloc[-1])
    outside=(price>vp["vah"]) or (price<vp["val"])
    return "trend" if (ad>=22 or outside) else "range"

# ========= RR / EV =========
def rr_targets(entry: float, sl: float, bias: str, min_rr: float = 2.0) -> Tuple[float,float,str]:
    risk=abs(entry-sl) or 1e-6
    if bias=="long": tp1=entry+min_rr*risk; tp2=entry+3.0*risk
    else:            tp1=entry-min_rr*risk; tp2=entry-3.0*risk
    return tp1, tp2, f"1:{int(min_rr)}/1:3"

def scenario_ev(entry, sl, tp1, prob):
    risk=abs(entry-sl); reward=abs(tp1-entry)
    fees=0.0002*(risk+reward)
    return prob*reward - (1-prob)*risk - fees

# ========= Scenario =========
@dataclass
class Scenario:
    name: str
    bias: str
    etype: str
    trigger: str
    entry: float
    sl: float
    tp1: float
    tp2: Optional[float]
    rr: str
    explain: str

TREND_TYPES = ("FVG","BOS","OB Retest","EMA Pullback","Structure Breakout","VWAP","POC Flip","SFP")
RANGE_TYPES = ("Value Area","Breaker")

# — разнообразное объяснение сетапов
def explain_scenario(name: str, bias: str, details: Dict[str, float | str]) -> str:
    side="LONG" if bias=="long" else "SHORT"; parts=[]
    if name=="FVG mitigation":
        parts+=["Откат к середине актуального FVG по направлению BOS.",
                f"Подтверждение: свеча закрепляется в пользу {side} после касания mid-FVG."]
    elif name=="OB Retest":
        parts+=["Ретест импульсного Order Block по тренду.",
                "Подтверждение: отбой от границы блока; хвосты внутри OB слабые."]
    elif name=="BOS Break & Retest":
        lvl=details.get("lvl"); parts+=[
            f"Пробой ключевого swing-уровня {lvl:.2f} и ретест.",
            "Подтверждение: свеча закрывается за уровнем; ретест даёт импульс в сторону пробоя."]
    elif name=="Breaker":
        lvl=details.get("lvl"); parts+=[
            f"Свип ликвидности на {lvl:.2f} и возврат (breaker).",
            "Подтверждение: ложный прокол и уверенное закрытие обратно."]
    elif name=="Value Area Reversion":
        edge=details.get("edge"); parts+=[
            f"От края value area ({edge}) к POC (mean-reversion).",
            "Подтверждение: разворотный паттерн у VAL/VAH при низком ADX."]
    elif name=="EMA Pullback":
        parts+=["Продолжение тренда через откат к EMA20.",
                "Подтверждение: касание EMA и возобновление импульса."]
    elif name=="Structure Breakout":
        lvl=details.get("lvl"); parts+=[
            f"Пробой последнего фрактала {lvl:.2f} и продолжение.",
            "Подтверждение: ускорение без глубокого возврата."]
    elif name=="VWAP Bounce":
        parts+=["Работа от VWAP: возврат к балансу объёма.",
                "Подтверждение: удержание VWAP и согласованный OBV."]
    elif name=="POC Flip":
        parts+=["Перехват контроля у POC (flip) и ретест с обратной стороны.",
                "Подтверждение: POC становится поддержкой/сопротивлением."]
    elif name=="SFP Reversal":
        lvl=details.get("lvl"); parts+=[
            f"SFP на {lvl:.2f}: прокол и закрытие обратно → вероятен реверс.",
            "Подтверждение: сильная разворотная свеча; хвост больше тела."]
    else:
        parts+=["Сетап по контексту и уровням."]
    parts.append("Отмена: сильное закрепление по другую сторону стоп-уровня.")
    return " ".join(parts)

# ========= Propose scenarios =========
def propose(df: pd.DataFrame, htf_bias: str, d_bias: str, regime: str,
            vp: Dict[str,float|np.ndarray], obv_slope: float) -> List[Scenario]:
    price=float(df["close"].iloc[-1]); at=float(atr(df).iloc[-1]); at=max(at,1e-6)
    SH,SL=swings(df); dir_,t,_=bos(df,SH,SL); gaps=fvg(df)
    pools=liquidity_pools(df,SH,SL); ob=simple_ob(df,dir_,t)
    sh_lvl, sl_lvl = last_swing_levels(df, SH, SL)
    vw = float(vwap_series(df).iloc[-1]); ema20=float(ema(df["close"],20).iloc[-1])

    sc: List[Scenario]=[]

    # FVG
    if dir_=="up" and gaps["bull"] and regime=="trend":
        _,lo,hi=list(reversed(gaps["bull"]))[0]; e=(lo+hi)/2; sl=e-1.2*at
        tp1,tp2,rr=rr_targets(e,sl,"long")
        sc.append(Scenario("FVG mitigation","long","limit",f"mid FVG {e:.2f}",e,sl,tp1,tp2,rr,
                           explain_scenario("FVG mitigation","long",{})))
    if dir_=="down" and gaps["bear"] and regime=="trend":
        _,lo,hi=list(reversed(gaps["bear"]))[0]; e=(lo+hi)/2; sl=e+1.2*at
        tp1,tp2,rr=rr_targets(e,sl,"short")
        sc.append(Scenario("FVG mitigation","short","limit",f"mid FVG {e:.2f}",e,sl,tp1,tp2,rr,
                           explain_scenario("FVG mitigation","short",{})))

    # OB Retest
    if dir_=="up" and ob.get("demand") and regime=="trend":
        _,lo,hi=ob["demand"]; e=hi; sl=lo-0.6*at; tp1,tp2,rr=rr_targets(e,sl,"long")
        sc.append(Scenario("OB Retest","long","limit",f"retest OB {lo:.2f}-{hi:.2f}",e,sl,tp1,tp2,rr,
                           explain_scenario("OB Retest","long",{})))
    if dir_=="down" and ob.get("supply") and regime=="trend":
        _,lo,hi=ob["supply"]; e=lo; sl=hi+0.6*at; tp1,tp2,rr=rr_targets(e,sl,"short")
        sc.append(Scenario("OB Retest","short","limit",f"retest OB {lo:.2f}-{hi:.2f}",e,sl,tp1,tp2,rr,
                           explain_scenario("OB Retest","short",{})))

    # BOS Break & Retest
    if dir_=="up" and t is not None:
        lvl=float(df.loc[t,"high"]); e=lvl+0.2*at; sl=lvl-0.8*at
        tp1,tp2,rr=rr_targets(e,sl,"long")
        sc.append(Scenario("BOS Break & Retest","long","stop",f"пробой {lvl:.2f}+ретест",e,sl,tp1,tp2,rr,
                           explain_scenario("BOS Break & Retest","long",{"lvl":lvl})))
    if dir_=="down" and t is not None:
        lvl=float(df.loc[t,"low"]); e=lvl-0.2*at; sl=lvl+0.8*at
        tp1,tp2,rr=rr_targets(e,sl,"short")
        sc.append(Scenario("BOS Break & Retest","short","stop",f"пробой {lvl:.2f}+ретест",e,sl,tp1,tp2,rr,
                           explain_scenario("BOS Break & Retest","short",{"lvl":lvl})))

    # Breaker после свипа
    sw=sweeps(df,SH,SL)
    if sw["high"] and dir_=="down":
        _,lvl_s=sw["high"][-1]; e=lvl_s-0.1*at; sl=lvl_s+0.7*at
        tp1,tp2,rr=rr_targets(e,sl,"short")
        sc.append(Scenario("Breaker","short","stop",f"breaker после свипа {lvl_s:.2f}",e,sl,tp1,tp2,rr,
                           explain_scenario("Breaker","short",{"lvl":lvl_s})))
    if sw["low"] and dir_=="up":
        _,lvl_s=sw["low"][-1]; e=lvl_s+0.1*at; sl=lvl_s-0.7*at
        tp1,tp2,rr=rr_targets(e,sl,"long")
        sc.append(Scenario("Breaker","long","stop",f"breaker после свипа {lvl_s:.2f}",e,sl,tp1,tp2,rr,
                           explain_scenario("Breaker","long",{"lvl":lvl_s})))

    # Range: VA→POC
    if regime=="range":
        if abs(price-vp["val"]) <= max(0.6*at, 0.1*(vp["vah"]-vp["val"])):
            e=vp["val"]+0.1*at; sl=vp["val"]-0.8*at; tp1,tp2,rr=rr_targets(e,sl,"long")
            sc.append(Scenario("Value Area Reversion","long","limit","от VAL к POC",e,sl,tp1,tp2,rr,
                               explain_scenario("Value Area Reversion","long",{"edge":"VAL"})))
        if abs(price-vp["vah"]) <= max(0.6*at, 0.1*(vp["vah"]-vp["val"])):
            e=vp["vah"]-0.1*at; sl=vp["vah"]+0.8*at; tp1,tp2,rr=rr_targets(e,sl,"short")
            sc.append(Scenario("Value Area Reversion","short","limit","от VAH к POC",e,sl,tp1,tp2,rr,
                               explain_scenario("Value Area Reversion","short",{"edge":"VAH"})))

    # EMA pullback
    if regime=="trend" and price>ema20:
        e=ema20; sl=e-1.2*at; tp1,tp2,rr=rr_targets(e,sl,"long")
        sc.append(Scenario("EMA Pullback","long","limit",f"к EMA20 {e:.2f}",e,sl,tp1,tp2,rr,
                           explain_scenario("EMA Pullback","long",{})))
    if regime=="trend" and price<ema20:
        e=ema20; sl=e+1.2*at; tp1,tp2,rr=rr_targets(e,sl,"short")
        sc.append(Scenario("EMA Pullback","short","limit",f"к EMA20 {e:.2f}",e,sl,tp1,tp2,rr,
                           explain_scenario("EMA Pullback","short",{})))

    # Structure Breakout
    if sh_lvl is not None:
        base_sl=(sl_lvl if sl_lvl is not None else price-1.2*at)
        e=sh_lvl+0.2*at; sl=base_sl-0.4*at; tp1,tp2,rr=rr_targets(e,sl,"long")
        sc.append(Scenario("Structure Breakout","long","stop",f"breakout swing-high {sh_lvl:.2f}",e,sl,tp1,tp2,rr,
                           explain_scenario("Structure Breakout","long",{"lvl":sh_lvl})))
    if sl_lvl is not None:
        base_sl=(sh_lvl if sh_lvl is not None else price+1.2*at)
        e=sl_lvl-0.2*at; sl=base_sl+0.4*at; tp1,tp2,rr=rr_targets(e,sl,"short")
        sc.append(Scenario("Structure Breakout","short","stop",f"breakdown swing-low {sl_lvl:.2f}",e,sl,tp1,tp2,rr,
                           explain_scenario("Structure Breakout","short",{"lvl":sl_lvl})))

    # VWAP bounce
    if abs(price-vw) <= 1.2*at:
        if obv_slope>0:
            e=vw; sl=e-1.0*at; tp1,tp2,rr=rr_targets(e,sl,"long")
            sc.append(Scenario("VWAP Bounce","long","limit",f"от VWAP {vw:.2f}",e,sl,tp1,tp2,rr,
                               explain_scenario("VWAP Bounce","long",{})))
        if obv_slope<0:
            e=vw; sl=e+1.0*at; tp1,tp2,rr=rr_targets(e,sl,"short")
            sc.append(Scenario("VWAP Bounce","short","limit",f"от VWAP {vw:.2f}",e,sl,tp1,tp2,rr,
                               explain_scenario("VWAP Bounce","short",{})))

    # POC flip
    if abs(price-vp["poc"]) <= 0.6*at:
        if price>vp["poc"]:
            e=vp["poc"]+0.1*at; sl=vp["poc"]-0.7*at; tp1,tp2,rr=rr_targets(e,sl,"long")
            sc.append(Scenario("POC Flip","long","limit","ретест POC сверху",e,sl,tp1,tp2,rr,
                               explain_scenario("POC Flip","long",{})))
        else:
            e=vp["poc"]-0.1*at; sl=vp["poc"]+0.7*at; tp1,tp2,rr=rr_targets(e,sl,"short")
            sc.append(Scenario("POC Flip","short","limit","ретест POC снизу",e,sl,tp1,tp2,rr,
                               explain_scenario("POC Flip","short",{})))

    # SFP (swing failure)
    if sh_lvl is not None and df["high"].iloc[-1]>sh_lvl and df["close"].iloc[-1]<sh_lvl:
        e=sh_lvl-0.1*at; sl=sh_lvl+0.9*at; tp1,tp2,rr=rr_targets(e,sl,"short")
        sc.append(Scenario("SFP Reversal","short","stop",f"SFP у {sh_lvl:.2f}",e,sl,tp1,tp2,rr,
                           explain_scenario("SFP Reversal","short",{"lvl":sh_lvl})))
    if sl_lvl is not None and df["low"].iloc[-1]<sl_lvl and df["close"].iloc[-1]>sl_lvl:
        e=sl_lvl+0.1*at; sl=sl_lvl-0.9*at; tp1,tp2,rr=rr_targets(e,sl,"long")
        sc.append(Scenario("SFP Reversal","long","stop",f"SFP у {sl_lvl:.2f}",e,sl,tp1,tp2,rr,
                           explain_scenario("SFP Reversal","long",{"lvl":sl_lvl})))

    # сортировка и уникальность
    def _sort_key(x: Scenario):
        base=0
        if (x.name.startswith(TREND_TYPES) and regime=="trend") or (x.name.startswith(RANGE_TYPES) and regime=="range"): base-=2
        if x.bias==htf_bias: base-=1
        return base

    sc=sorted(sc, key=_sort_key)
    uniq=[]; seen=set()
    for s in sc:
        key=(s.name, s.bias)
        if key in seen: continue
        seen.add(key); uniq.append(s)
        if len(uniq)>=8: break
    if not uniq:
        c=price
        uniq.append(Scenario("Wait (no-trade)","none","—","нет валидных сетапов",c,c,c,c,"—",
                             "Пауза: дождаться свипа/ретеста ключевых уровней."))
    return uniq

# ========= Probabilities (по конфлюэнсам) =========
def scenario_probabilities(
    scen: List[Scenario], htf_bias: str, d_bias: str, obv_slope: float,
    price: float, vp: Dict[str,float|np.ndarray], atr_val: float, regime: str,
    *, cap: float = 0.90, floor: float = 0.05, temp: float = 1.25
) -> Tuple[Dict[str, float], Dict[str, float]]:
    if not scen: return {"Wait (no-trade)":100.0}, {"long":0.0,"short":0.0}
    scores=[]; labels=[]
    for s in scen:
        if s.name.startswith("Wait"): continue
        sc=0.0
        if s.bias==htf_bias: sc+=2.2
        if s.bias==d_bias:  sc+=1.2
        sc += 0.9 if (obv_slope>0 and s.bias=="long") or (obv_slope<0 and s.bias=="short") else -0.2
        if (s.name.startswith(TREND_TYPES) and regime=="trend") or (s.name.startswith(RANGE_TYPES) and regime=="range"): sc+=1.0
        if regime=="range" and s.name.startswith("Structure"): sc-=0.6
        dist=abs(s.entry-price)/max(atr_val,1e-6)
        if dist>2.0: sc-=1.0
        elif dist>1.5: sc-=0.5
        above_poc = price>vp["poc"]
        if s.name.startswith(("FVG","BOS","OB","EMA","Structure","POC Flip","VWAP","SFP")):
            sc += 0.4 if (above_poc and s.bias=="long") or ((not above_poc) and s.bias=="short") else 0.0
        if s.name.startswith("Value Area"):
            near_edge = min(abs(price-vp["val"]), abs(price-vp["vah"])) <= 0.8*atr_val
            sc += 0.5 if near_edge else -0.2
        scores.append(sc); labels.append((s.name, s.bias))
    if not scores: return {"Wait (no-trade)":100.0}, {"long":0.0,"short":0.0}
    scores=np.array(scores,dtype=float)/temp
    ex=np.exp(scores - scores.max()); probs=ex/ex.sum()
    probs=np.clip(probs,floor,cap); probs=probs/probs.sum()
    out={}; agg={"long":0.0,"short":0.0}
    for (lbl,bias),p in zip(labels,probs):
        val=float(np.round(p*100.0,2)); out[f"{lbl} ({bias})"]=val; agg[bias]+=val
    out=dict(sorted(out.items(), key=lambda x:x[1], reverse=True))
    return out, {k:round(v,2) for k,v in agg.items()}

# ========= Readable analysis + market story =========
def _fmt_price(x: float) -> str: return f"{x:,.2f}".replace(","," ")
def _fmt_pct(x: float)   -> str: return f"{x:.1f}%"
def _rr(entry: float, sl: float, tp1: float) -> float:
    risk=abs(entry-sl) or 1e-6; reward=abs(tp1-entry); return round(reward/risk,2)

def market_story(df: pd.DataFrame, df_h: pd.DataFrame, df_d: pd.DataFrame,
                 vp: Dict[str,float|np.ndarray], regime: str) -> str:
    price=float(df["close"].iloc[-1]); ad=float(adx(df).iloc[-1])
    rsi_now=float(rsi(df["close"]).iloc[-1]); macd_hist=float(macd(df["close"])[2].iloc[-1])
    o=obv(df); wnd=min(len(o),160); obv_slope = (np.polyfit(np.arange(wnd), o.tail(wnd), 1)[0] if wnd>=20 else 0.0)
    vw=float(vwap_series(df).iloc[-1]); ema20=float(ema(df["close"],20).iloc[-1])
    poc_state="выше VAH" if price>vp["vah"] else ("ниже VAL" if price<vp["val"] else "внутри value area")
    bias_ltf=score_bias(df); bias_htf=score_bias(df_h); bias_d=regime_daily(df_d)

    # простенькие дивергенции RSI по последним двум swing-high/low
    SH,SL=swings(df)
    def _last_two(levels: pd.Series) -> List[pd.Timestamp]:
        idx=levels[levels].index; return list(idx[-2:]) if len(idx)>=2 else list(idx)
    bear_div = False; bull_div = False
    sh_list=_last_two(SH); sl_list=_last_two(SL)
    if len(sh_list)==2:
        p1,p2=float(df.loc[sh_list[-2],"high"]), float(df.loc[sh_list[-1],"high"])
        r1,r2=float(rsi(df["close"]).loc[sh_list[-2]]), float(rsi(df["close"]).loc[sh_list[-1]])
        bear_div = (p2>p1 and r2<r1)
    if len(sl_list)==2:
        p1,p2=float(df.loc[sl_list[-2],"low"]), float(df.loc[sl_list[-1],"low"])
        r1,r2=float(rsi(df["close"]).loc[sl_list[-2]]), float(rsi(df["close"]).loc[sl_list[-1]])
        bull_div = (p2<p1 and r2>r1)

    hints=[]
    hints.append(f"Режим: **{regime.upper()}** (ADX≈{ad:.1f}); цена {poc_state}.")
    hints.append(f"Поток: OBV {'растёт' if obv_slope>0 else 'падает' if obv_slope<0 else 'плоский'}; RSI≈{rsi_now:.0f}, MACD-hist {'+' if macd_hist>0 else '-'}.")
    hints.append(f"Баланс ТФ: LTF={bias_ltf.upper()}, HTF={bias_htf.upper()}, Daily={bias_d.upper()}.")
    hints.append(f"Относительно VWAP/EMA20: цена {'выше' if price>vw else 'ниже'} VWAP и "
                 f"{'выше' if price>ema20 else 'ниже'} EMA20.")
    if bear_div: hints.append("Есть **медвежья дивергенция RSI** на последних хайях → риск отката выше среднего.")
    if bull_div: hints.append("Есть **бычья дивергенция RSI** на последних лоях → шанс реверса выше среднего.")
    if regime=="range":
        hints.append("Боковик: от VAL/VAH чаще возвращаемся к POC (mean-reversion); импульсы часто затухают.")
    else:
        hints.append("Тренд: откаты к EMA20/VWAP и продолжение — приоритетные идеи; контртренд только по сильным сигналам (SFP/Breaker).")

    return "• " + "\n• ".join(hints)

def readable_analysis(symbol: str, tf: str,
                      df_ltf: pd.DataFrame, df_htf: pd.DataFrame, df_d: pd.DataFrame,
                      scen: List[Scenario], vp: Dict[str,float|np.ndarray],
                      sc_probs: Dict[str,float], bias_summary: Dict[str,float], regime: str) -> str:
    price=float(df_ltf["close"].iloc[-1])
    top=list(sc_probs.items())[0] if sc_probs else ("Wait (no-trade)",100.0)
    lines=[
        f"**Контекст (TF {tf} / HTF {HTF_MAP[tf]} / Daily)**",
        market_story(df_ltf, df_htf, df_d, vp, regime),
        f"**Самый вероятный:** {top[0]} ≈ {top[1]:.1f}%",
        f"**Баланс сценариев:** LONG {bias_summary['long']:.1f}% vs SHORT {bias_summary['short']:.1f}%",
        f"Текущая цена: **{_fmt_price(price)}**"
    ]
    return "\n\n".join(lines)

# ========= Beginner helpers =========
def make_beginner_summary(symbol: str, tf: str, price: float,
                          regime: str, ltf_bias: str, htf_bias: str, d_bias: str,
                          vp: Dict[str,float|np.ndarray],
                          top_name: str, top_prob: float,
                          long_vs_short: Dict[str,float]) -> str:
    where_price = ("выше диапазона (VAH)" if price>vp["vah"] else
                   "ниже диапазона (VAL)" if price<vp["val"] else
                   "внутри нормального диапазона (между VAL и VAH)")
    regime_txt = "трендовый рынок" if regime=="trend" else "боковик/флэт"
    return (
        f"**{symbol} ({tf}) — цена:** {_fmt_price(price)}\n\n"
        f"**Простая картинка:** сейчас {regime_txt}; на младшем ТФ — {ltf_bias.upper()}, "
        f"на старшем — {htf_bias.upper()}, по дневному — {d_bias.upper()}. Цена {where_price}.\n\n"
        f"**Главный план:** {top_name} (вероятность ≈ {_fmt_pct(top_prob)}).\n"
        f"**Баланс сценариев:** LONG {_fmt_pct(long_vs_short.get('long',0))} / "
        f"SHORT {_fmt_pct(long_vs_short.get('short',0))}."
    )

def render_beginner_card(st, sc: Scenario, prob: float, atr_val: float):
    rr_to_tp1=_rr(sc.entry,sc.sl,sc.tp1); side="ПОКУПКА (LONG)" if sc.bias=="long" else "ПРОДАЖА (SHORT)"
    atr_note=f"~{_fmt_price(atr_val)} по цене (ATR)."
    st.markdown(f"### Что сделать сейчас — {side}")
    st.markdown(
        "- **Почему:** " + sc.explain + "\n"
        f"- **Вход:** {_fmt_price(sc.entry)}  \n"
        f"- **Стоп:** {_fmt_price(sc.sl)} (риск ≈ {_fmt_price(abs(sc.entry-sc.sl))}, {atr_note})  \n"
        f"- **Цели:** TP1 {_fmt_price(sc.tp1)} (R:R≈{rr_to_tp1}), "
        + (f"TP2 {_fmt_price(sc.tp2)}  " if sc.tp2 else "без TP2  ")
        + f"• **вероятность:** {_fmt_pct(prob)}\n"
        "- **Не входить, если:** импульс ушёл далеко и стоп становится чрезмерным, "
          "или цена закрепилась за уровнем, который должен был удерживаться."
    )

    # --- Быстрый экспорт в TradingView ---
pine_code = pine_for_scenario(s, tf, main_sc)
st.markdown("**Экспорт в TradingView:**")
st.code(pine_code, language="pine")  # у блока есть кнопка "Copy" справа

st.download_button(
    label="⬇️ Скачать .pine",
    data=pine_code.encode("utf-8"),
    file_name=f"{s}_{tf}_{main_sc.name.replace(' ','_')}.pine",
    mime="text/plain",
    use_container_width=False
)

st.link_button("📈 Открыть график TradingView", tv_chart_url(s, tf))
st.caption("Открой ссылку, вставь код в Pine Editor (New > Paste > Save), затем 'Add to chart'.")


# ========= Data via yfinance (фикс многомерных колонок) =========
@st.cache_data(show_spinner=False, ttl=50)
def binance_klines(symbol: str, interval: str, limit: int = 800) -> pd.DataFrame:
    yf_symbol = TICKER_MAP.get(symbol, symbol)
    yf_interval, yf_period = YF_INTERVAL.get(interval, ("60m", "365d"))

    df = yf.download(yf_symbol, interval=yf_interval, period=yf_period,
                     auto_adjust=False, progress=False)
    if df.empty: raise RuntimeError(f"yfinance: пустые данные для {yf_symbol} @ {yf_interval}/{yf_period}")
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)

    df = df.rename(columns={"Open":"open","High":"high","Low":"low","Close":"close","Adj Close":"adj_close","Volume":"volume"})
    need=["open","high","low","close","volume"]
    for c in need:
        if c not in df.columns:
            df[c] = 0.0 if c=="volume" else np.nan
        if isinstance(df[c], pd.DataFrame): df[c]=df[c].iloc[:,0]
        df[c]=pd.to_numeric(df[c], errors="coerce")
    try:
        if df.index.tz is None: df.index=pd.to_datetime(df.index, utc=True)
        else: df.index=df.index.tz_convert("UTC")
    except Exception:
        df.index=pd.to_datetime(df.index, utc=True)

    out=df[need].dropna().tail(limit)
    if out.empty: raise RuntimeError("После очистки нет данных.")
    return out

# ========= Guide =========
def show_guide():
    st.markdown("## Справочник: термины, модели, как пользоваться")
    st.markdown(
        "- **ATR** — средний истинный диапазон (стопы/подтверждения/фильтры).\n"
        "- **Фракталы** — локальные экстремумы; по ним уровни и BOS.\n"
        "- **BOS** — break of structure (подтверждение: закрытие ≥ 0.30×ATR за уровнем).\n"
        "- **FVG** — разрыв ликвидности; играем откат в середину по тренду.\n"
        "- **OB** — импульсная свеча противоположного цвета; ретест даёт вход.\n"
        "- **Breaker** — свип фрактала и возврат под/над уровень.\n"
        "- **Value Area/POC** — объёмная область и её центр; в боковике mean-reversion.\n"
        "- **VWAP** — средневзвешенная по объёму; полезен как «баланс» цены.\n"
        "- **SFP** — прокол и возврат (часто реверс).\n"
        "- **EV** — ожидаемая ценность: сравнение сценариев по матожиданию.\n"
    )
    st.markdown("### Как пользоваться")
    st.markdown(
        "1) Выбери **TF** (по умолчанию 5m) и автообновление.\n"
        "2) В **Фильтрах** отсеки слишком мелкие стопы и слабые цели.\n"
        "3) Включи **Простой режим**, если нужна короткая конкретика.\n"
        "4) Сверяй таблицу сценариев и блок EV, чтобы выбрать лучший вариант.\n"
        "5) Уважай режим: в боковике — от краёв к POC, в тренде — продолжение."
    )
    st.info("Вероятности — относительные (softmax): инструмент ранжирования идей, а не гарантия результата.")

# ========= App =========
st.set_page_config(page_title="SMC Intraday (text)", layout="wide")
st.title("SMC Intraday — BTCUSDT / ETHUSDT (text)")

colA, colB, colE = st.columns([1,1,1])
with colA: tf = st.selectbox("TF", ["5m","15m","1h"], index=0)
with colB: it = st.selectbox("Auto-refresh", ["30s","1m","2m","5m"], index=3)
with colE: beginner_mode = st.checkbox("Простой режим (для новичка)", value=True)

with st.expander("⚙️ Фильтры (риск/цель)"):
    min_risk_pct = st.slider("Мин. риск (%ATR)", 5, 60, 25, step=5)
    min_tp1_atr = st.slider("Мин. TP1 (×ATR)", 1.0, 3.0, 1.5, step=0.25)

st.caption("Фильтры скрывают сетапы с риском ниже %ATR и с TP1 меньше заданного множителя ATR. TP1=2R, TP2=3R. Вероятности нормированы (<100%).")
with st.expander("📘 Справочник (нажми, чтобы открыть)"):
    show_guide()

# авто-refresh
st.markdown(f"<meta http-equiv='refresh' content='{ {'30s':30,'1m':60,'2m':120,'5m':300}[it] }'>", unsafe_allow_html=True)

summary=[]
for s in SYMBOLS:
    try:
        df   = binance_klines(s, BINANCE_INTERVAL[tf], limit=LTF_LIMIT)
        htf  = HTF_MAP[tf]; htf_interval = BINANCE_INTERVAL.get(htf, "1h")
        df_h = binance_klines(s, htf_interval, limit=HTF_LIMIT)
        df_d = binance_klines(s, "1h", limit=24*200)  # эмуляция дневки из 1h

        price=float(df["close"].iloc[-1])
        vp    = volume_profile(df)
        reg   = market_regime(df, vp)
        atr_v = float(atr(df).iloc[-1])

        # поток объёма (наклон OBV для вероятностей)
        o=obv(df); wnd=min(len(o),160)
        slope = (np.polyfit(np.arange(wnd), o.tail(wnd), 1)[0] if wnd>=20 else 0.0)

        htf_bias = score_bias(df_h); d_bias = regime_daily(df_d)

        scenarios_all = propose(df, htf_bias, d_bias, reg, vp, slope)

        # фильтры
        min_risk = atr_v * (min_risk_pct/100.0)
        scenarios=[]
        for sc in scenarios_all:
            if sc.name.startswith("Wait"):
                scenarios.append(sc); continue
            risk_ok = abs(sc.entry-sc.sl) >= min_risk
            tp1_ok  = (abs(sc.tp1-sc.entry)/max(atr_v,1e-6)) >= min_tp1_atr
            if risk_ok and tp1_ok: scenarios.append(sc)
        if not [x for x in scenarios if not x.name.startswith("Wait")]:
            scenarios=scenarios_all

        sc_probs, bias_summary = scenario_probabilities(scenarios, htf_bias, d_bias, slope, price, vp, atr_v, reg)

        st.markdown(f"## {s} ({tf}) — цена: {_fmt_price(price)}")

        top_pair = list(sc_probs.items())[0] if sc_probs else ("Wait (no-trade)", 100.0)
        top_name, top_prob = top_pair[0], top_pair[1]

        if beginner_mode:
            st.markdown(
                make_beginner_summary(
                    s, tf, price, reg, score_bias(df), score_bias(df_h), regime_daily(df_d),
                    vp, top_name, top_prob, bias_summary
                )
            )
        else:
            st.markdown(readable_analysis(s, tf, df, df_h, df_d, scenarios, vp, sc_probs, bias_summary, reg))

        # Главная карточка
        if len(scenarios)==1 and scenarios[0].name.startswith("Wait"):
            st.info("Нет понятного входа: лучше подождать свипа/ретеста ключевых уровней.")
        else:
            main_sc=None; main_prob=0.0
            for sc in scenarios:
                key=f"{sc.name} ({sc.bias})"
                if key==top_name: main_sc=sc; main_prob=sc_probs.get(key,0.0); break
            if main_sc is None:
                main_sc=[x for x in scenarios if not x.name.startswith("Wait")][0]
                main_prob=sc_probs.get(f"{main_sc.name} ({main_sc.bias})",0.0)
            render_beginner_card(st, main_sc, main_prob, atr_v)

        # Таблица сценариев
        rows=[]
        for sc in scenarios:
            if sc.name.startswith("Wait"): continue
            key=f"{sc.name} ({sc.bias})"
            rows.append({
                "Сценарий": key,
                "Тип": sc.etype,
                "Вход": _fmt_price(sc.entry),
                "Стоп": _fmt_price(sc.sl),
                "TP1": _fmt_price(sc.tp1),
                "TP2": _fmt_price(sc.tp2) if sc.tp2 else "—",
                "R:R до TP1": _rr(sc.entry, sc.sl, sc.tp1),
                "Prob%": round(sc_probs.get(key,0.0), 2),
            })
        if rows:
            st.markdown("### Все варианты (кратко)")
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # EV
        with st.expander("Показать математику (EV)"):
            ev_rows=[]
            for sc in scenarios:
                if sc.name.startswith("Wait"): continue
                key=f"{sc.name} ({sc.bias})"; p=sc_probs.get(key,0.0)/100.0
                ev=scenario_ev(sc.entry, sc.sl, sc.tp1, p)
                ev_rows.append({"Сценарий": key, "Prob%": round(p*100,2), "EV": round(ev,6)})
            if ev_rows:
                st.dataframe(pd.DataFrame(ev_rows), use_container_width=True, hide_index=True)

        top_line = max(sc_probs, key=sc_probs.get) if sc_probs else "Wait (no-trade)"
        summary.append(f"{s} {tf} → режим {reg}; HTF {htf} bias {htf_bias}; Top: {top_line}")
        st.divider()
    except Exception as e:
        st.error(f"{s}: {e}")

st.subheader("Зведення")
for line in summary:
    st.write("• ", line)
