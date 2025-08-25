# smc_dashboard.py
"""
SMC Intraday — text-only dashboard (yfinance)
Анализ одного выбранного актива (BTC, ETH, XAUUSD, XAUEUR) с экспортом в TradingView.
"""

from __future__ import annotations

import base64
import urllib.parse
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

# ========= Config =========
# Ключи инструментов внутри дашборда:
SYMBOLS = ["BTCUSDT", "ETHUSDT", "XAUUSD", "XAUEUR"]

# Соответствие для yfinance
# BTC-USD, ETH-USD — спот; золото: XAUUSD=X (spot), XAUEUR=X (spot к EUR)
TICKER_MAP = {
    "BTCUSDT": "BTC-USD",
    "ETHUSDT": "ETH-USD",
    "XAUUSD":  "XAUUSD=X",
    "XAUEUR":  "XAUEUR=X",
}

# Соответствие интервалов для yfinance (interval, period)
YF_INTERVAL = {
    "5m":  ("5m",  "60d"),
    "15m": ("15m", "60d"),
    "1h":  ("60m", "730d"),  # ~2 года
    "1d":  ("1d",  "10y"),
}

# HTF для обсуждения контекста
HTF_MAP = {"5m": "15m", "15m": "1h", "1h": "4h"}  # 4h эмулируем через 60m
LTF_LIMIT = 600
HTF_LIMIT = 400
DAILY_LIMIT = 240

# ========= TradingView helpers =========
def _tv_symbol(symbol: str) -> str:
    # Подберите мэппинг под свой провайдер на TV
    # Крипта: BINANCE, золото: OANDA/FX_IDC
    m = {
        "BTCUSDT": "BINANCE:BTCUSDT",
        "ETHUSDT": "BINANCE:ETHUSDT",
        "XAUUSD":  "OANDA:XAUUSD",   # альтернатива: "FOREXCOM:XAUUSD"
        "XAUEUR":  "FX_IDC:XAUEUR",
    }
    return m.get(symbol, symbol)

def _tv_interval(tf: str) -> str:
    return {"5m": "5", "15m": "15", "1h": "60"}.get(tf, "15")

def tv_chart_url(symbol: str, tf: str) -> str:
    return f"https://www.tradingview.com/chart/?symbol={_tv_symbol(symbol)}&interval={_tv_interval(tf)}"

def pine_for_scenario(symbol: str, tf: str, sc) -> str:
    side = "LONG" if sc.bias == "long" else "SHORT"
    tfn = _tv_interval(tf)
    tp2_val = "na" if sc.tp2 is None else f"{sc.tp2}"
    return f"""//@version=5
indicator("SMC Idea — {symbol} {tf} — {sc.name} ({side})", overlay=true, timeframe="{tfn}", timeframe_gaps=true)

// ---- levels ----
var float entry = {sc.entry}
var float sl    = {sc.sl}
var float tp1   = {sc.tp1}
var float tp2   = {tp2_val}
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

// ---- labels у последней свечи ----
if barstate.islast
    label.new(bar_index, entry, "ENTRY\\n" + str.tostring(entry, format.mintick), style=label.style_label_left, color=col_entry, textcolor=color.white)
    label.new(bar_index, sl,    "SL\\n"    + str.tostring(sl,    format.mintick), style=label.style_label_left, color=col_sl,    textcolor=color.white)
    label.new(bar_index, tp1,   "TP1\\n"   + str.tostring(tp1,   format.mintick), style=label.style_label_left, color=col_tp1,   textcolor=color.white)
    if not na(tp2)
        label.new(bar_index, tp2, "TP2\\n" + str.tostring(tp2, format.mintick),   style=label.style_label_left, color=col_tp2,   textcolor=color.white)

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

def _tv_auto_url(symbol: str, tf: str, sc) -> str:
    payload = base64.b64encode(pine_for_scenario(symbol, tf, sc).encode("utf-8")).decode("ascii")
    return f"{tv_chart_url(symbol, tf)}#smc={urllib.parse.quote(payload)}"

# ========= Indicators / utils =========
def ema(x: pd.Series, n: int): 
    return x.ewm(span=n, adjust=False).mean()

def rsi(x: pd.Series, n=14):
    d = x.diff()
    up = (d.clip(lower=0)).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / (dn.replace(0, np.nan))
    out = 100 - 100 / (1 + rs)
    return out.fillna(method="bfill").fillna(50)

def macd(x: pd.Series):
    f = ema(x, 12); s = ema(x, 26); m = f - s; sig = ema(m, 9)
    return m, sig, m - sig

def atr(df: pd.DataFrame, n=14):
    c = df["close"]
    tr = pd.concat(
        [(df["high"] - df["low"]),
         (df["high"] - c.shift()).abs(),
         (df["low"]  - c.shift()).abs()],
        axis=1
    ).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False).mean()

def obv(df: pd.DataFrame) -> pd.Series:
    sign = np.sign(df["close"].diff().fillna(0.0))
    return (sign * df["volume"]).cumsum()

def adx(df: pd.DataFrame, n: int = 14) -> pd.Series:
    up = df["high"].diff(); dn = -df["low"].diff()
    plus_dm  = np.where((up>dn) & (up>0), up, 0.0)
    minus_dm = np.where((dn>up) & (dn>0), dn, 0.0)
    tr = pd.concat(
        [(df["high"]-df["low"]),
         (df["high"]-df["close"].shift()).abs(),
         (df["low"] -df["close"].shift()).abs()],
        axis=1
    ).max(axis=1)
    atr_s = tr.ewm(alpha=1/n, adjust=False).mean()
    pdi = 100*pd.Series(plus_dm, index=df.index).ewm(alpha=1/n, adjust=False).mean()/atr_s
    mdi = 100*pd.Series(minus_dm, index=df.index).ewm(alpha=1/n, adjust=False).mean()/atr_s
    dx  = 100*(pdi.subtract(mdi).abs() / (pdi+mdi).replace(0, np.nan))
    return dx.ewm(alpha=1/n, adjust=False).mean().fillna(20)

# ========= SMC primitives =========
def swings(df: pd.DataFrame, L=3, R=3):
    hi, lo = df["high"].values, df["low"].values
    n = len(df); SH = np.zeros(n, bool); SL = np.zeros(n, bool)
    for i in range(L, n-R):
        if hi[i] == hi[i-L:i+R+1].max(): SH[i] = True
        if lo[i] == lo[i-L:i+R+1].min(): SL[i] = True
    return pd.Series(SH, df.index), pd.Series(SL, df.index)

def bos(df, SH, SL, look=200, confirm_mult=0.30):
    recent = df.iloc[-look:]
    sh_idx = recent[SH.loc[recent.index]].index
    sl_idx = recent[SL.loc[recent.index]].index
    last_sh = recent.loc[sh_idx[-1]] if len(sh_idx) else None
    last_sl = recent.loc[sl_idx[-1]] if len(sl_idx) else None
    a = float(atr(df).iloc[-1]) or 1e-6
    if last_sh is not None:
        lvl = last_sh.high
        post = recent[recent.index > last_sh.name]
        brk = post[post["close"] > lvl + confirm_mult*a]
        if len(brk): return "up",  brk.index[0], lvl
    if last_sl is not None:
        lvl = last_sl.low
        post = recent[recent.index > last_sl.name]
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
    for i in range(s, n):
        if i-2>=0 and lo[i] > hi[i-2]: out["bull"].append((idx[i], hi[i-2], lo[i]))
        if i-2>=0 and hi[i] < lo[i-2]: out["bear"].append((idx[i], hi[i], lo[i-2]))
    return out

def simple_ob(df, dir_, t, back=70):
    res={"demand":None, "supply":None}
    if dir_ is None or t is None: return res
    before=df[df.index<t].iloc[-back:]
    if dir_=="up":
        reds=before[before["close"]<before["open"]]
        if len(reds):
            last=reds.iloc[-1]
            res["demand"]=(last.name, float(min(last.open,last.close)), float(max(last.open,last.close)))
    if dir_=="down":
        greens=before[before["close"]>before["open"]]
        if len(greens):
            last=greens.iloc[-1]
            res["supply"]=(last.name, float(min(last.open,last.close)), float(max(last.open,last.close)))
    return res

def liquidity_pools(df, SH, SL, win=260):
    rec=df.iloc[-win:]
    bsl=[(t, float(df.loc[t,"high"])) for t in rec[SH.loc[rec.index]].index]
    ssl=[(t, float(df.loc[t,"low"]))  for t in rec[SL.loc[rec.index]].index]
    return {"BSL": bsl, "SSL": ssl}

# ========= Volume Profile (approx) =========
def volume_profile(df: pd.DataFrame, bins: int = 40) -> Dict[str, float | np.ndarray]:
    lo=float(df["low"].min()); hi=float(df["high"].max())
    if hi<=lo: hi = lo+1e-6
    edges=np.linspace(lo, hi, bins+1); vol=np.zeros(bins)
    prices=df["close"].values; vols=df["volume"].values
    idx=np.clip(np.digitize(prices, edges)-1, 0, bins-1)
    for i, v in zip(idx, vols): vol[i]+=v
    total=max(vol.sum(),1.0); poc_i=int(vol.argmax()); poc=(edges[poc_i]+edges[poc_i+1])/2
    area=[poc_i]; L=poc_i-1; R=poc_i+1; acc=vol[poc_i]
    while acc<0.7*total and (L>=0 or R<bins):
        if R>=bins or (L>=0 and vol[L]>=vol[R]): area.append(L); acc+=vol[L]; L-=1
        else: area.append(R); acc+=vol[R]; R+=1
    val=edges[max(min(area),0)]; vah=edges[min(max(area)+1,bins)]
    return {"edges":edges,"volume":vol,"poc":float(poc),"val":float(val),"vah":float(vah)}

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

# ========= RR / EV / text =========
def rr_targets(entry: float, sl: float, bias: str, min_rr: float = 2.0) -> Tuple[float, float, str]:
    risk=abs(entry-sl) or 1e-6
    if bias=="long": tp1=entry+min_rr*risk; tp2=entry+3.0*risk
    else:            tp1=entry-min_rr*risk; tp2=entry-3.0*risk
    return tp1, tp2, f"1:{int(min_rr)}/1:3"

def scenario_ev(entry, sl, tp1, prob):
    risk=abs(entry-sl); reward=abs(tp1-entry); fees=0.0002*(risk+reward)
    return prob*reward - (1-prob)*risk - fees

def model_text(name: str, bias: str, trigger: str, sl_note: str) -> str:
    base = {
        "FVG mitigation": "Откат в середину последнего FVG после подтверждённого BOS (≥0.30×ATR).",
        "OB Retest": "Ретест импульсного Order Block по направлению BOS (граница блока).",
        "BOS Break & Retest": "Пробой swing-уровня + ретест, вход по стоп-ордеру.",
        "Breaker": "Свип ликвидности (фрактал BSL/SSL) и возврат под/над уровень (breaker).",
        "Value Area Reversion": "От края VA к POC внутри боковика (mean-reversion).",
        "EMA Pullback": "Продолжение тренда: откат к EMA20 и возобновление импульса.",
        "Structure Breakout": "Универсальное продолжение: пробой последнего фрактального swing-уровня.",
    }
    side = "LONG" if bias == "long" else "SHORT"
    msg = base.get(name, "Сетап") + "\n"
    msg += "Подтверждение: закрытие свечи за уровнем; микро-дивергенции OBV/RSI приветствуются.\n"
    msg += f"Триггер: {trigger}\n"
    msg += f"SL: {sl_note}\n"
    msg += f"Сторона: {side}."
    return msg

# ========= Scenarios =========
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

TREND_TYPES = ("FVG", "BOS", "OB Retest", "EMA Pullback", "Structure Breakout")
RANGE_TYPES = ("Sweep", "Value Area")

def propose(df: pd.DataFrame, htf_bias: str, d_bias: str, regime: str,
            vp: Dict[str,float|np.ndarray], obv_slope: float) -> List[Scenario]:
    c=float(df["close"].iloc[-1]); at=float(atr(df).iloc[-1]); at=max(at,1e-6)
    SH,SL=swings(df); dir_,t,_=bos(df,SH,SL); gaps=fvg(df)
    pools=liquidity_pools(df,SH,SL); ob=simple_ob(df,dir_,t)
    sh_lvl=SL_lvl=None
    sh_idx=SH[SH].index; sl_idx=SL[SL].index
    sh_lvl=float(df.loc[sh_idx[-1],"high"]) if len(sh_idx) else None
    sl_lvl=float(df.loc[sl_idx[-1],"low"])  if len(sl_idx) else None

    sc: List[Scenario]=[]

    # FVG
    if dir_=="up" and gaps["bull"] and regime=="trend":
        _,lo,hi=list(reversed(gaps["bull"]))[0]; e=(lo+hi)/2; sl=e-1.2*at
        tp1,tp2,rr=rr_targets(e,sl,"long")
        sc.append(Scenario("FVG mitigation","long","limit",f"откат к mid FVG {e:.2f}",e,sl,tp1,tp2,rr,
                           model_text("FVG mitigation","long",f"касание {e:.2f}","под FVG −1.2 ATR")))
    if dir_=="down" and gaps["bear"] and regime=="trend":
        _,lo,hi=list(reversed(gaps["bear"]))[0]; e=(lo+hi)/2; sl=e+1.2*at
        tp1,tp2,rr=rr_targets(e,sl,"short")
        sc.append(Scenario("FVG mitigation","short","limit",f"откат к mid FVG {e:.2f}",e,sl,tp1,tp2,rr,
                           model_text("FVG mitigation","short",f"касание {e:.2f}","над FVG +1.2 ATR")))

    # OB Retest
    if dir_=="up" and ob.get("demand") and regime=="trend":
        _,lo,hi=ob["demand"]; e=hi; sl=lo-0.6*at; tp1,tp2,rr=rr_targets(e,sl,"long")
        sc.append(Scenario("OB Retest","long","limit",f"retest demand OB {lo:.2f}-{hi:.2f}",e,sl,tp1,tp2,rr,
                           model_text("OB Retest","long",f"удержание {hi:.2f}","за низ OB −0.6 ATR")))
    if dir_=="down" and ob.get("supply") and regime=="trend":
        _,lo,hi=ob["supply"]; e=lo; sl=hi+0.6*at; tp1,tp2,rr=rr_targets(e,sl,"short")
        sc.append(Scenario("OB Retest","short","limit",f"retest supply OB {lo:.2f}-{hi:.2f}",e,sl,tp1,tp2,rr,
                           model_text("OB Retest","short",f"удержание {lo:.2f}","за верх OB +0.6 ATR")))

    # BOS Break & Retest
    if dir_=="up" and t is not None:
        lvl=float(df.loc[t,"high"]); e=lvl+0.2*at; sl=lvl-0.8*at
        tp1,tp2,rr=rr_targets(e,sl,"long")
        sc.append(Scenario("BOS Break & Retest","long","stop",f"пробой {lvl:.2f} + ретест",e,sl,tp1,tp2,rr,
                           model_text("BOS Break & Retest","long",f"breakout {lvl:.2f}","за уровень −0.8 ATR")))
    if dir_=="down" and t is not None:
        lvl=float(df.loc[t,"low"]); e=lvl-0.2*at; sl=lvl+0.8*at
        tp1,tp2,rr=rr_targets(e,sl,"short")
        sc.append(Scenario("BOS Break & Retest","short","stop",f"пробой {lvl:.2f} + ретест",e,sl,tp1,tp2,rr,
                           model_text("BOS Break & Retest","short",f"breakdown {lvl:.2f}","за уровень +0.8 ATR")))

    # Breaker
    sw=sweeps(df,SH,SL)
    if sw["high"] and dir_=="down":
        _,lvl_s=sw["high"][-1]; e=lvl_s-0.1*at; sl=lvl_s+0.7*at
        tp1,tp2,rr=rr_targets(e,sl,"short")
        sc.append(Scenario("Breaker","short","stop",f"после срыва BSL {lvl_s:.2f} и BOS↓",e,sl,tp1,tp2,rr,
                           model_text("Breaker","short",f"возврат под {lvl_s:.2f}","над уровень +0.7 ATR")))
    if sw["low"] and dir_=="up":
        _,lvl_s=sw["low"][-1]; e=lvl_s+0.1*at; sl=lvl_s-0.7*at
        tp1,tp2,rr=rr_targets(e,sl,"long")
        sc.append(Scenario("Breaker","long","stop",f"после срыва SSL {lvl_s:.2f} и BOS↑",e,sl,tp1,tp2,rr,
                           model_text("Breaker","long",f"возврат над {lvl_s:.2f}","под уровень −0.7 ATR")))

    # Range: VA -> POC
    vp_width = vp["vah"] - vp["val"]
    if regime=="range":
        if abs(c - vp["val"]) <= max(0.6*at, 0.1*vp_width):
            e=vp["val"]+0.1*at; sl=vp["val"]-0.8*at; tp1,tp2,rr=rr_targets(e,sl,"long")
            sc.append(Scenario("Value Area Reversion","long","limit","от VAL к POC",e,sl,tp1,tp2,rr,
                               model_text("Value Area Reversion","long","отбой от VAL","под VAL −0.8 ATR")))
        if abs(c - vp["vah"]) <= max(0.6*at, 0.1*vp_width):
            e=vp["vah"]-0.1*at; sl=vp["vah"]+0.8*at; tp1,tp2,rr=rr_targets(e,sl,"short")
            sc.append(Scenario("Value Area Reversion","short","limit","от VAH к POC",e,sl,tp1,tp2,rr,
                               model_text("Value Area Reversion","short","отбой от VAH","над VAH +0.8 ATR")))

    # EMA pullback
    ema20=float(ema(df["close"],20).iloc[-1])
    if regime=="trend" and c>ema20:
        e=ema20; sl=e-1.2*at; tp1,tp2,rr=rr_targets(e,sl,"long")
        sc.append(Scenario("EMA Pullback","long","limit",f"откат к EMA20 {e:.2f}",e,sl,tp1,tp2,rr,
                           model_text("EMA Pullback","long",f"касание EMA20 {e:.2f}","ниже EMA −1.2 ATR")))
    if regime=="trend" and c<ema20:
        e=ema20; sl=e+1.2*at; tp1,tp2,rr=rr_targets(e,sl,"short")
        sc.append(Scenario("EMA Pullback","short","limit",f"откат к EMA20 {e:.2f}",e,sl,tp1,tp2,rr,
                           model_text("EMA Pullback","short",f"касание EMA20 {e:.2f}","выше EMA +1.2 ATR")))

    # Structure Breakout
    if sh_lvl is not None:
        base_sl=(sl_lvl if sl_lvl is not None else c-1.2*at)
        e=sh_lvl+0.2*at; sl=base_sl-0.4*at; tp1,tp2,rr=rr_targets(e,sl,"long")
        sc.append(Scenario("Structure Breakout","long","stop",f"пробой swing-high {sh_lvl:.2f}",e,sl,tp1,tp2,rr,
                           model_text("Structure Breakout","long",f"breakout {sh_lvl:.2f}","за ближайший swing-low −0.4 ATR")))
    if sl_lvl is not None:
        base_sl=(sh_lvl if sh_lvl is not None else c+1.2*at)
        e=sl_lvl-0.2*at; sl=base_sl+0.4*at; tp1,tp2,rr=rr_targets(e,sl,"short")
        sc.append(Scenario("Structure Breakout","short","stop",f"пробой swing-low {sl_lvl:.2f}",e,sl,tp1,tp2,rr,
                           model_text("Structure Breakout","short",f"breakdown {sl_lvl:.2f}","за ближайший swing-high +0.4 ATR")))

    # Сортировка/уникальность
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
        if len(uniq)>=6: break

    if not uniq:
        c=float(df["close"].iloc[-1])
        uniq.append(Scenario("Wait (no-trade)","none","—","нет валидных сетапов",c,c,c,c,"—",
                             "Пауза: дождаться свипа/ретеста ключевых уровней."))
    return uniq

# ========= Probabilities =========
def scenario_probabilities(
    scen: List[Scenario],
    htf_bias: str,
    d_bias: str,
    obv_slope: float,
    price: float,
    vp: Dict[str, float | np.ndarray],
    atr_val: float,
    regime: str,
    *,
    cap: float = 0.92,
    floor: float = 0.05,
    temp: float = 1.3,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    if not scen: return {"Wait (no-trade)": 100.0}, {"long": 0.0, "short": 0.0}
    scores=[]; labels=[]
    for s in scen:
        if s.name.startswith("Wait"): continue
        sc=0.0
        if s.bias==htf_bias: sc+=2.0
        if s.bias==d_bias:  sc+=1.2
        sc += 0.9 if (obv_slope>0 and s.bias=="long") or (obv_slope<0 and s.bias=="short") else -0.3
        if (s.name.startswith(("FVG","BOS","OB","EMA","Structure")) and regime=="trend") or \
           (s.name.startswith(("Value Area","Breaker")) and regime=="range"):
            sc+=1.0
        if regime=="range" and s.name.startswith("Structure"): sc-=0.8
        dist=abs(s.entry-price)/max(atr_val,1e-6)
        if dist>2.0: sc-=1.0
        elif dist>1.5: sc-=0.5
        above_poc = price>vp["poc"]
        if s.name.startswith(("FVG","BOS","OB","EMA","Structure")):
            sc += 0.4 if (above_poc and s.bias=="long") or ((not above_poc) and s.bias=="short") else 0.0
        if s.name.startswith("Value Area"):
            near_edge = min(abs(price-vp["val"]), abs(price-vp["vah"])) <= 0.8*atr_val
            sc += 0.5 if near_edge else -0.2
        scores.append(sc); labels.append((s.name, s.bias))
    if not scores: return {"Wait (no-trade)": 100.0}, {"long": 0.0, "short": 0.0}
    scores=np.array(scores,dtype=float)/temp
    ex=np.exp(scores - scores.max()); probs=ex/ex.sum()
    probs=np.clip(probs,floor,cap); probs=probs/probs.sum()
    out={}; agg={"long":0.0,"short":0.0}
    for (lbl,bias),p in zip(labels,probs):
        val=float(np.round(p*100.0,2)); out[f"{lbl} ({bias})"]=val; agg[bias]+=val
    out=dict(sorted(out.items(), key=lambda x:x[1], reverse=True))
    return out, {k:round(v,2) for k,v in agg.items()}

# ========= Readable helpers =========
def _fmt_price(x: float) -> str: return f"{x:,.2f}".replace(","," ")
def _fmt_pct(x: float)   -> str: return f"{x:.1f}%"
def _rr(entry: float, sl: float, tp1: float) -> float:
    risk=abs(entry-sl) or 1e-6; reward=abs(tp1-entry); return round(reward/risk,2)

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
        f"**Контекст простыми словами:** сейчас {regime_txt}; на младшем ТФ — {ltf_bias.upper()}, "
        f"на старшем — {htf_bias.upper()}, по дневному — {d_bias.upper()}.\n"
        f"Цена {where_price}.\n\n"
        f"**Главный план:** {top_name} (вероятность ≈ {_fmt_pct(top_prob)}).\n"
        f"**Баланс сценариев:** LONG {_fmt_pct(long_vs_short.get('long',0))} / SHORT {_fmt_pct(long_vs_short.get('short',0))}."
    )

def render_beginner_card(st, sc: Scenario, prob: float, atr_val: float, symbol: str, tf: str):
    rr_to_tp1=_rr(sc.entry,sc.sl,sc.tp1); side="ПОКУПКА (LONG)" if sc.bias=="long" else "ПРОДАЖА (SHORT)"
    atr_note=f"~{_fmt_price(atr_val)} по цене (ATR)."
    st.markdown(f"### Что сделать сейчас — {side}")
    st.markdown(
        "- **Почему этот вариант:** " + sc.explain.split("\n")[0] + "\n"
        f"- **Где вход:** {_fmt_price(sc.entry)}  \n"
        f"- **Где стоп:** {_fmt_price(sc.sl)} (риск ≈ {_fmt_price(abs(sc.entry-sc.sl))}, {atr_note})  \n"
        f"- **Цели:** TP1 {_fmt_price(sc.tp1)} (R:R~{rr_to_tp1}), "
        + (f"TP2 {_fmt_price(sc.tp2)}  " if sc.tp2 else "без TP2  ")
        + f"• **вероятность:** {_fmt_pct(prob)}\n"
        "- **Подтверждение:** закрытие свечи за уровнем / возврат после свипа.  \n"
        "- **Когда НЕ входить:** если импульс утащил цену далеко от входа и стоп стал слишком большим."
    )
    # Экспорт в TradingView
    pine_code = pine_for_scenario(symbol, tf, sc)
    st.markdown("**Экспорт в TradingView:**")
    st.code(pine_code, language="pine")
    st.download_button(
        label="⬇️ Скачать .pine",
        data=pine_code.encode("utf-8"),
        file_name=f"{symbol}_{tf}_{sc.name.replace(' ','_')}.pine",
        mime="text/plain",
    )
    st.link_button("📈 Открыть график TradingView", tv_chart_url(symbol, tf))
    st.link_button("⚡ Нанести на TradingView автоматически", _tv_auto_url(symbol, tf, sc))
    st.caption("Для автонанесения нужен Tampermonkey userscript. Иначе — Copy/Download.")

# ========= Data via yfinance =========
@st.cache_data(show_spinner=False, ttl=50)
def yf_klines(symbol_key: str, interval_key: str, limit: int) -> pd.DataFrame:
    yf_symbol = TICKER_MAP.get(symbol_key, symbol_key)
    yf_interval, yf_period = YF_INTERVAL.get(interval_key, ("60m", "365d"))

    df = yf.download(yf_symbol, interval=yf_interval, period=yf_period,
                     auto_adjust=False, progress=False)
    if df.empty:
        raise RuntimeError(f"yfinance: пустые данные для {yf_symbol} @ {yf_interval}/{yf_period}")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.rename(columns={
        "Open":"open","High":"high","Low":"low","Close":"close","Adj Close":"adj_close","Volume":"volume"
    })

    need = ["open","high","low","close","volume"]
    for c in need:
        if c not in df.columns:
            df[c] = 0.0 if c=="volume" else np.nan
        df[c] = pd.to_numeric(df[c], errors="coerce")

    try:
        if df.index.tz is None:
            df.index = pd.to_datetime(df.index, utc=True)
        else:
            df.index = df.index.tz_convert("UTC")
    except Exception:
        df.index = pd.to_datetime(df.index, utc=True)

    out = df[need].dropna().tail(limit)
    if out.empty:
        raise RuntimeError("После очистки нет данных.")
    return out

# ========= Guide =========
def show_guide():
    st.markdown("## Справочник")
    st.markdown(
        "- **ATR** — для стопов/подтверждений.\n"
        "- **Фракталы** → уровни и BOS.\n"
        "- **BOS** — закрытие ≥ 0.30×ATR за уровнем.\n"
        "- **FVG** — дисбаланс; играем откат к mid-FVG по тренду.\n"
        "- **OB / Breaker / SFP** — входы от ретестов/свипов.\n"
        "- **Value Area / POC** — mean-reversion в боковике.\n"
        "- **TP1=2R, TP2=3R**; вероятности — относительные (softmax)."
    )

# ========= App =========
st.set_page_config(page_title="SMC Intraday (text)", layout="wide")
st.title("SMC Intraday — BTC / ETH / XAUUSD / XAUEUR (text)")

top1, top2, top3 = st.columns([1,1,1])
with top1:
    symbol = st.selectbox("Актив", SYMBOLS, index=0)
with top2:
    tf = st.selectbox("TF", ["5m","15m","1h"], index=0)
with top3:
    it = st.selectbox("Auto-refresh", ["30s","1m","2m","5m"], index=3)

colC, colD, colE = st.columns([1,1,1])
with colC:
    min_risk_pct = st.slider("Мин. риск (%ATR)", 5, 60, 25, step=5)
with colD:
    min_tp1_atr = st.slider("Мин. TP1 (×ATR)", 1.0, 3.0, 1.5, step=0.25)
with colE:
    beginner_mode = st.checkbox("Простой режим (для новичка)", value=True)

# авто-refresh в браузере
st.markdown(f"<meta http-equiv='refresh' content='{ {'30s':30,'1m':60,'2m':120,'5m':300}[it] }'>", unsafe_allow_html=True)
with st.expander("📘 Справочник (нажми, чтобы открыть)"):
    show_guide()

try:
    df   = yf_klines(symbol, tf, LTF_LIMIT)
    # HTF: используем 1h как суррогат для 4h
    htf  = HTF_MAP[tf]; htf_interval = {"4h":"1h"}.get(htf, htf)
    df_h = yf_klines(symbol, {"15m":"15m","1h":"60m"}.get(htf_interval, "60m"), HTF_LIMIT)
    df_d = yf_klines(symbol, "1d", DAILY_LIMIT)

    price=float(df["close"].iloc[-1])
    vp    = volume_profile(df)
    reg   = market_regime(df, vp)
    atr_v = float(atr(df).iloc[-1])

    # OBV slope
    o=df["close"].pipe(lambda s: obv(pd.DataFrame({"close": s, "volume": df["volume"]})))  # защитный способ
    wnd=min(len(o),160); slope = (np.polyfit(np.arange(wnd), o.tail(wnd), 1)[0] if wnd>=20 else 0.0)

    htf_bias = score_bias(df_h); d_bias = regime_daily(df_d)
    scenarios_all = propose(df, htf_bias, d_bias, reg, vp, slope)

    # Фильтры
    min_risk = atr_v * (min_risk_pct/100.0)
    scenarios=[]
    for sc in scenarios_all:
        if sc.name.startswith("Wait"):
            scenarios.append(sc); continue
        risk_ok = abs(sc.entry-sc.sl) >= min_risk
        tp1_ok  = (abs(sc.tp1-sc.entry)/max(atr_v,1e-6)) >= min_tp1_atr
        if risk_ok and tp1_ok: scenarios.append(sc)
    if not [x for x in scenarios if not x.name.startswith("Wait")]:
        scenarios = scenarios_all

    sc_probs, bias_summary = scenario_probabilities(
        scenarios, htf_bias, d_bias, slope, price, vp, atr_v, reg
    )

    st.markdown(f"## {symbol} ({tf}) — цена: {_fmt_price(price)}")

    ltf_b = score_bias(df); htf_b = score_bias(df_h); d_b = regime_daily(df_d)
    top_pair = list(sc_probs.items())[0] if sc_probs else ("Wait (no-trade)", 100.0)
    top_name, top_prob = top_pair[0], top_pair[1]

    if beginner_mode:
        st.markdown(
            make_beginner_summary(
                symbol, tf, price, reg, ltf_b, htf_b, d_b, vp, top_name, top_prob, bias_summary
            )
        )
    else:
        # можно добавить детальный текстовый отчёт при желании
        pass

    # Главная карточка
    if len(scenarios)==1 and scenarios[0].name.startswith("Wait"):
        st.info("Нет понятного входа: лучше подождать свипа/ретеста ключевых уровней.")
    else:
        main_sc=None; main_prob=0.0
        for sc in scenarios:
            key=f"{sc.name} ({sc.bias})"
            if key==top_name:
                main_sc=sc; main_prob=sc_probs.get(key,0.0); break
        if main_sc is None:
            main_sc=[x for x in scenarios if not x.name.startswith("Wait")][0]
            main_prob=sc_probs.get(f"{main_sc.name} ({main_sc.bias})",0.0)
        render_beginner_card(st, main_sc, main_prob, atr_v, symbol, tf)

    # Таблица
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
        st.markdown("### Варианты (кратко)")
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

except Exception as e:
    st.error(f"{symbol}: {e}")
