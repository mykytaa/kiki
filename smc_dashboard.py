# smc_dashboard.py
# -*- coding: utf-8 -*-

"""
SMC Intraday — BTC / ETH / XAUUSD / XAUEUR / EURUSD (text)
Надёжные SMC-сигналы с подтверждениями (SFP→BOS→FVG/OB), конфлюенсами (EMA/VWAP/POC),
аргументированными SL/TP и подробными карточками сценариев. Источник данных — yfinance.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf


# ==============================
#        Конфигурация
# ==============================

ASSETS = ["BTCUSDT", "ETHUSDT", "XAUUSD", "XAUEUR", "EURUSD"]

YF_TICKER_CANDIDATES: Dict[str, List[str]] = {
    "BTCUSDT": ["BTC-USD"],
    "ETHUSDT": ["ETH-USD"],
    "XAUUSD":  ["XAUUSD=X", "GC=F"],
    "XAUEUR":  ["XAUEUR=X"],
    "EURUSD":  ["EURUSD=X"],
}

TF_FALLBACKS = {
    "5m":  [("5m", "60d"), ("15m", "60d"), ("60m", "730d")],
    "15m": [("15m", "60d"), ("60m", "730d")],
    "1h":  [("60m", "730d"), ("1d", "730d")],
}

HTF_OF = {"5m": "15m", "15m": "60m", "1h": "1d"}


# ==============================
#   Вспомогательные индикаторы
# ==============================

def ema(x: pd.Series, n: int) -> pd.Series:
    return x.ewm(span=n, adjust=False).mean()

def rsi(x: pd.Series, n=14) -> pd.Series:
    d = x.diff()
    up = (d.clip(lower=0)).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / (dn.replace(0, np.nan))
    out = 100 - 100 / (1 + rs)
    return out.fillna(method="bfill").fillna(50)

def macd(x: pd.Series):
    f = ema(x, 12); s = ema(x, 26); m = f - s; sig = ema(m, 9)
    return m, sig, m - sig

def atr(df: pd.DataFrame, n=14) -> pd.Series:
    c = df["close"]
    tr = pd.concat(
        [(df["high"] - df["low"]),
         (df["high"] - c.shift()).abs(),
         (df["low"] - c.shift()).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()

def obv(df: pd.DataFrame) -> pd.Series:
    sign = np.sign(df["close"].diff().fillna(0.0))
    return (sign * df["volume"]).cumsum()

def adx(df: pd.DataFrame, n: int = 14) -> pd.Series:
    up = df["high"].diff(); dn = -df["low"].diff()
    plus_dm  = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = pd.concat(
        [(df["high"] - df["low"]),
         (df["high"] - df["close"].shift()).abs(),
         (df["low"]  - df["close"].shift()).abs()],
        axis=1,
    ).max(axis=1)
    atr_s = tr.ewm(alpha=1 / n, adjust=False).mean()
    pdi = 100 * pd.Series(plus_dm,  index=df.index).ewm(alpha=1 / n, adjust=False).mean() / atr_s
    mdi = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1 / n, adjust=False).mean() / atr_s
    dx = 100 * (pdi.subtract(mdi).abs() / (pdi + mdi).replace(0, np.nan))
    return dx.ewm(alpha=1 / n, adjust=False).mean().fillna(20)

def vwap_series(df: pd.DataFrame) -> pd.Series:
    tp  = (df["high"] + df["low"] + df["close"]) / 3.0
    vol = df["volume"].replace(0, np.nan).fillna(0.0)
    num = (tp * vol).cumsum()
    den = vol.cumsum().replace(0, np.nan)
    return (num / den).fillna(method="bfill").fillna(df["close"])

def slope(series: pd.Series, last_n: int = 80) -> float:
    n = min(len(series), last_n)
    if n < 8: return 0.0
    y = series.tail(n).values
    x = np.arange(n, dtype=float)
    return float(np.polyfit(x, y, 1)[0])


# ==============================
#         SMC элементы
# ==============================

def swings(df: pd.DataFrame, L=3, R=3):
    hi = df["high"].values; lo = df["low"].values
    n = len(df); SH = np.zeros(n, bool); SL = np.zeros(n, bool)
    for i in range(L, n - R):
        if hi[i] == hi[i - L:i + R + 1].max(): SH[i] = True
        if lo[i] == lo[i - L:i + R + 1].min(): SL[i] = True
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
        brk = post[post["close"] > lvl + confirm_mult * a]
        if len(brk): return "up", brk.index[0], lvl
    if last_sl is not None:
        lvl = last_sl.low
        post = recent[recent.index > last_sl.name]
        brk = post[post["close"] < lvl - confirm_mult * a]
        if len(brk): return "down", brk.index[0], lvl
    return None, None, None

def sweeps(df, SH, SL, win=180):
    res = {"high": [], "low": []}; rec = df.iloc[-win:]
    for t in rec[SH.loc[rec.index]].index:
        level = df.loc[t, "high"]; post = rec[rec.index > t]
        if len(post[(post["high"] > level) & (post["close"] < level)]):
            res["high"].append((t, level))
    for t in rec[SL.loc[rec.index]].index:
        level = df.loc[t, "low"]; post = rec[rec.index > t]
        if len(post[(post["low"] < level) & (post["close"] > level)]):
            res["low"].append((t, level))
    return res

def fvg(df, look=140):
    out = {"bull": [], "bear": []}
    hi = df["high"].values; lo = df["low"].values; idx = df.index
    n = len(df); s = max(2, n - look)
    for i in range(s, n):
        if i - 2 >= 0 and lo[i] > hi[i - 2]:
            out["bull"].append((idx[i], hi[i - 2], lo[i]))
        if i - 2 >= 0 and hi[i] < lo[i - 2]:
            out["bear"].append((idx[i], hi[i], lo[i - 2]))
    return out

def simple_ob(df, dir_, t, back=70):
    res = {"demand": None, "supply": None}
    if dir_ is None or t is None: return res
    before = df[df.index < t].iloc[-back:]
    if dir_ == "up":
        reds = before[before["close"] < before["open"]]
        if len(reds):
            last = reds.iloc[-1]
            res["demand"] = (last.name,
                             float(min(last["open"], last["close"])),
                             float(max(last["open"], last["close"])))
    if dir_ == "down":
        greens = before[before["close"] > before["open"]]
        if len(greens):
            last = greens.iloc[-1]
            res["supply"] = (last.name,
                             float(min(last["open"], last["close"])),
                             float(max(last["open"], last["close"])))
    return res

def volume_profile(df: pd.DataFrame, bins: int = 40) -> Dict[str, float | np.ndarray]:
    lo = float(df["low"].min()); hi = float(df["high"].max())
    if hi <= lo: hi = lo + 1e-6
    edges = np.linspace(lo, hi, bins + 1)
    prices = df["close"].values
    vols = df["volume"].values.astype(float)
    if np.nansum(vols) <= 1e-12:
        vols = np.ones_like(vols, dtype=float)
    vol = np.zeros(bins, dtype=float)
    idx = np.clip(np.digitize(prices, edges) - 1, 0, bins - 1)
    for i, v in zip(idx, vols): vol[i] += float(v)
    total = max(vol.sum(), 1.0)
    poc_i = int(vol.argmax()); poc = (edges[poc_i] + edges[poc_i + 1]) / 2
    area = [poc_i]; L = poc_i - 1; R = poc_i + 1; acc = vol[poc_i]
    while acc < 0.7 * total and (L >= 0 or R < bins):
        if R >= bins or (L >= 0 and vol[L] >= vol[R]):
            area.append(L); acc += vol[L]; L -= 1
        else:
            area.append(R); acc += vol[R]; R += 1
    val = edges[max(min(area), 0)]; vah = edges[min(max(area) + 1, bins)]
    return {"edges": edges, "volume": vol, "poc": float(poc), "val": float(val), "vah": float(vah)}

def last_swing_levels(df: pd.DataFrame, SH: pd.Series, SL: pd.Series) -> Tuple[Optional[float], Optional[float]]:
    sh_idx = SH[SH].index; sl_idx = SL[SL].index
    sh_lvl = float(df.loc[sh_idx[-1], "high"]) if len(sh_idx) else None
    sl_lvl = float(df.loc[sl_idx[-1], "low"])  if len(sl_idx) else None
    return sh_lvl, sl_lvl

def nearest_struct_target(df: pd.DataFrame, SH: pd.Series, SL: pd.Series,
                          entry: float, direction: str) -> Optional[float]:
    """Ближайшая цель по свингам в сторону входа."""
    if direction == "long":
        highs = [float(df.loc[i, "high"]) for i in SH[SH].index if float(df.loc[i, "high"]) > entry]
        return min(highs) if highs else None
    else:
        lows = [float(df.loc[i, "low"]) for i in SL[SL].index if float(df.loc[i, "low"]) < entry]
        return max(lows) if lows else None


# ==============================
#   Режимы/байасы/вероятности
# ==============================

def score_bias(df: pd.DataFrame) -> str:
    c = df["close"]; s = 0
    r = float(rsi(c).iloc[-1]); h = float(macd(c)[2].iloc[-1])
    if r > 55: s += 1
    elif r < 45: s -= 1
    if h > 0: s += 1
    elif h < 0: s -= 1
    return "long" if s >= 1 else ("short" if s <= -1 else "none")

def regime_daily(df_d: pd.DataFrame) -> str:
    e50 = ema(df_d["close"], 50).iloc[-1]; e200 = ema(df_d["close"], 200).iloc[-1]
    if np.isnan(e50) or np.isnan(e200): return "none"
    return "long" if e50 > e200 else "short" if e50 < e200 else "none"

def market_regime(df: pd.DataFrame, vp: Dict[str, float | np.ndarray]) -> str:
    ad = float(adx(df).iloc[-1]); price = float(df["close"].iloc[-1])
    outside = (price > vp["vah"]) or (price < vp["val"])
    return "trend" if (ad >= 22 or outside) else "range"


# ==============================
#       Сценарии с подтверждениями
# ==============================

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
    confirms: int
    confirm_list: List[str]
    explain_short: str
    stop_reason: str
    tp_reason: str
    logic_path: List[str]

def rr_targets(entry: float, sl: float, bias: str, min_rr: float = 2.0) -> Tuple[float, float, str]:
    risk = abs(entry - sl) or 1e-9
    if bias == "long":
        tp1 = entry + min_rr * risk; tp2 = entry + 3.0 * risk
    else:
        tp1 = entry - min_rr * risk; tp2 = entry - 3.0 * risk
    return tp1, tp2, f"1:{int(min_rr)}/1:3"

def scenario_ev(entry, sl, tp1, prob):
    risk = abs(entry - sl); reward = abs(tp1 - entry)
    fees = 0.0002 * (risk + reward)
    return prob * reward - (1 - prob) * risk - fees

def fmt_reason_list(items: List[str]) -> str:
    return ", ".join(items) if items else "—"

def propose(df: pd.DataFrame, htf_bias: str, d_bias: str, regime: str,
            vp: Dict[str, float | np.ndarray], obv_slope_val: float) -> List[Scenario]:
    c = float(df["close"].iloc[-1])
    at = float(atr(df).iloc[-1]); at = max(at, 1e-9)
    SH, SL = swings(df); dir_bos, t_bos, lvl_bos = bos(df, SH, SL)
    gaps = fvg(df); swp = sweeps(df, SH, SL); ob = simple_ob(df, dir_bos, t_bos)
    sh_lvl, sl_lvl = last_swing_levels(df, SH, SL)
    vw = float(vwap_series(df).iloc[-1]); ema20_val = float(ema(df["close"], 20).iloc[-1])
    ema_up = slope(ema(df["close"], 20)) > 0
    ema_dn = slope(ema(df["close"], 20)) < 0
    above_poc = c > vp["poc"]

    def generic_confs(bias: str, entry: float) -> List[str]:
        out = []
        if (bias == "long" and obv_slope_val > 0) or (bias == "short" and obv_slope_val < 0):
            out.append("OBV подтверждает")
        if (bias == "long" and ema_up) or (bias == "short" and ema_dn):
            out.append("тренд EMA20")
        if (bias == "long" and above_poc) or (bias == "short" and not above_poc):
            out.append("сторона POC")
        if abs(entry - vw) <= 0.6 * at:
            out.append("рядом VWAP")
        if (bias == htf_bias) or (bias == d_bias):
            out.append("HTF/Daily согласование")
        return out

    def structural_tp(entry: float, bias: str) -> Tuple[float, str]:
        tgt = nearest_struct_target(df, SH, SL, entry, "long" if bias == "long" else "short")
        if tgt is not None:
            return tgt, "ближайший свинг по направлению"
        # запасные варианты — POC/VAH/VAL
        if bias == "long":
            if entry < vp["poc"] <= vp["vah"]: return vp["poc"], "POC"
            return entry + 2.0 * abs(entry - (entry - 1.0 * at)), "≈2R (структурной цели рядом нет)"
        else:
            if entry > vp["poc"] >= vp["val"]: return vp["poc"], "POC"
            return entry - 2.0 * abs(entry - (entry + 1.0 * at)), "≈2R (структурной цели рядом нет)"

    scenarios: List[Scenario] = []

    def add(name: str, bias: str, etype: str, trigger: str,
            entry: float, sl: float, base_confirms: List[str],
            stop_reason: str, tp_hint: Optional[str],
            logic_path: List[str], fvg_touch: bool = False, ob_touch: bool = False):
        confs = base_confirms[:]
        confs += generic_confs(bias, entry)
        if fvg_touch: confs.append("FVG конfluence")
        if ob_touch: confs.append("OB ретест")
        confirms = len(confs)

        # цели — сначала структурная, затем R-бекапы
        tp1_struct, tp1_reason = structural_tp(entry, bias)
        if bias == "long":
            tp1 = min(tp1_struct, entry + 3.0 * abs(entry - sl))
        else:
            tp1 = max(tp1_struct, entry - 3.0 * abs(entry - sl))
        # TP2 — более дальний ориентир: VAH/VAL либо 3R
        if bias == "long":
            tp2 = max(tp1, vp["vah"]) if vp["vah"] > entry else entry + 3.0 * abs(entry - sl)
        else:
            tp2 = min(tp1, vp["val"]) if vp["val"] < entry else entry - 3.0 * abs(entry - sl)

        rr = f"1:2/1:3"
        scenarios.append(
            Scenario(
                name=name, bias=bias, etype=etype, trigger=trigger,
                entry=float(entry), sl=float(sl), tp1=float(tp1), tp2=float(tp2),
                rr=rr, confirms=confirms, confirm_list=confs,
                explain_short=f"{name}: " + (" → ".join(logic_path)),
                stop_reason=stop_reason,
                tp_reason=tp1_reason if tp_hint is None else tp_hint,
                logic_path=logic_path
            )
        )

    # ===== 1) SFP → BOS → FVG (самый «надёжный» паттерн) =====
    if swp["high"] and dir_bos == "down" and gaps["bear"]:
        t_sfp, lvl_sfp = swp["high"][-1]
        _, lo, hi = list(reversed(gaps["bear"]))[0]
        mid = (lo + hi) / 2
        e = mid - 0.05 * at
        sl = lvl_sfp + 0.7 * at
        add(
            name="SFP→BOS→FVG", bias="short", etype="limit",
            trigger=f"касание mid FVG {mid:.5f}", entry=e, sl=sl,
            base_confirms=["SFP high", "BOS↓"],
            stop_reason="за SFP-уровень +0.7×ATR",
            tp_hint="первая цель — ближайший свинг/POC",
            logic_path=["срыв high", "пробой вниз", "имбаланс вниз"],
            fvg_touch=True
        )
    if swp["low"] and dir_bos == "up" and gaps["bull"]:
        t_sfp, lvl_sfp = swp["low"][-1]
        _, lo, hi = list(reversed(gaps["bull"]))[0]
        mid = (lo + hi) / 2
        e = mid + 0.05 * at
        sl = lvl_sfp - 0.7 * at
        add(
            name="SFP→BOS→FVG", bias="long", etype="limit",
            trigger=f"касание mid FVG {mid:.5f}", entry=e, sl=sl,
            base_confirms=["SFP low", "BOS↑"],
            stop_reason="за SFP-уровень −0.7×ATR",
            tp_hint="первая цель — ближайший свинг/POC",
            logic_path=["срыв low", "пробой вверх", "имбаланс вверх"],
            fvg_touch=True
        )

    # ===== 2) BOS → OB-ретест (+FVG) =====
    if dir_bos == "up" and ob.get("demand"):
        _, lo, hi = ob["demand"]
        e = hi; sl = lo - 0.6 * at
        add(
            name="BOS→OB Retest", bias="long", etype="limit",
            trigger=f"retest OB {lo:.5f}-{hi:.5f}", entry=e, sl=sl,
            base_confirms=["BOS↑"], stop_reason="за противоположную грань OB −0.6×ATR",
            tp_hint=None, logic_path=["пробой вверх", "ретест спроса"],
            ob_touch=True, fvg_touch=bool(gaps["bull"])
        )
    if dir_bos == "down" and ob.get("supply"):
        _, lo, hi = ob["supply"]
        e = lo; sl = hi + 0.6 * at
        add(
            name="BOS→OB Retest", bias="short", etype="limit",
            trigger=f"retest OB {lo:.5f}-{hi:.5f}", entry=e, sl=sl,
            base_confirms=["BOS↓"], stop_reason="за противоположную грань OB +0.6×ATR",
            tp_hint=None, logic_path=["пробой вниз", "ретест предложения"],
            ob_touch=True, fvg_touch=bool(gaps["bear"])
        )

    # ===== 3) Breaker после свипа =====
    if swp["high"] and dir_bos == "down":
        _, lvl = swp["high"][-1]
        e = lvl - 0.1 * at; sl = lvl + 0.7 * at
        add("Breaker", "short", "stop", f"возврат под {lvl:.5f}", e, sl,
            base_confirms=["SFP high", "BOS↓"], stop_reason="над свип-уровнем +0.7×ATR",
            tp_hint="TP1 у ближайшего low/POC", logic_path=["срыв high", "возврат/пробой вниз"])
    if swp["low"] and dir_bos == "up":
        _, lvl = swp["low"][-1]
        e = lvl + 0.1 * at; sl = lvl - 0.7 * at
        add("Breaker", "long", "stop", f"возврат над {lvl:.5f}", e, sl,
            base_confirms=["SFP low", "BOS↑"], stop_reason="под свип-уровнем −0.7×ATR",
            tp_hint="TP1 у ближайшего high/POC", logic_path=["срыв low", "возврат/пробой вверх"])

    # ===== 4) Range Reversion (VAL/VAH) с SFP у края =====
    adx_val = float(adx(df).iloc[-1])
    if regime == "range" and adx_val < 22:
        near_val = abs(c - vp["val"]) <= max(0.6 * at, 0.1 * (vp["vah"] - vp["val"]))
        near_vah = abs(c - vp["vah"]) <= max(0.6 * at, 0.1 * (vp["vah"] - vp["val"]))
        if near_val:
            e = vp["val"] + 0.1 * at; sl = vp["val"] - 0.8 * at
            confs = ["VAL edge", "ADX низкий"]
            if swp["low"]: confs.append("SFP low у края")
            add("Value Area Reversion", "long", "limit", "от VAL к POC", e, sl,
                base_confirms=confs, stop_reason="за VAL −0.8×ATR",
                tp_hint="TP1 — POC", logic_path=["боковик", "край VAL", "отбой"])
        if near_vah:
            e = vp["vah"] - 0.1 * at; sl = vp["vah"] + 0.8 * at
            confs = ["VAH edge", "ADX низкий"]
            if swp["high"]: confs.append("SFP high у края")
            add("Value Area Reversion", "short", "limit", "от VAH к POC", e, sl,
                base_confirms=confs, stop_reason="за VAH +0.8×ATR",
                tp_hint="TP1 — POC", logic_path=["боковик", "край VAH", "отбой"])

    # ===== 5) EMA Pullback в тренде (с VWAP/POC конfluence) =====
    if regime == "trend" and c > ema20_val:
        e = ema20_val; sl = e - 1.2 * at
        add("EMA Pullback", "long", "limit", f"к EMA20 {e:.5f}", e, sl,
            base_confirms=["тренд вверх", "EMA20"], stop_reason="ниже EMA20 −1.2×ATR",
            tp_hint=None, logic_path=["тренд", "pullback к EMA/VWAP"], fvg_touch=bool(gaps["bull"]))
    if regime == "trend" and c < ema20_val:
        e = ema20_val; sl = e + 1.2 * at
        add("EMA Pullback", "short", "limit", f"к EMA20 {e:.5f}", e, sl,
            base_confirms=["тренд вниз", "EMA20"], stop_reason="выше EMA20 +1.2×ATR",
            tp_hint=None, logic_path=["тренд", "pullback к EMA/VWAP"], fvg_touch=bool(gaps["bear"]))

    # ===== 6) Структурный пробой + ретест =====
    if sh_lvl is not None:
        base_sl = (sl_lvl if sl_lvl is not None else c - 1.2 * at)
        e = sh_lvl + 0.2 * at; sl = base_sl - 0.4 * at
        add("Structure Breakout", "long", "stop", f"пробой {sh_lvl:.5f}", e, sl,
            base_confirms=["уровень swing↑", "ретест"], stop_reason="за ближайший swing −0.4×ATR",
            tp_hint=None, logic_path=["breakout", "ret"])
    if sl_lvl is not None:
        base_sl = (sh_lvl if sh_lvl is not None else c + 1.2 * at)
        e = sl_lvl - 0.2 * at; sl = base_sl + 0.4 * at
        add("Structure Breakout", "short", "stop", f"пробой {sl_lvl:.5f}", e, sl,
            base_confirms=["уровень swing↓", "ретест"], stop_reason="за ближайший swing +0.4×ATR",
            tp_hint=None, logic_path=["breakdown", "ret"])

    # сортировка: сначала по числу подтверждений, затем по контексту
    def _sort_key(x: Scenario):
        score = -x.confirms
        if (x.name.startswith(("SFP→BOS→FVG", "BOS→OB", "EMA", "Structure")) and regime == "trend") or \
           (x.name.startswith(("Value Area", "Breaker")) and regime == "range"):
            score -= 1
        if x.bias == htf_bias: score -= 0.5
        return score

    uniq, seen = [], set()
    for s in sorted(scenarios, key=_sort_key):
        k = (s.name, s.bias)
        if k in seen: continue
        seen.add(k); uniq.append(s)
        if len(uniq) >= 8: break

    if not uniq:
        c0 = c
        uniq.append(Scenario("Wait (no-trade)", "none", "—", "нет валидных сетапов",
                             c0, c0, c0, c0, "—", 0, [], "Пауза.", "—", "—", []))
    return uniq


# ==============================
#       Вероятности идей
# ==============================

def scenario_probabilities(
    scen: List[Scenario], htf_bias: str, d_bias: str, obv_slope_val: float,
    price: float, vp: Dict[str, float | np.ndarray], atr_val: float, regime: str,
    *, cap: float = 0.90, floor: float = 0.05, temp: float = 1.15
) -> Tuple[Dict[str, float], Dict[str, float]]:
    if not scen: return {"Wait (no-trade)": 100.0}, {"long": 0.0, "short": 0.0}
    scores, labels = [], []
    for s in scen:
        if s.name.startswith("Wait"): continue
        sc = 0.6 * s.confirms  # весим подтверждения сильнее
        if s.bias == htf_bias: sc += 1.8
        if s.bias == d_bias:  sc += 1.0
        sc += 0.8 if (obv_slope_val > 0 and s.bias == "long") or (obv_slope_val < 0 and s.bias == "short") else -0.2
        if (s.name.startswith(("SFP→BOS→FVG","BOS→OB","EMA","Structure")) and regime == "trend") or \
           (s.name.startswith(("Value Area","Breaker")) and regime == "range"): sc += 1.0
        dist = abs(s.entry - price) / max(atr_val, 1e-6)
        if dist > 2.0: sc -= 1.0
        elif dist > 1.5: sc -= 0.5
        scores.append(sc); labels.append((s.name, s.bias))
    if not scores: return {"Wait (no-trade)": 100.0}, {"long": 0.0, "short": 0.0}
    scores = np.array(scores, dtype=float) / temp
    ex = np.exp(scores - scores.max()); probs = ex / ex.sum()
    probs = np.clip(probs, floor, cap); probs = probs / probs.sum()
    out, agg = {}, {"long": 0.0, "short": 0.0}
    for (lbl, bias), p in zip(labels, probs):
        val = float(np.round(p * 100.0, 2))
        out[f"{lbl} ({bias})"] = val; agg[bias] += val
    out = dict(sorted(out.items(), key=lambda x: x[1], reverse=True))
    return out, {k: round(v, 2) for k, v in agg.items()}


# ==============================
#       Данные (yfinance)
# ==============================

@st.cache_data(show_spinner=False, ttl=60)
def yf_ohlc_first_success(asset_key: str, tf: str, limit: int = 800) -> Tuple[pd.DataFrame, str, str]:
    cands = YF_TICKER_CANDIDATES.get(asset_key, [asset_key])
    tries = TF_FALLBACKS.get(tf, TF_FALLBACKS["15m"])
    last_err = None
    for tkr in cands:
        for interval, period in tries:
            try:
                df = yf.download(tkr, interval=interval, period=period,
                                 auto_adjust=False, progress=False)
                if df.empty:
                    last_err = f"{tkr}@{interval}/{period}: пусто"; continue
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                df = df.rename(columns={"Open":"open","High":"high","Low":"low","Close":"close","Adj Close":"adj_close","Volume":"volume"})
                need = ["open","high","low","close","volume"]
                for c in need:
                    if c not in df.columns:
                        df[c] = 0.0 if c == "volume" else np.nan
                    df[c] = pd.to_numeric(df[c], errors="coerce")
                try:
                    if df.index.tz is None: df.index = pd.to_datetime(df.index, utc=True)
                    else: df.index = df.index.tz_convert("UTC")
                except Exception:
                    df.index = pd.to_datetime(df.index, utc=True)
                out = df[need].dropna().tail(limit)
                if out.empty:
                    last_err = f"{tkr}@{interval}/{period}: после очистки нет данных"; continue
                return out, interval, period
            except Exception as e:
                last_err = f"{tkr}@{interval}/{period}: {e}"
                continue
    raise RuntimeError(f"yfinance: не удалось получить данные для {asset_key}. Последняя ошибка: {last_err}")


# ==============================
#          Вспомогалка UI
# ==============================

def infer_decimals(df: pd.DataFrame, asset: str) -> int:
    x = df["close"].tail(300).diff().abs()
    step = float(np.nanmin(x[x > 0])) if np.any(x > 0) else 0.0
    if step > 0:
        p = max(2, min(6, int(np.ceil(-np.log10(step)) + 1)))
    else:
        default = {"EURUSD": 5, "XAUEUR": 5, "XAUUSD": 2, "BTCUSDT": 2, "ETHUSDT": 2}
        p = default.get(asset, 4)
    return p

def fmt_price(x: float, decimals: int) -> str:
    s = f"{x:.{decimals}f}"
    if "." in s:
        a, b = s.split("."); a = f"{int(float(a)):,}".replace(",", " ")
        return f"{a}.{b}"
    return f"{int(float(s)):,}".replace(",", " ")


# ==============================
#             UI
# ==============================

st.set_page_config(page_title="SMC Intraday (text)", layout="wide")
st.title("SMC Intraday — BTC / ETH / XAUUSD / XAUEUR / EURUSD (text)")

colA, colB, colC, colD, colE = st.columns([1.2, 1, 1, 1, 1])
with colA:
    asset = st.selectbox("Актив", ASSETS, index=4 if "EURUSD" in ASSETS else 0)
with colB:
    tf = st.selectbox("TF", ["5m", "15m", "1h"], index=0)
with colC:
    min_risk_pct = st.slider("Мин. риск (%ATR)", 5, 60, 25, step=5)
with colD:
    min_tp1_atr = st.slider("Мин. TP1 (×ATR)", 1.0, 3.0, 1.5, step=0.25)
with colE:
    min_confirms = st.slider("Мин. подтверждений", 2, 7, 4, step=1)

colF, colG = st.columns([1, 1])
with colF:
    refresh_mode = st.selectbox("Обновление", ["Выключено", "Каждые 30s", "1m", "2m", "5m"], index=0)
with colG:
    if st.button("🔄 Обновить сейчас"):
        st.cache_data.clear(); st.toast("Данные обновлены"); st.experimental_rerun()

INTERVALS = {"Каждые 30s": 30, "1m": 60, "2m": 120, "5m": 300}
if "next_refresh_ts" not in st.session_state:
    st.session_state.next_refresh_ts = time.time() + 10**9
if refresh_mode != "Выключено":
    interval = INTERVALS[refresh_mode]; now = time.time()
    if now >= st.session_state.next_refresh_ts:
        st.session_state.next_refresh_ts = now + interval
        st.cache_data.clear(); st.experimental_rerun()
else:
    st.session_state.next_refresh_ts = time.time() + 10**9

beginner_mode = st.checkbox("Простой режим (для новичка)", value=True)

with st.expander("📘 Справочник"):
    st.markdown(
        "- **Иерархия подтверждений:**\n"
        "  - SFP/срыв → BOS → FVG или OB-ретест — базовый корпус (2–3 подтверждения).\n"
        "  - Конфлюенсы: тренд EMA20, сторона к POC, близость VWAP, OBV в сторону, HTF/Daily в сторону.\n"
        "  - В боковике добавляется край VAL/VAH и низкий ADX.\n"
        "- **Стопы** ставятся за структурный уровень + ATR-буфер (0.4–1.2×ATR в зависимости от сетапа).\n"
        "- **TP1** — ближайший свинг/POC/VAL/VAH (если рядом) либо ≈2R; **TP2** — дальняя цель (VAH/VAL или ≈3R)."
    )

st.caption(
    "Идеи с риском ниже порога %ATR и с TP1 меньше заданного множителя ATR скрываются. "
    "TP1=~2R (или структурная цель), TP2≈3R. Вероятности — относительные."
)


# ==============================
#        Основной поток
# ==============================

summary: List[str] = []

try:
    df, tf_eff, _ = yf_ohlc_first_success(asset, tf, limit=800)
    htf = HTF_OF[tf]; df_h, _, _ = yf_ohlc_first_success(asset, htf, limit=400)
    df_d, _, _ = yf_ohlc_first_success(asset, "1d", limit=600)

    price = float(df["close"].iloc[-1])
    vp = volume_profile(df); reg = market_regime(df, vp)
    atr_v = float(atr(df).iloc[-1]); obv_s = slope(obv(df), 160)

    htf_bias = score_bias(df_h); d_bias = regime_daily(df_d)
    scenarios_all = propose(df, htf_bias, d_bias, reg, vp, obv_s)

    # фильтры
    min_risk = atr_v * (min_risk_pct / 100.0)
    scenarios: List[Scenario] = []
    for sc in scenarios_all:
        if sc.name.startswith("Wait"): continue
        if sc.confirms < min_confirms: continue
        risk_ok = abs(sc.entry - sc.sl) >= min_risk
        tp1_ok = (abs(sc.tp1 - sc.entry) / max(atr_v, 1e-6)) >= min_tp1_atr
        if risk_ok and tp1_ok:
            scenarios.append(sc)
    if not scenarios:
        scenarios = [s for s in scenarios_all if not s.name.startswith("Wait")] or scenarios_all

    sc_probs, bias_summary = scenario_probabilities(scenarios, htf_bias, d_bias, obv_s, price, vp, atr_v, reg)

    decimals = infer_decimals(df, asset)
    st.markdown(f"## {asset} ({tf}) — цена: {fmt_price(price, decimals)}")

    ltf_b = score_bias(df); htf_b = score_bias(df_h); d_b = regime_daily(df_d)
    poc_state = "выше VAH" if price > vp["vah"] else ("ниже VAL" if price < vp["val"] else "внутри value area")
    st.markdown(
        f"**Контекст:** LTF={ltf_b.upper()}, HTF={htf_b.upper()}, Daily={d_b.upper()} • "
        f"Режим: {reg.upper()} (ADX≈{float(adx(df).iloc[-1]):.1f}) • "
        f"POC {fmt_price(vp['poc'], decimals)}, VAL {fmt_price(vp['val'], decimals)}, VAH {fmt_price(vp['vah'], decimals)} → цена {poc_state}.  \n"
        f"**Баланс:** LONG {bias_summary['long']:.1f}% vs SHORT {bias_summary['short']:.1f}%"
    )

    # главная карточка
    if scenarios:
        top_key = list(sc_probs.keys())[0] if sc_probs else f"{scenarios[0].name} ({scenarios[0].bias})"
        main_sc = next((s for s in scenarios if f"{s.name} ({s.bias})" == top_key), scenarios[0])
        if beginner_mode:
            rr_to_tp1 = round(abs(main_sc.tp1 - main_sc.entry) / max(abs(main_sc.entry - main_sc.sl), 1e-9), 2)
            st.markdown(f"### Что сделать сейчас — {'ПОКУПКА (LONG)' if main_sc.bias=='long' else 'ПРОДАЖА (SHORT)'}")
            st.markdown(
                f"- **Почему:** {main_sc.explain_short}  \n"
                f"- **Подтверждений:** {main_sc.confirms} — {', '.join(main_sc.confirm_list)}  \n"
                f"- **Вход:** {fmt_price(main_sc.entry, decimals)}  \n"
                f"- **Стоп:** {fmt_price(main_sc.sl, decimals)} ({main_sc.stop_reason}; "
                f"риск ≈ {fmt_price(abs(main_sc.entry-main_sc.sl), decimals)}, ATR≈{fmt_price(atr_v, decimals)})  \n"
                f"- **Цели:** TP1 {fmt_price(main_sc.tp1, decimals)} ({main_sc.tp_reason}); "
                + (f"TP2 {fmt_price(main_sc.tp2, decimals)}" if main_sc.tp2 else "без TP2")
                + f" • **R:R≈{rr_to_tp1}**"
            )
    else:
        st.info("Нет понятного входа: подождать свипа/ретеста ключевых уровней.")

    # таблица
    rows = []
    for sc in scenarios:
        key = f"{sc.name} ({sc.bias})"
        rows.append({
            "Сценарий": key,
            "Тип": sc.etype,
            "Подтв.": f"{sc.confirms} — {', '.join(sc.confirm_list)}",
            "Вход": fmt_price(sc.entry, decimals),
            "Стоп": fmt_price(sc.sl, decimals),
            "TP1": fmt_price(sc.tp1, decimals),
            "TP2": fmt_price(sc.tp2, decimals) if sc.tp2 else "—",
            "R:R до TP1": round(abs(sc.tp1 - sc.entry) / max(abs(sc.entry - sc.sl), 1e-9), 2),
            "Prob%": round(sc_probs.get(key, 0.0), 2),
        })
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # подробные карточки по каждому сценарию
    st.markdown("### Подробности по сценариям")
    for sc in scenarios:
        with st.expander(f"{sc.name} ({sc.bias}) — {sc.confirms} подтвержд.: {fmt_reason_list(sc.confirm_list)}"):
            st.markdown(
                f"**Логика:** {' → '.join(sc.logic_path)}  \n"
                f"**Триггер:** {sc.trigger}  \n"
                f"**Вход:** {fmt_price(sc.entry, decimals)}  \n"
                f"**Стоп:** {fmt_price(sc.sl, decimals)} — {sc.stop_reason}  \n"
                f"**TP1:** {fmt_price(sc.tp1, decimals)} — {sc.tp_reason}  \n"
                f"**TP2:** {fmt_price(sc.tp2, decimals) if sc.tp2 else '—'}  \n"
                f"**Замечание:** вход только при подтверждении свечой в пользу стороны; "
                f"если импульс утащил цену и стоп стал чрезмерным — пропускаем."
            )

    summary.append(f"{asset} {tf} → режим {reg}; идей {len(scenarios)}; минимум подтверждений {min_confirms}")
    st.divider()

except Exception as e:
    st.error(f"{asset}: {e}")

st.subheader("Зведення")
for line in summary:
    st.write("•", line)
