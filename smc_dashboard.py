# smc_dashboard.py
# -*- coding: utf-8 -*-

"""
SMC Intraday — BTC / ETH / XAUUSD / XAUEUR / EURUSD (text)
Качественный SMC-анализ c фоллбэком источников (yfinance), выбором актива и экспортом в Pine v5.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf


# ==============================
#        Конфигурация
# ==============================

# Список активов в UI
ASSETS = [
    "BTCUSDT",   # BTC-USD
    "ETHUSDT",   # ETH-USD
    "XAUUSD",    # золото к USD
    "XAUEUR",    # золото к EUR
    "EURUSD",    # EURUSD
]

# Карта кандидатов тикеров Yahoo для каждого актива (по порядку пробуем, пока не появятся свечи)
YF_TICKER_CANDIDATES: Dict[str, List[str]] = {
    "BTCUSDT": ["BTC-USD"],
    "ETHUSDT": ["ETH-USD"],
    # XAUUSD: сначала спот курс, затем фьючерс (у фьючерса обычно есть 5m/15m)
    "XAUUSD": ["XAUUSD=X", "GC=F"],
    "XAUEUR": ["XAUEUR=X"],
    "EURUSD": ["EURUSD=X"],
}

# Интервалы в UI -> интервалы Yahoo + период
# Если выбранный tf не даёт данных для тикера — попробуем следующий по списку
TF_FALLBACKS = {
    "5m":  [("5m", "60d"), ("15m", "60d"), ("60m", "730d")],
    "15m": [("15m", "60d"), ("60m", "730d")],
    "1h":  [("60m", "730d"), ("1d", "730d")],
}

# HTF для контекста (подбираем доступные интервалы Yahoo)
HTF_OF = {"5m": "15m", "15m": "60m", "1h": "1d"}

# Ссылка TradingView для удобного открытия графика
TV_SYMBOL = {
    "BTCUSDT": "BINANCE:BTCUSDT",
    "ETHUSDT": "BINANCE:ETHUSDT",
    "XAUUSD": "OANDA:XAUUSD",     # обычно доступен в TV
    "XAUEUR": "OANDA:XAU_EUR",    # если нет — поменяйте на свою биржу
    "EURUSD": "OANDA:EURUSD",
}


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
    # Если объёмов нет (FX), VWAP деградирует к цене
    tp  = (df["high"] + df["low"] + df["close"]) / 3.0
    vol = df["volume"].replace(0, np.nan).fillna(0.0)
    num = (tp * vol).cumsum()
    den = vol.cumsum().replace(0, np.nan)
    vw  = (num / den).fillna(method="bfill").fillna(df["close"])
    return vw


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
    edges = np.linspace(lo, hi, bins + 1); vol = np.zeros(bins)
    prices = df["close"].values; vols = df["volume"].values
    idx = np.clip(np.digitize(prices, edges) - 1, 0, bins - 1)
    for i, v in zip(idx, vols): vol[i] += v
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


# ==============================
#   Bias/режим/вероятности
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

def rr_targets(entry: float, sl: float, bias: str, min_rr: float = 2.0) -> Tuple[float, float, str]:
    risk = abs(entry - sl) or 1e-6
    if bias == "long":
        tp1 = entry + min_rr * risk; tp2 = entry + 3.0 * risk
    else:
        tp1 = entry - min_rr * risk; tp2 = entry - 3.0 * risk
    return tp1, tp2, f"1:{int(min_rr)}/1:3"

def scenario_ev(entry, sl, tp1, prob):
    risk = abs(entry - sl); reward = abs(tp1 - entry)
    fees = 0.0002 * (risk + reward)
    return prob * reward - (1 - prob) * risk - fees

def explain_scenario(name: str, bias: str, details: Dict[str, float | str]) -> str:
    side = "LONG" if bias == "long" else "SHORT"; parts = []
    if name == "FVG mitigation":
        parts += ["Откат к середине актуального FVG по направлению BOS.",
                  f"Подтверждение: свеча в пользу {side} после касания mid-FVG."]
    elif name == "OB Retest":
        parts += ["Ретест импульсного Order Block по тренду.",
                  "Подтверждение: отбой у границы блока; хвосты внутри OB слабые."]
    elif name == "BOS Break & Retest":
        lvl = details.get("lvl"); parts += [f"Пробой ключевого swing {lvl:.2f} и ретест.",
                                            "Подтверждение: закрепление и отскок по направлению пробоя."]
    elif name == "Breaker":
        lvl = details.get("lvl"); parts += [f"Свип ликвидности на {lvl:.2f} и возврат (breaker).",
                                            "Подтверждение: ложный прокол и закрытие обратно."]
    elif name == "Value Area Reversion":
        edge = details.get("edge"); parts += [f"От края value area ({edge}) к POC (mean-reversion).",
                                              "Подтверждение: разворот у VAL/VAH при низком ADX."]
    elif name == "EMA Pullback":
        parts += ["Продолжение тренда через откат к EMA20.",
                  "Подтверждение: касание EMA и возобновление импульса."]
    elif name == "Structure Breakout":
        lvl = details.get("lvl"); parts += [f"Пробой последнего фрактала {lvl:.2f} и продолжение.",
                                            "Подтверждение: ускорение без глубоких возвратов."]
    elif name == "VWAP Bounce":
        parts += ["Откат к VWAP и отбой по тренду.",
                  "Подтверждение: удержание VWAP и согласованный OBV."]
    elif name == "POC Flip":
        parts += ["Перехват контроля на POC (flip) и ретест с обратной стороны.",
                  "Подтверждение: POC становится поддержкой/сопротивлением."]
    elif name == "SFP Reversal":
        lvl = details.get("lvl"); parts += [f"SFP у {lvl:.2f}: прокол и закрытие обратно → вероятен реверс.",
                                            "Подтверждение: сильная разворотная свеча; хвост больше тела."]
    else:
        parts += ["Сетап по контексту и уровням."]
    parts.append("Отмена: сильное закрепление по другую сторону стоп-уровня.")
    return " ".join(parts)

def propose(df: pd.DataFrame, htf_bias: str, d_bias: str, regime: str,
            vp: Dict[str, float | np.ndarray], obv_slope: float) -> List[Scenario]:
    price = float(df["close"].iloc[-1])
    at = float(atr(df).iloc[-1]); at = max(at, 1e-6)
    SH, SL = swings(df); dir_, t, _ = bos(df, SH, SL)
    gaps = fvg(df); swp = sweeps(df, SH, SL); ob = simple_ob(df, dir_, t)
    sh_lvl, sl_lvl = last_swing_levels(df, SH, SL)
    vw = float(vwap_series(df).iloc[-1]); ema20 = float(ema(df["close"], 20).iloc[-1])

    sc: List[Scenario] = []

    # FVG
    if dir_ == "up" and gaps["bull"] and regime == "trend":
        _, lo, hi = list(reversed(gaps["bull"]))[0]; e = (lo + hi) / 2; sl = e - 1.2 * at
        tp1, tp2, rr = rr_targets(e, sl, "long")
        sc.append(Scenario("FVG mitigation", "long", "limit", f"mid FVG {e:.2f}", e, sl, tp1, tp2, rr,
                           explain_scenario("FVG mitigation", "long", {})))
    if dir_ == "down" and gaps["bear"] and regime == "trend":
        _, lo, hi = list(reversed(gaps["bear"]))[0]; e = (lo + hi) / 2; sl = e + 1.2 * at
        tp1, tp2, rr = rr_targets(e, sl, "short")
        sc.append(Scenario("FVG mitigation", "short", "limit", f"mid FVG {e:.2f}", e, sl, tp1, tp2, rr,
                           explain_scenario("FVG mitigation", "short", {})))

    # OB ретест
    if dir_ == "up" and ob.get("demand") and regime == "trend":
        _, lo, hi = ob["demand"]; e = hi; sl = lo - 0.6 * at
        tp1, tp2, rr = rr_targets(e, sl, "long")
        sc.append(Scenario("OB Retest", "long", "limit", f"retest OB {lo:.2f}-{hi:.2f}", e, sl, tp1, tp2, rr,
                           explain_scenario("OB Retest", "long", {})))
    if dir_ == "down" and ob.get("supply") and regime == "trend":
        _, lo, hi = ob["supply"]; e = lo; sl = hi + 0.6 * at
        tp1, tp2, rr = rr_targets(e, sl, "short")
        sc.append(Scenario("OB Retest", "short", "limit", f"retest OB {lo:.2f}-{hi:.2f}", e, sl, tp1, tp2, rr,
                           explain_scenario("OB Retest", "short", {})))

    # BOS + ретест
    if dir_ == "up" and t is not None:
        lvl = float(df.loc[t, "high"]); e = lvl + 0.2 * at; sl = lvl - 0.8 * at
        tp1, tp2, rr = rr_targets(e, sl, "long")
        sc.append(Scenario("BOS Break & Retest", "long", "stop", f"пробой {lvl:.2f}+ретест", e, sl, tp1, tp2, rr,
                           explain_scenario("BOS Break & Retest", "long", {"lvl": lvl})))
    if dir_ == "down" and t is not None:
        lvl = float(df.loc[t, "low"]); e = lvl - 0.2 * at; sl = lvl + 0.8 * at
        tp1, tp2, rr = rr_targets(e, sl, "short")
        sc.append(Scenario("BOS Break & Retest", "short", "stop", f"пробой {lvl:.2f}+ретест", e, sl, tp1, tp2, rr,
                           explain_scenario("BOS Break & Retest", "short", {"lvl": lvl})))

    # Breaker (после свипа)
    if swp["high"] and dir_ == "down":
        _, lvl_s = swp["high"][-1]; e = lvl_s - 0.1 * at; sl = lvl_s + 0.7 * at
        tp1, tp2, rr = rr_targets(e, sl, "short")
        sc.append(Scenario("Breaker", "short", "stop", f"breaker после свипа {lvl_s:.2f}", e, sl, tp1, tp2, rr,
                           explain_scenario("Breaker", "short", {"lvl": lvl_s})))
    if swp["low"] and dir_ == "up":
        _, lvl_s = swp["low"][-1]; e = lvl_s + 0.1 * at; sl = lvl_s - 0.7 * at
        tp1, tp2, rr = rr_targets(e, sl, "long")
        sc.append(Scenario("Breaker", "long", "stop", f"breaker после свипа {lvl_s:.2f}", e, sl, tp1, tp2, rr,
                           explain_scenario("Breaker", "long", {"lvl": lvl_s})))

    # Диапазон: VAL/VAH → POC
    if regime == "range":
        if abs(price - vp["val"]) <= max(0.6 * at, 0.1 * (vp["vah"] - vp["val"])):
            e = vp["val"] + 0.1 * at; sl = vp["val"] - 0.8 * at
            tp1, tp2, rr = rr_targets(e, sl, "long")
            sc.append(Scenario("Value Area Reversion", "long", "limit", "от VAL к POC", e, sl, tp1, tp2, rr,
                               explain_scenario("Value Area Reversion", "long", {"edge": "VAL"})))
        if abs(price - vp["vah"]) <= max(0.6 * at, 0.1 * (vp["vah"] - vp["val"])):
            e = vp["vah"] - 0.1 * at; sl = vp["vah"] + 0.8 * at
            tp1, tp2, rr = rr_targets(e, sl, "short")
            sc.append(Scenario("Value Area Reversion", "short", "limit", "от VAH к POC", e, sl, tp1, tp2, rr,
                               explain_scenario("Value Area Reversion", "short", {"edge": "VAH"})))

    # EMA pullback
    if regime == "trend" and price > ema20:
        e = ema20; sl = e - 1.2 * at
        tp1, tp2, rr = rr_targets(e, sl, "long")
        sc.append(Scenario("EMA Pullback", "long", "limit", f"к EMA20 {e:.2f}", e, sl, tp1, tp2, rr,
                           explain_scenario("EMA Pullback", "long", {})))
    if regime == "trend" and price < ema20:
        e = ema20; sl = e + 1.2 * at
        tp1, tp2, rr = rr_targets(e, sl, "short")
        sc.append(Scenario("EMA Pullback", "short", "limit", f"к EMA20 {e:.2f}", e, sl, tp1, tp2, rr,
                           explain_scenario("EMA Pullback", "short", {})))

    # Structure Breakout / Breakdown
    if sh_lvl is not None:
        base_sl = (sl_lvl if sl_lvl is not None else price - 1.2 * at)
        e = sh_lvl + 0.2 * at; sl = base_sl - 0.4 * at
        tp1, tp2, rr = rr_targets(e, sl, "long")
        sc.append(Scenario("Structure Breakout", "long", "stop",
                           f"breakout swing-high {sh_lvl:.2f}", e, sl, tp1, tp2, rr,
                           explain_scenario("Structure Breakout", "long", {"lvl": sh_lvl})))
    if sl_lvl is not None:
        base_sl = (sh_lvl if sh_lvl is not None else price + 1.2 * at)
        e = sl_lvl - 0.2 * at; sl = base_sl + 0.4 * at
        tp1, tp2, rr = rr_targets(e, sl, "short")
        sc.append(Scenario("Structure Breakout", "short", "stop",
                           f"breakdown swing-low {sl_lvl:.2f}", e, sl, tp1, tp2, rr,
                           explain_scenario("Structure Breakout", "short", {"lvl": sl_lvl})))

    # VWAP bounce/flip
    if abs(price - vw) <= 1.2 * at:
        if obv_slope > 0:
            e = vw; sl = e - 1.0 * at
            tp1, tp2, rr = rr_targets(e, sl, "long")
            sc.append(Scenario("VWAP Bounce", "long", "limit", f"от VWAP {vw:.2f}", e, sl, tp1, tp2, rr,
                               explain_scenario("VWAP Bounce", "long", {})))
        if obv_slope < 0:
            e = vw; sl = e + 1.0 * at
            tp1, tp2, rr = rr_targets(e, sl, "short")
            sc.append(Scenario("VWAP Bounce", "short", "limit", f"от VWAP {vw:.2f}", e, sl, tp1, tp2, rr,
                               explain_scenario("VWAP Bounce", "short", {})))

    # POC flip
    if abs(price - vp["poc"]) <= 0.6 * at:
        if price > vp["poc"]:
            e = vp["poc"] + 0.1 * at; sl = vp["poc"] - 0.7 * at
            tp1, tp2, rr = rr_targets(e, sl, "long")
            sc.append(Scenario("POC Flip", "long", "limit", "ретест POC сверху", e, sl, tp1, tp2, rr,
                               explain_scenario("POC Flip", "long", {})))
        else:
            e = vp["poc"] - 0.1 * at; sl = vp["poc"] + 0.7 * at
            tp1, tp2, rr = rr_targets(e, sl, "short")
            sc.append(Scenario("POC Flip", "short", "limit", "ретест POC снизу", e, sl, tp1, tp2, rr,
                               explain_scenario("POC Flip", "short", {})))

    # SFP (контр-тренд у последнего свинга)
    if sh_lvl is not None and df["high"].iloc[-1] > sh_lvl and df["close"].iloc[-1] < sh_lvl:
        e = sh_lvl - 0.1 * at; sl = sh_lvl + 0.9 * at
        tp1, tp2, rr = rr_targets(e, sl, "short")
        sc.append(Scenario("SFP Reversal", "short", "stop", f"SFP у {sh_lvl:.2f}", e, sl, tp1, tp2, rr,
                           explain_scenario("SFP Reversal", "short", {"lvl": sh_lvl})))
    if sl_lvl is not None and df["low"].iloc[-1] < sl_lvl and df["close"].iloc[-1] > sl_lvl:
        e = sl_lvl + 0.1 * at; sl = sl_lvl - 0.9 * at
        tp1, tp2, rr = rr_targets(e, sl, "long")
        sc.append(Scenario("SFP Reversal", "long", "stop", f"SFP у {sl_lvl:.2f}", e, sl, tp1, tp2, rr,
                           explain_scenario("SFP Reversal", "long", {"lvl": sl_lvl})))

    # Сортировка по контексту
    def _sort_key(x: Scenario):
        base = 0
        if (x.name.startswith(("FVG", "BOS", "OB", "EMA", "Structure", "VWAP", "POC", "SFP")) and regime == "trend") or \
           (x.name.startswith(("Value Area", "Breaker")) and regime == "range"):
            base -= 2
        if x.bias == htf_bias: base -= 1
        return base

    sc = sorted(sc, key=_sort_key)

    # Уникальность и лимит
    uniq, seen = [], set()
    for s in sc:
        k = (s.name, s.bias)
        if k in seen: continue
        seen.add(k); uniq.append(s)
        if len(uniq) >= 8: break
    if not uniq:
        c = price
        uniq.append(Scenario("Wait (no-trade)", "none", "—", "нет валидных сетапов", c, c, c, c, "—",
                             "Пауза: дождаться свипа/ретеста ключевых уровней."))
    return uniq

def scenario_probabilities(
    scen: List[Scenario], htf_bias: str, d_bias: str, obv_slope: float,
    price: float, vp: Dict[str, float | np.ndarray], atr_val: float, regime: str,
    *, cap: float = 0.90, floor: float = 0.05, temp: float = 1.25
) -> Tuple[Dict[str, float], Dict[str, float]]:
    if not scen: return {"Wait (no-trade)": 100.0}, {"long": 0.0, "short": 0.0}
    scores, labels = [], []
    for s in scen:
        if s.name.startswith("Wait"): continue
        sc = 0.0
        if s.bias == htf_bias: sc += 2.2
        if s.bias == d_bias:  sc += 1.2
        sc += 0.9 if (obv_slope > 0 and s.bias == "long") or (obv_slope < 0 and s.bias == "short") else -0.2
        if (s.name.startswith(("FVG","BOS","OB","EMA","Structure","POC","VWAP","SFP")) and regime == "trend") or \
           (s.name.startswith(("Value Area","Breaker")) and regime == "range"): sc += 1.0
        if regime == "range" and s.name.startswith("Structure"): sc -= 0.6
        dist = abs(s.entry - price) / max(atr_val, 1e-6)
        if dist > 2.0: sc -= 1.0
        elif dist > 1.5: sc -= 0.5
        above_poc = price > vp["poc"]
        if s.name.startswith(("FVG","BOS","OB","EMA","Structure","POC Flip","VWAP","SFP")):
            sc += 0.4 if (above_poc and s.bias == "long") or ((not above_poc) and s.bias == "short") else 0.0
        if s.name.startswith("Value Area"):
            near_edge = min(abs(price - vp["val"]), abs(price - vp["vah"])) <= 0.8 * atr_val
            sc += 0.5 if near_edge else -0.2
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
#       Данные через Yahoo
# ==============================

@st.cache_data(show_spinner=False, ttl=60)
def yf_ohlc_first_success(asset_key: str, tf: str, limit: int = 800) -> Tuple[pd.DataFrame, str, str]:
    """
    Пробуем несколько тикеров и фоллбэк-интервалы для выбранного tf.
    Возвращаем (df, фактический_interval, фактический_period)
    """
    cands = YF_TICKER_CANDIDATES.get(asset_key, [asset_key])
    tries = TF_FALLBACKS.get(tf, TF_FALLBACKS["15m"])
    last_err = None
    for tkr in cands:
        for interval, period in tries:
            try:
                df = yf.download(tkr, interval=interval, period=period, auto_adjust=False, progress=False)
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
                # TZ → UTC
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

def tv_chart_url(asset_key: str, tf_effective: str) -> str:
    interval_map = {"5m":"5","15m":"15","60m":"60","1d":"D"}
    sym = TV_SYMBOL.get(asset_key, asset_key)
    itv = interval_map.get(tf_effective, "15")
    return f"https://www.tradingview.com/chart/?symbol={sym}&interval={itv}"

def pine_for_scenario(asset_key: str, tf_label: str, sc: Scenario) -> str:
    """Pine v5 для отрисовки Entry/SL/TP на графике TradingView."""
    side = "LONG" if sc.bias == "long" else "SHORT"
    return f"""//@version=5
indicator("SMC Idea — {asset_key} {tf_label} — {sc.name} ({side})", overlay=true)

// ---- levels ----
var float entry = {sc.entry}
var float sl    = {sc.sl}
var float tp1   = {sc.tp1}
var float tp2   = {('na' if sc.tp2 is None else sc.tp2)}
var string side = "{side}"
var string name = "{sc.name}"

// ---- plot ----
plot(entry, "ENTRY", color.new(color.teal, 0), 2)
plot(sl,    "SL",    color.new(color.red, 0), 2)
plot(tp1,   "TP1",   color.new(color.green, 0), 2)
plot(tp2,   "TP2",   color.new(color.green, 40), 2)

if barstate.islast
    label.new(bar_index, entry, "ENTRY\\n" + str.tostring(entry, format.mintick), style=label.style_label_left, color=color.teal, textcolor=color.white)
    label.new(bar_index, sl,    "SL\\n"    + str.tostring(sl,    format.mintick), style=label.style_label_left, color=color.red,  textcolor=color.white)
    label.new(bar_index, tp1,   "TP1\\n"   + str.tostring(tp1,   format.mintick), style=label.style_label_left, color=color.green,textcolor=color.white)
    if not na(tp2)
        label.new(bar_index, tp2, "TP2\\n" + str.tostring(tp2,   format.mintick), style=label.style_label_left, color=color.new(color.green,40), textcolor=color.white)

rr = math.abs(tp1 - entry) / math.abs(entry - sl)
txt = name + " (" + side + ")\\n" +
      "Entry: " + str.tostring(entry, format.mintick) + "\\n" +
      "SL: "    + str.tostring(sl,    format.mintick) + "\\n" +
      "TP1: "   + str.tostring(tp1,   format.mintick) + "\\n" +
      "TP2: "   + (na(tp2) ? "—" : str.tostring(tp2, format.mintick)) + "\\n" +
      "R:R to TP1 ≈ " + str.tostring(rr, format.mintick)
if barstate.islast
    label.new(bar_index, high, txt, style=label.style_label_upper_left, textcolor=color.white, color=color.new(color.black, 0))
"""


# ==============================
#             UI
# ==============================

st.set_page_config(page_title="SMC Intraday (text)", layout="wide")
st.title("SMC Intraday — BTC / ETH / XAUUSD / XAUEUR / EURUSD (text)")

colA, colB, colC, colD = st.columns([1.2, 1, 1, 1])
with colA:
    asset = st.selectbox("Актив", ASSETS, index=0)
with colB:
    tf = st.selectbox("TF", ["5m", "15m", "1h"], index=0)
with colC:
    min_risk_pct = st.slider("Мин. риск (%ATR)", 5, 60, 25, step=5)
with colD:
    min_tp1_atr = st.slider("Мин. TP1 (×ATR)", 1.0, 3.0, 1.5, step=0.25)

colE, colF = st.columns([1, 1])
with colE:
    it = st.selectbox("Auto-refresh", ["30s", "1m", "2m", "5m"], index=3)
with colF:
    beginner_mode = st.checkbox("Простой режим (для новичка)", value=True)

st.markdown(
    f"<meta http-equiv='refresh' content='{ {'30s':30,'1m':60,'2m':120,'5m':300}[it] }'>",
    unsafe_allow_html=True,
)

with st.expander("📘 Справочник (нажми, чтобы открыть)"):
    st.markdown(
        "- **ATR** — диапазон для стопов/подтверждений.  "
        "- **BOS** — пробой структуры с закрытием ≥ 0.30×ATR.  \n"
        "- **FVG/OB/Breaker/SFP** — базовые SMC-паттерны.  "
        "- **Value Area/POC** — объёмная зона (приблизительно), в боковике играем от VAL/VAH к POC.  \n"
        "- Вероятности — относительные (softmax), это ранжирование идей, а не гарантия.  \n"
        "- В **TradingView** нельзя автоматически нанести уровни программно — используйте сгенерированный **Pine v5** (копировать → вставить в Pine-редактор)."
    )

st.caption(
    "Сценарии с риском ниже порога %ATR и с TP1 меньше заданного множителя ATR скрываются. "
    "TP1=2R, TP2=3R. Вероятности нормированы (<100%)."
)


# ==============================
#        Основной поток
# ==============================

summary = []

try:
    # LTF
    df, tf_eff, period_eff = yf_ohlc_first_success(asset, tf, limit=800)

    # HTF
    htf = HTF_OF[tf]
    df_h, _, _ = yf_ohlc_first_success(asset, htf, limit=400)

    # Daily (1d)
    df_d, _, _ = yf_ohlc_first_success(asset, "1h", limit=24 * 200) if tf != "1h" else yf_ohlc_first_success(asset, "1h", limit=24 * 200)

    price = float(df["close"].iloc[-1])
    vp = volume_profile(df)
    reg = market_regime(df, vp)
    atr_v = float(atr(df).iloc[-1])

    # OBV-уклон (если нет объёмов, будет около 0)
    o = obv(df); wnd = min(len(o), 160)
    slope = (np.polyfit(np.arange(wnd), o.tail(wnd), 1)[0] if wnd >= 20 else 0.0)

    htf_bias = score_bias(df_h)
    d_bias = regime_daily(df_d)

    scenarios_all = propose(df, htf_bias, d_bias, reg, vp, slope)

    # Фильтры
    min_risk = atr_v * (min_risk_pct / 100.0)
    scenarios: List[Scenario] = []
    for sc in scenarios_all:
        if sc.name.startswith("Wait"):
            scenarios.append(sc)
            continue
        risk_ok = abs(sc.entry - sc.sl) >= min_risk
        tp1_ok = (abs(sc.tp1 - sc.entry) / max(atr_v, 1e-6)) >= min_tp1_atr
        if risk_ok and tp1_ok:
            scenarios.append(sc)
    if not [x for x in scenarios if not x.name.startswith("Wait")]:
        scenarios = scenarios_all

    sc_probs, bias_summary = scenario_probabilities(scenarios, htf_bias, d_bias, slope, price, vp, atr_v, reg)

    st.markdown(f"## {asset} ({tf}) — цена: {price:,.2f}".replace(",", " "))

    ltf_b = score_bias(df); htf_b = score_bias(df_h); d_b = regime_daily(df_d)
    top_pair = list(sc_probs.items())[0] if sc_probs else ("Wait (no-trade)", 100.0)
    top_name, top_prob = top_pair[0], top_pair[1]

    def _fmt_price(x: float) -> str: return f"{x:,.2f}".replace(",", " ")
    def _fmt_pct(x: float) -> str: return f"{x:.1f}%"
    def _rr(entry: float, sl: float, tp1: float) -> float:
        risk = abs(entry - sl) or 1e-6; reward = abs(tp1 - entry); return round(reward / risk, 2)

    poc_state = "выше VAH" if price > vp["vah"] else ("ниже VAL" if price < vp["val"] else "внутри value area")
    st.markdown(
        f"**Контекст:** LTF={ltf_b.upper()}, HTF={htf_b.upper()}, Daily={d_b.upper()} • "
        f"Режим: {reg.upper()} (ADX≈{float(adx(df).iloc[-1]):.1f}) • "
        f"POC {vp['poc']:.2f}, VAL {vp['val']:.2f}, VAH {vp['vah']:.2f} → цена {poc_state}.  \n"
        f"**Самый вероятный:** {top_name} ≈ {top_prob:.1f}% • "
        f"**Баланс:** LONG {bias_summary['long']:.1f}% vs SHORT {bias_summary['short']:.1f}%"
    )

    # Главная карточка / Beginner
    if len(scenarios) == 1 and scenarios[0].name.startswith("Wait"):
        st.info("Нет понятного входа: подождать свипа/ретеста ключевых уровней.")
        main_sc = scenarios[0]; main_prob = 100.0
    else:
        main_sc = None; main_prob = 0.0
        for sc in scenarios:
            key = f"{sc.name} ({sc.bias})"
            if key == top_name:
                main_sc = sc; main_prob = sc_probs.get(key, 0.0); break
        if main_sc is None:
            main_sc = [x for x in scenarios if not x.name.startswith("Wait")][0]
            main_prob = sc_probs.get(f"{main_sc.name} ({main_sc.bias})", 0.0)

    if beginner_mode:
        rr_to_tp1 = _rr(main_sc.entry, main_sc.sl, main_sc.tp1)
        st.markdown(f"### Что сделать сейчас — {'ПОКУПКА (LONG)' if main_sc.bias=='long' else 'ПРОДАЖА (SHORT)'}")
        st.markdown(
            "- **Почему:** " + main_sc.explain + "\n"
            f"- **Вход:** {_fmt_price(main_sc.entry)}  \n"
            f"- **Стоп:** {_fmt_price(main_sc.sl)} (риск ≈ {_fmt_price(abs(main_sc.entry-main_sc.sl))}, ~{_fmt_price(atr_v)} по цене (ATR))  \n"
            f"- **Цели:** TP1 {_fmt_price(main_sc.tp1)} (R:R≈{rr_to_tp1}), "
            + (f"TP2 {_fmt_price(main_sc.tp2)}  " if main_sc.tp2 else "без TP2  ")
            + f"• **вероятность:** {_fmt_pct(main_prob)}\n"
            "- **Не входить, если:** импульс ушёл далеко и стоп становится чрезмерным, "
              "или цена закрепилась за уровнем, который должен был удерживаться."
        )
    else:
        st.markdown("### Основные сценарии по убыванию вероятности")

    # Таблица
    rows = []
    for sc in scenarios:
        if sc.name.startswith("Wait"): continue
        key = f"{sc.name} ({sc.bias})"
        rows.append({
            "Сценарий": key,
            "Тип": sc.etype,
            "Вход": _fmt_price(sc.entry),
            "Стоп": _fmt_price(sc.sl),
            "TP1": _fmt_price(sc.tp1),
            "TP2": _fmt_price(sc.tp2) if sc.tp2 else "—",
            "R:R до TP1": _rr(sc.entry, sc.sl, sc.tp1),
            "Prob%": round(sc_probs.get(key, 0.0), 2),
        })
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # EV
    with st.expander("Показать математику (EV)"):
        ev_rows = []
        for sc in scenarios:
            if sc.name.startswith("Wait"): continue
            key = f"{sc.name} ({sc.bias})"; p = sc_probs.get(key, 0.0) / 100.0
            ev_rows.append({"Сценарий": key, "Prob%": round(p * 100, 2),
                            "EV": round(scenario_ev(sc.entry, sc.sl, sc.tp1, p), 6)})
        if ev_rows:
            st.dataframe(pd.DataFrame(ev_rows), use_container_width=True, hide_index=True)

    # Экспорт в TradingView (ручной, без «автонакатки» — API нет)
    pine_code = pine_for_scenario(asset, tf, main_sc)
    st.markdown("### Экспорт в TradingView")
    st.code(pine_code, language="pine")
    st.link_button("📈 Открыть график TradingView", tv_chart_url(asset, tf_eff))
    st.caption("Скопируйте код выше → откройте TradingView → Pine Editor → вставьте → Save → Add to chart.")

    summary.append(f"{asset} {tf} → режим {reg}; HTF {htf} bias {htf_bias}; Top: {top_name}")
    st.divider()

except Exception as e:
    st.error(f"{asset}: {e}")

st.subheader("Зведення")
for line in summary:
    st.write("•", line)
