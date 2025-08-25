# smc_dashboard.py
# -*- coding: utf-8 -*-

"""
SMC Intraday — BTC / ETH / XAUUSD / XAUEUR / EURUSD (text, SQLite journal)
- Источник данных: Yahoo Finance (с фоллбэком интервалов)
- Ядро подтверждений: sweep, BOS, imbalance(FVG), breaker, OB-retest
- Вход всегда от mid-FVG (0.5)
- Уведомления в Telegram при РОВНО 2 подтверждениях (антиспам в alerts)
- Журнал сделок на SQLite + простая статистика
"""

from __future__ import annotations

import json
import math
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf

# ==============================
#          CONFIG
# ==============================

ASSETS = ["BTCUSDT", "ETHUSDT", "XAUUSD", "XAUEUR", "EURUSD"]

YF_TICKER_CANDIDATES: Dict[str, List[str]] = {
    "BTCUSDT": ["BTC-USD"],
    "ETHUSDT": ["ETH-USD"],
    "XAUUSD": ["XAUUSD=X", "GC=F"],
    "XAUEUR": ["XAUEUR=X"],
    "EURUSD": ["EURUSD=X"],
}

TF_FALLBACKS = {
    "5m":  [("5m", "60d"), ("15m", "60d"), ("60m", "730d")],
    "15m": [("15m", "60d"), ("60m", "730d")],
    "1h":  [("60m", "730d"), ("1d", "730d")],
}
HTF_OF = {"5m": "15m", "15m": "60m", "1h": "1d"}

DB_PATH = "trades.sqlite"

# ⚠️ По просьбе пользователя: токен прямо в коде.
# Обязательно ПЕРЕВЫПУСТИТЕ токен после тестов.
TELEGRAM_BOT_TOKEN = "6231361993:AAFCKT2rPnoJv2K4OXAZCpOq8KcjmFGvJjw"

# ==============================
#        DB helpers (SQLite)
# ==============================

def db_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def db_init():
    conn = db_conn()
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS trades (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      ts_utc TEXT NOT NULL,
      asset TEXT NOT NULL,
      tf TEXT NOT NULL,
      side TEXT NOT NULL,
      entry REAL NOT NULL,
      sl REAL NOT NULL,
      tp1 REAL NOT NULL,
      tp2 REAL,
      rr REAL,
      confirmations TEXT,        -- JSON list
      confirm_count INTEGER,
      context TEXT,               -- JSON: regime, biases, vp, atr, notes
      scenario TEXT,              -- human explanation
      status TEXT DEFAULT 'open', -- open/win/stop/be/closed
      result_r REAL,              -- realized R if known
      notes TEXT
    );""")
    cur.execute("""
    CREATE TABLE IF NOT EXISTS alerts (
      key TEXT PRIMARY KEY,       -- asset:tf:side
      last_sent_utc TEXT,
      last_price REAL,
      confirms INTEGER
    );""")
    cur.execute("""
    CREATE TABLE IF NOT EXISTS settings (
      skey TEXT PRIMARY KEY,
      sval TEXT
    );""")
    conn.commit()
    conn.close()

def settings_get(key: str) -> Optional[str]:
    conn = db_conn(); cur = conn.cursor()
    cur.execute("SELECT sval FROM settings WHERE skey=?", (key,))
    r = cur.fetchone()
    conn.close()
    return r["sval"] if r else None

def settings_set(key: str, value: str):
    conn = db_conn(); cur = conn.cursor()
    cur.execute("INSERT INTO settings(skey, sval) VALUES(?,?) ON CONFLICT(skey) DO UPDATE SET sval=excluded.sval", (key, value))
    conn.commit(); conn.close()

def journal_insert(asset: str, tf: str, side: str, entry: float, sl: float, tp1: float,
                   tp2: Optional[float], rr: float, confirmations: List[str],
                   context: Dict, scenario: str):
    conn = db_conn(); cur = conn.cursor()
    cur.execute("""INSERT INTO trades
      (ts_utc, asset, tf, side, entry, sl, tp1, tp2, rr, confirmations, confirm_count, context, scenario)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
      (datetime.utcnow().isoformat(), asset, tf, side, entry, sl, tp1, tp2, rr,
       json.dumps(confirmations, ensure_ascii=False), len(confirmations),
       json.dumps(context, ensure_ascii=False), scenario))
    conn.commit(); conn.close()

def journal_read_df() -> pd.DataFrame:
    conn = db_conn()
    df = pd.read_sql_query("SELECT * FROM trades ORDER BY id DESC", conn)
    conn.close()
    return df

def journal_update_row(row_id: int, status: str, result_r: Optional[float], notes: str):
    conn = db_conn(); cur = conn.cursor()
    cur.execute("UPDATE trades SET status=?, result_r=?, notes=? WHERE id=?",
                (status, result_r if result_r is not None else None, notes, row_id))
    conn.commit(); conn.close()

def alert_should_send(asset: str, tf: str, side: str, price: float, confirms: int, cool_minutes=30) -> bool:
    key = f"{asset}:{tf}:{side}"
    conn = db_conn(); cur = conn.cursor()
    cur.execute("SELECT * FROM alerts WHERE key=?", (key,))
    r = cur.fetchone()
    now = datetime.utcnow()
    if not r:
        cur.execute("INSERT INTO alerts(key, last_sent_utc, last_price, confirms) VALUES(?,?,?,?)",
                    (key, now.isoformat(), price, confirms))
        conn.commit(); conn.close()
        return True
    last = datetime.fromisoformat(r["last_sent_utc"])
    ok = (now - last) > timedelta(minutes=cool_minutes)
    if ok:
        cur.execute("UPDATE alerts SET last_sent_utc=?, last_price=?, confirms=? WHERE key=?",
                    (now.isoformat(), price, confirms, key))
        conn.commit()
    conn.close()
    return ok

# ==============================
#   Telegram helpers
# ==============================

def tg_get_chat_id_from_updates(token: str) -> Optional[int]:
    try:
        r = requests.get(f"https://api.telegram.org/bot{token}/getUpdates", timeout=7)
        if r.status_code == 200:
            j = r.json()
            if j.get("ok") and j.get("result"):
                # берём последний апдейт
                for upd in reversed(j["result"]):
                    msg = upd.get("message") or upd.get("edited_message") or {}
                    chat = msg.get("chat") or {}
                    cid = chat.get("id")
                    if cid: return int(cid)
    except Exception:
        pass
    return None

def tg_send_message(token: str, chat_id: int, text: str):
    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        requests.post(url, json={"chat_id": chat_id, "text": text, "parse_mode": "HTML"}, timeout=7)
    except Exception:
        pass

def tg_ensure_chat_id() -> Optional[int]:
    cid = settings_get("telegram_chat_id")
    if cid:
        try:
            return int(cid)
        except Exception:
            pass
    # Попытка авто-детекта (нужно написать боту /start)
    cid = tg_get_chat_id_from_updates(TELEGRAM_BOT_TOKEN)
    if cid:
        settings_set("telegram_chat_id", str(cid))
        return cid
    return None

# ==============================
#   Utils & Indicators
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

def price_decimals(asset: str, price: float) -> int:
    # приближённо, чтобы не было "1.16 везде"
    if asset in ("EURUSD", "XAUEUR"):
        return 5
    if asset == "XAUUSD":
        return 2
    if price < 10:
        return 5
    if price < 100:
        return 3
    if price < 10000:
        return 2
    return 1

def fmt_price(asset: str, x: float) -> str:
    d = price_decimals(asset, x)
    return f"{x:.{d}f}"

# ==============================
#   SMC primitives
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
            res["high"].append((t, float(level)))
    for t in rec[SL.loc[rec.index]].index:
        level = df.loc[t, "low"]; post = rec[rec.index > t]
        if len(post[(post["low"] < level) & (post["close"] > level)]):
            res["low"].append((t, float(level)))
    return res

def fvg(df, look=140):
    out = {"bull": [], "bear": []}
    hi = df["high"].values; lo = df["low"].values; idx = df.index
    n = len(df); s = max(2, n - look)
    for i in range(s, n):
        if i - 2 >= 0 and lo[i] > hi[i - 2]:
            out["bull"].append((idx[i], float(hi[i - 2]), float(lo[i])))
        if i - 2 >= 0 and hi[i] < lo[i - 2]:
            out["bear"].append((idx[i], float(hi[i]), float(lo[i - 2])))
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
#    Bias & Regime
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
#   Confirm Engine (3-of-N)
# ==============================

@dataclass
class ConfirmPack:
    side: str                # "long" / "short"
    types: List[str]         # ["sweep","bos","imbalance",...]
    extra: List[str]         # доп.факторы (EMA/VWAP/VP, etc.)
    fvg_mid: Optional[float] # средина релевантного FVG для входа
    fvg_edges: Optional[Tuple[float, float]]  # (low_edge, high_edge) для стопа

def confirm_engine(df: pd.DataFrame) -> List[ConfirmPack]:
    packs: List[ConfirmPack] = []
    price = float(df["close"].iloc[-1])
    at = float(atr(df).iloc[-1]); at = max(at, 1e-6)
    SH, SL = swings(df)
    dir_bos, t_bos, _ = bos(df, SH, SL)
    gaps = fvg(df)
    swp = sweeps(df, SH, SL)

    # imbalance near/active
    bull_imb = list(reversed(gaps["bull"]))[:3]
    bear_imb = list(reversed(gaps["bear"]))[:3]

    # last relevant FVG (we need mid for entry)
    def last_mid_and_edges(kind: str) -> Tuple[Optional[float], Optional[Tuple[float,float]]]:
        arr = bull_imb if kind == "long" else bear_imb
        if not arr: return None, None
        _, a, b = arr[0]  # bull: (t, hi[i-2], lo[i]) mid of gap = (hi + lo)/2
        lo, hi = (a, b) if kind == "long" else (b, a)
        mid = (lo + hi) / 2.0
        return mid, (lo, hi)

    # LONG side confirmations
    conf_long: List[str] = []
    if swp["low"]: conf_long.append("sweep")
    if dir_bos == "up": conf_long.append("bos")
    if bull_imb: conf_long.append("imbalance")
    # breaker: если был свип low и затем закрытие выше уровня свипа (упрощённо)
    if swp["low"]:
        _, lvl = swp["low"][-1]
        if df["close"].iloc[-1] > lvl and dir_bos == "up":
            conf_long.append("breaker")
    # OB-retest упрощённо: после BOS up есть demand-OB и цена рядом
    ob = simple_ob(df, dir_bos, t_bos)
    if ob.get("demand"):
        _, lo, hi = ob["demand"]
        if abs(price - hi) <= 0.6 * at:
            conf_long.append("ob_retest")

    mid_long, edges_long = last_mid_and_edges("long")

    # SHORT side confirmations
    conf_short: List[str] = []
    if swp["high"]: conf_short.append("sweep")
    if dir_bos == "down": conf_short.append("bos")
    if bear_imb: conf_short.append("imbalance")
    if swp["high"]:
        _, lvl = swp["high"][-1]
        if df["close"].iloc[-1] < lvl and dir_bos == "down":
            conf_short.append("breaker")
    if ob.get("supply"):
        _, lo, hi = ob["supply"]
        if abs(price - lo) <= 0.6 * at:
            conf_short.append("ob_retest")

    mid_short, edges_short = last_mid_and_edges("short")

    # Доп. контекст (не считаются как подтверждения ядра)
    extra: List[str] = []
    vw = float(vwap_series(df).iloc[-1])
    e20 = float(ema(df["close"], 20).iloc[-1])
    if price > vw: extra.append("above VWAP")
    else: extra.append("below VWAP")
    if price > e20: extra.append("above EMA20")
    else: extra.append("below EMA20")

    packs.append(ConfirmPack("long", conf_long, extra, mid_long, edges_long))
    packs.append(ConfirmPack("short", conf_short, extra, mid_short, edges_short))
    return packs

def confirm_rule_ok(types: List[str]) -> Tuple[bool, int]:
    """
    Требование пользователя:
    - нужно 3 подтверждения
    - лучше разного типа (sweep+bos+imb), НО допустимо sweep+imb+imb
    """
    n = len(types)
    if n < 3:
        return False, n
    uniq = set(types)
    if len(uniq) >= 3:
        return True, n
    # иначе разрешим кейс с двойным IMB
    if types.count("imbalance") >= 2 and "sweep" in types:
        return True, n
    return False, n

# ==============================
#   RR / Targets
# ==============================

def rr_targets(entry: float, sl: float, bias: str, min_rr: float = 2.0) -> Tuple[float, float, float]:
    risk = abs(entry - sl) or 1e-6
    if bias == "long":
        tp1 = entry + min_rr * risk; tp2 = entry + 3.0 * risk
    else:
        tp1 = entry - min_rr * risk; tp2 = entry - 3.0 * risk
    rr = round(abs(tp1 - entry) / risk, 2)
    return tp1, tp2, rr

# ==============================
#   Data via yfinance (fallback)
# ==============================

@st.cache_data(show_spinner=False, ttl=60)
def yf_ohlc_first_success(asset_key: str, tf: str, limit: int = 800) -> Tuple[pd.DataFrame, str, str]:
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
                    if c not in df.columns: df[c] = 0.0 if c == "volume" else np.nan
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
#             UI
# ==============================

st.set_page_config(page_title="SMC Intraday (text)", layout="wide")
st.title("SMC Intraday — BTC / ETH / XAUUSD / XAUEUR / EURUSD (text)")

db_init()

tab_an, tab_jrnl, tab_cfg = st.tabs(["Анализ", "Журнал", "Настройки"])

with tab_cfg:
    st.subheader("Telegram")
    cid = settings_get("telegram_chat_id")
    st.write("Chat ID:", cid if cid else "пока не определён")
    if st.button("Попробовать autodetect chat_id через getUpdates"):
        cid_new = tg_ensure_chat_id()
        if cid_new: st.success(f"Нашёл chat_id: {cid_new}")
        else: st.warning("Не удалось. Напишите боту /start и попробуйте снова.")
    st.caption("Уведомление отправляется при РОВНО 2 подтверждениях (антиспам: не чаще раза в 30 минут для одной стороны).")

with tab_an:
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
        it = st.selectbox("Auto-refresh", ["Выкл", "30s", "1m", "2m", "5m"], index=0)
    with colF:
        beginner_mode = st.checkbox("Простой режим (для новичка)", value=True)

    if it != "Выкл":
        st.markdown(f"<meta http-equiv='refresh' content='{ {'30s':30,'1m':60,'2m':120,'5m':300}[it] }'>",
                    unsafe_allow_html=True)

    st.caption("Входы строятся строго от середины последнего актуального FVG по направлению подтверждений. Отображаются только сетапы с ≥3 подтверждениями ядра.")

    try:
        # Data
        df, tf_eff, _ = yf_ohlc_first_success(asset, tf, limit=800)
        df_h, _, _ = yf_ohlc_first_success(asset, HTF_OF[tf], limit=400)
        df_d, _, _ = yf_ohlc_first_success(asset, "1h", limit=24 * 200)  # дневку суррогируем из 1h

        price = float(df["close"].iloc[-1])
        decs = price_decimals(asset, price)
        vp = volume_profile(df)
        reg = market_regime(df, vp)
        atr_v = float(atr(df).iloc[-1])

        ltf_b = score_bias(df); htf_b = score_bias(df_h); d_b = regime_daily(df_d)

        st.markdown(
            f"**{asset} ({tf}) — цена:** {fmt_price(asset, price)}  \n"
            f"**Контекст:** LTF={ltf_b.upper()}, HTF={htf_b.upper()}, Daily={d_b.upper()} • "
            f"Режим: {reg.upper()} (ADX≈{float(adx(df).iloc[-1]):.1f}) • "
            f"VP: POC {fmt_price(asset, vp['poc'])}, VAL {fmt_price(asset, vp['val'])}, VAH {fmt_price(asset, vp['vah'])}"
        )

        # Confirm engine
        packs = confirm_engine(df)

        rows = []
        made_any = False

        for pk in packs:
            # Telegram alert if exactly 2 confirmations
            if len(pk.types) == 2:
                if alert_should_send(asset, tf, pk.side, price, 2):
                    cid = tg_ensure_chat_id()
                    if cid:
                        text = (f"⚠️ {asset} {tf} — {pk.side.upper()}\n"
                                f"Подтв.: {', '.join(pk.types)} (2/3)\n"
                                f"Цена: {fmt_price(asset, price)}")
                        tg_send_message(TELEGRAM_BOT_TOKEN, cid, text)

            ok, nconf = confirm_rule_ok(pk.types)
            if not ok:  # меньше 3 (или неподходящая комбинация) — пропускаем
                continue

            # Только вход от mid-FVG. Если FVG нет — пропускаем.
            if pk.fvg_mid is None or pk.fvg_edges is None:
                continue

            entry = pk.fvg_mid
            # стоп за дальний край FVG с небольшим буфером
            lo, hi = pk.fvg_edges
            if pk.side == "long":
                sl = min(lo, hi) - 0.2 * atr_v
            else:
                sl = max(lo, hi) + 0.2 * atr_v

            tp1, tp2, rr_val = rr_targets(entry, sl, pk.side, min_rr=2.0)

            # фильтры
            risk_ok = abs(entry - sl) >= atr_v * (min_risk_pct / 100.0)
            tp1_ok = (abs(tp1 - entry) / max(atr_v, 1e-6)) >= min_tp1_atr
            if not (risk_ok and tp1_ok):
                continue

            made_any = True

            # Доп. описание
            expl = []
            expl.append(f"Вход по mid-FVG (0.5). Подтверждения ядра: {', '.join(pk.types)}.")
            expl.append(f"Стоп за дальний край FVG ±0.2×ATR. Доп. контекст: {', '.join(pk.extra)}.")
            scenario_text = " ".join(expl)

            # Карточка новичка
            if beginner_mode:
                st.markdown(f"### Что сделать сейчас — {'ПОКУПКА (LONG)' if pk.side=='long' else 'ПРОДАЖА (SHORT)'}")
                st.markdown(
                    "- **Почему:** " + scenario_text + "\n"
                    f"- **Вход:** {fmt_price(asset, entry)}  \n"
                    f"- **Стоп:** {fmt_price(asset, sl)} (риск≈{fmt_price(asset, abs(entry-sl))}, ATR≈{fmt_price(asset, atr_v)})  \n"
                    f"- **Цели:** TP1 {fmt_price(asset, tp1)} (R:R≈{rr_val}), TP2 {fmt_price(asset, tp2)}  \n"
                    f"- **Подтв.:** {', '.join(pk.types)} (всего {len(pk.types)})"
                )

            # строка в таблицу
            rows.append({
                "Сторона": pk.side,
                "Подтв. (ядро)": ", ".join(pk.types),
                "Вход": fmt_price(asset, entry),
                "Стоп": fmt_price(asset, sl),
                "TP1": fmt_price(asset, tp1),
                "TP2": fmt_price(asset, tp2),
                "R:R до TP1": rr_val,
                "Доп. факторы": ", ".join(pk.extra),
            })

            # кнопка в журнал
            with st.container():
                c1, c2, c3 = st.columns([1, 1, 6])
                with c1:
                    if st.button(f"Добавить в журнал ({pk.side})", key=f"add_{pk.side}_{len(rows)}"):
                        context = {
                            "regime": reg, "atr": atr_v,
                            "bias_ltf": ltf_b, "bias_htf": htf_b, "bias_d": d_b,
                            "poc": vp["poc"], "val": vp["val"], "vah": vp["vah"]
                        }
                        journal_insert(asset, tf, pk.side, float(entry), float(sl), float(tp1), float(tp2),
                                       float(rr_val), pk.types, context, scenario_text)
                        st.success("Сделка добавлена в журнал.")
                with c2:
                    st.write("")

        if rows:
            st.markdown("### Все валидные варианты")
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        if not made_any:
            st.info("Сейчас нет валидных (3-подтв.) входов от mid-FVG. Подождите формирования условий.")

    except Exception as e:
        st.error(f"{asset}: {e}")

with tab_jrnl:
    st.subheader("Журнал сделок (SQLite)")
    dfj = journal_read_df()
    if dfj.empty:
        st.info("Пока пусто. Добавляйте сделки из вкладки «Анализ».")
    else:
        # редактирование строк
        for _, row in dfj.iterrows():
            with st.expander(f"#{int(row['id'])} • {row['ts_utc']} • {row['asset']} {row['tf']} • {row['side'].upper()} • entry {fmt_price(row['asset'], row['entry'])}"):
                st.write(f"SL {fmt_price(row['asset'], row['sl'])} • TP1 {fmt_price(row['asset'], row['tp1'])} • TP2 {fmt_price(row['asset'], row['tp2']) if not pd.isna(row['tp2']) else '—'} • RR≈{row['rr']}")
                st.write("Подтв.:", ", ".join(json.loads(row["confirmations"])) if row["confirmations"] else "—")
                st.write("Сценарий:", row["scenario"])
                c1, c2, c3, c4 = st.columns([1, 1, 1, 3])
                with c1:
                    status = st.selectbox("Статус", ["open","win","stop","be","closed"], index=["open","win","stop","be","closed"].index(row["status"]), key=f"st_{row['id']}")
                with c2:
                    res_r = st.number_input("Result R", value=float(row["result_r"]) if row["result_r"] is not None else 0.0, step=0.1, key=f"rr_{row['id']}")
                with c3:
                    if st.button("Сохранить", key=f"sv_{row['id']}"):
                        journal_update_row(int(row["id"]), status, float(res_r), row.get("notes") or "")
                        st.success("Сохранено.")
                with c4:
                    new_notes = st.text_area("Заметки/пояснение (почему win/stop)", value=row.get("notes") or "", key=f"nt_{row['id']}")
                    if new_notes != (row.get("notes") or ""):
                        journal_update_row(int(row["id"]), status, float(res_r), new_notes)
                        st.toast("Заметка сохранена", icon="✍️")

        st.markdown("### Статистика")
        dfj = journal_read_df()
        total = len(dfj)
        wins = int((dfj["status"] == "win").sum())
        stops = int((dfj["status"] == "stop").sum())
        be = int((dfj["status"] == "be").sum())
        wr = (wins / max(wins + stops, 1)) * 100.0
        avg_r = float(dfj["result_r"].dropna().mean()) if (dfj["result_r"].notna().any()) else 0.0
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Всего", total)
        c2.metric("Win", wins)
        c3.metric("Stop", stops)
        c4.metric("Win-rate", f"{wr:.1f}%")
        c5.metric("Средн. R", f"{avg_r:.2f}")

        st.markdown("#### Разбивка по активу/стороне")
        br = dfj.groupby(["asset","side"])["status"].value_counts().unstack(fill_value=0)
        st.dataframe(br, use_container_width=True)
