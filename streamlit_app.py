import os
import math
import json
import time
import requests
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ---------- UI ----------
st.set_page_config(page_title="Gamma / GEX Levels", layout="wide")
st.title("GEX Levels & Magnet Profile (по комбинированной методике)")

# Секреты/ввод
host_default = st.secrets.get("RAPIDAPI_HOST", "")
key_default  = st.secrets.get("RAPIDAPI_KEY", "")

with st.sidebar:
    st.header("Провайдер (RapidAPI)")
    RAPIDAPI_HOST = st.text_input("X-RapidAPI-Host", host_default, placeholder="yahoo-finance15.p.rapidapi.com")
    RAPIDAPI_KEY  = st.text_input("X-RapidAPI-Key", key_default, type="password")
    st.caption("Можно хранить здесь или в Secrets Streamlit (рекомендовано).")
    st.divider()
    ticker = st.text_input("Тикер", value="SPY")
    btn_load = st.button("Загрузить экспирации")

# ---------- Параметры методики (можно менять) ----------
SECONDS_PER_YEAR = 31557600.0
DEFAULT_R = 0.01  # r
DEFAULT_Q = 0.00  # q
H_EXP = 7.0       # полупериод для W_exp
KAPPA = 1.0       # kappa для W_dist
SMOOTH_WINDOW = 3 # сглаживание по страйку (число страйков)
TOP_N_LEVELS = 5  # сколько магнитов показывать с каждой стороны

# ---------- API helpers ----------
def api_headers():
    return {
        "X-RapidAPI-Key": RAPIDAPI_KEY,
        "X-RapidAPI-Host": RAPIDAPI_HOST
    }

def fetch_chain_raw(symbol: str):
    """
    Гибкий фетч опционной цепочки для тикера.
    Для yahoo-finance15.p.rapidapi.com обычно:
    GET https://yahoo-finance15.p.rapidapi.com/api/yahoo/option/{symbol}
    """
    url = f"https://{RAPIDAPI_HOST}/api/yahoo/option/{symbol}"
    r = requests.get(url, headers=api_headers(), timeout=20)
    r.raise_for_status()
    return r.json()

def ensure_chain_shape(raw: dict):
    """
    Приводим JSON к унифицированному виду:
    {
       'quote': {... 'regularMarketPrice', 'regularMarketTime' ...},
       'expirationDates': [epoch,...],
       'chains': [ { 'expiration': epoch, 'calls': [...], 'puts': [...] }, ... ]
    }
    """
    quote = raw.get("quote", {})
    expirationDates = raw.get("expirationDates", [])
    chains = []
    # Вариант yahoo-finance15: "chains[0]" одна серия, остальные даты в 'expirationDates'
    if "chains[0]" in raw and isinstance(raw["chains[0]"], dict):
        c0 = raw["chains[0]"]
        chains.append({
            "expiration": c0.get("expiration"),
            "calls": c0.get("calls", []),
            "puts":  c0.get("puts",  [])
        })
    # Вариант, когда бывает "options" или "chains" списком
    if "options" in raw and isinstance(raw["options"], list):
        for ch in raw["options"]:
            if isinstance(ch, dict):
                chains.append({
                    "expiration": ch.get("expiration"),
                    "calls": ch.get("calls", []),
                    "puts":  ch.get("puts",  [])
                })
    if "chains" in raw and isinstance(raw["chains"], list):
        for ch in raw["chains"]:
            if isinstance(ch, dict):
                chains.append({
                    "expiration": ch.get("expiration"),
                    "calls": ch.get("calls", []),
                    "puts":  ch.get("puts",  [])
                })
    # Удалим дубли и отсортируем
    seen = set()
    norm_chains = []
    for ch in chains:
        exp = ch.get("expiration")
        sig = (exp, len(ch.get("calls", []))+len(ch.get("puts", [])))
        if exp is None or sig in seen:
            continue
        seen.add(sig)
        norm_chains.append(ch)
    norm_chains.sort(key=lambda x: x.get("expiration", 0))
    return {
        "quote": quote,
        "expirationDates": expirationDates,
        "chains": norm_chains
    }

def fetch_specific_expiry(symbol: str, epoch: int):
    """
    Подтянуть конкретную экспирацию: у некоторых провайдеров нужно добавить ?date=epoch
    """
    url = f"https://{RAPIDAPI_HOST}/api/yahoo/option/{symbol}?date={epoch}"
    r = requests.get(url, headers=api_headers(), timeout=20)
    r.raise_for_status()
    raw = r.json()
    shaped = ensure_chain_shape(raw)
    # выберем цепь, где expiration == epoch
    for ch in shaped["chains"]:
        if ch.get("expiration") == epoch:
            # добавим quote на всякий случай
            return {
                "quote": shaped.get("quote", {}),
                "chain": ch
            }
    # fallback: если провайдер вернул только одну серию
    if shaped["chains"]:
        return {
            "quote": shaped.get("quote", {}),
            "chain": shaped["chains"][0]
        }
    raise RuntimeError("Не удалось найти цепочку для заданной даты.")

# ---------- Математика ----------
def bsm_gamma(S, K, sigma, tau, r=DEFAULT_R, q=DEFAULT_Q):
    if tau <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0.0
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
    phi = np.exp(-0.5 * d1**2) / np.sqrt(2 * np.pi)
    gamma = phi * np.exp(-q * tau) / (S * sigma * np.sqrt(tau))
    return float(gamma)

def compute_chain_gex(chain: dict, quote: dict):
    """
    Возвращает DataFrame c колонками:
    ['strike','type','oi','iv','tau','gex_signed']
    gex_signed: для call положительный, для put — вычитаем при свёртке Net GEX.
    """
    S = quote.get("regularMarketPrice")
    t0 = quote.get("regularMarketTime")
    exp = chain.get("expiration")
    calls = chain.get("calls", [])
    puts  = chain.get("puts", [])
    if S is None or t0 is None or exp is None:
        raise RuntimeError("Недостаточно данных 'quote/expiration'.")

    tau = max((exp - t0) / SECONDS_PER_YEAR, 0.0)
    rows = []
    for row in calls:
        K   = row.get("strike")
        oi  = row.get("openInterest", 0) or 0
        iv  = row.get("impliedVolatility", 0) or 0
        cs  = row.get("contractSize", "REGULAR")
        mult = 100 if str(cs).upper() == "REGULAR" else 100
        gamma = bsm_gamma(S, K, iv, tau)
        gex   = oi * gamma * mult * S
        rows.append({"strike":K, "type":"call", "oi":oi, "iv":iv, "tau":tau, "gex_signed": gex})
    for row in puts:
        K   = row.get("strike")
        oi  = row.get("openInterest", 0) or 0
        iv  = row.get("impliedVolatility", 0) or 0
        cs  = row.get("contractSize", "REGULAR")
        mult = 100 if str(cs).upper() == "REGULAR" else 100
        gamma = bsm_gamma(S, K, iv, tau)
        gex   = oi * gamma * mult * S
        rows.append({"strike":K, "type":"put", "oi":oi, "iv":iv, "tau":tau, "gex_signed": -gex})
    df = pd.DataFrame(rows)
    return df, S

def weight_scheme(df_all_exp: pd.DataFrame, S: float, h=H_EXP, kappa=KAPPA):
    """
    Добавляет веса W_exp, W_liq, W_dist к каждой записи.
    df_all_exp: объединённый DF по нескольким экспирациям
      колонки: strike, type, oi, iv, tau, gex_signed, expiry (epoch)
    """
    # W_exp
    df = df_all_exp.copy()
    df["DTE"] = df["tau"] * 365.0
    df["W_exp"] = 2.0 ** (-df["DTE"] / h)

    # Для W_liq посчитаем суммарный OI и Volume по каждой экспирации.
    # (volume не у всех провайдеров стабильно есть — используем, если есть)
    if "volume" in df.columns:
        vol_by_exp = df.groupby("expiry")["volume"].sum().rename("exp_vol")
        df = df.merge(vol_by_exp, on="expiry", how="left")
    else:
        df["exp_vol"] = 0.0

    oi_by_exp = df.groupby("expiry")["oi"].sum().rename("exp_oi")
    df = df.merge(oi_by_exp, on="expiry", how="left")

    total_oi  = df["oi"].sum()
    total_vol = df["exp_vol"].sum()

    df["OI_share"]  = np.where(total_oi>0,  df["exp_oi"]/total_oi, 0.0)
    df["Vol_share"] = np.where(total_vol>0, df["exp_vol"]/total_vol, 0.0)
    df["W_liq"] = np.sqrt(df["OI_share"]) * np.sqrt(1.0 + df["Vol_share"])

    # W_dist
    # Возьмём sigma как iv строки; если нужно, можно заменить на ATM iv по экспирации
    sig = df["iv"].clip(lower=1e-6)
    root_tau = np.sqrt(df["tau"].clip(lower=1e-9))
    denom = 2.0 * (kappa**2) * (sig * root_tau)**2
    log_term = np.log(np.maximum(df["strike"], 1e-6) / max(S, 1e-6))
    df["W_dist"] = np.exp(- (log_term**2) / np.maximum(denom, 1e-12))

    return df

def build_profiles(df_w: pd.DataFrame, S: float, smooth_window=SMOOTH_WINDOW):
    """
    Строим NetGEX_raw(K) и Magnet(K) = sum_raw * W / S, где W = W_exp*W_liq*W_dist по каждой записи.
    """
    # raw Net GEX по страйку
    raw = df_w.groupby("strike")["gex_signed"].sum().rename("NetGEX_raw").reset_index()

    # итоговый вес по записи
    df_w["W_prod"] = df_w["W_exp"] * df_w["W_liq"] * df_w["W_dist"]
    df_w["contrib"] = df_w["gex_signed"] * df_w["W_prod"] / S

    # Magnet
    magnet = df_w.groupby("strike")["contrib"].sum().rename("Magnet").reset_index()

    prof = raw.merge(magnet, on="strike", how="outer").sort_values("strike")
    # сглаживание по страйку (простое скользящее)
    prof["NetGEX_smooth"] = prof["NetGEX_raw"].rolling(smooth_window, center=True, min_periods=1).mean()
    prof["Magnet_smooth"] = prof["Magnet"].rolling(smooth_window, center=True, min_periods=1).mean()
    return prof

def find_levels(profile: pd.DataFrame):
    """
    Находим:
      - основные магниты (локальные экстремумы по |Magnet_smooth|)
      - уровни нулевой гаммы (пересечения Magnet_smooth через 0)
    """
    prof = profile.dropna(subset=["Magnet_smooth"]).copy()
    strikes = prof["strike"].values
    vals = prof["Magnet_smooth"].values

    # Нулевая гамма — пересечения знака
    flips = []
    for i in range(1, len(vals)):
        if vals[i-1] == 0: 
            continue
        if (vals[i-1] > 0 and vals[i] < 0) or (vals[i-1] < 0 and vals[i] > 0):
            # линейная интерполяция
            x0, x1 = strikes[i-1], strikes[i]
            y0, y1 = vals[i-1], vals[i]
            if (y1 - y0) != 0:
                x_cross = x0 + (x1 - x0) * (-y0) / (y1 - y0)
                flips.append(x_cross)
            else:
                flips.append((x0 + x1)/2)

    # Магниты — локальные экстремумы |Magnet|
    mags = []
    absvals = np.abs(vals)
    for i in range(1, len(vals)-1):
        if absvals[i] >= absvals[i-1] and absvals[i] >= absvals[i+1]:
            mags.append((strikes[i], vals[i], absvals[i]))
    # Отсортируем по силе и возьмём топ-N по каждой стороне
    pos = sorted([(k,v,a) for (k,v,a) in mags if v>0], key=lambda x: x[2], reverse=True)[:TOP_N_LEVELS]
    neg = sorted([(k,v,a) for (k,v,a) in mags if v<0], key=lambda x: x[2], reverse=True)[:TOP_N_LEVELS]
    return flips, pos, neg

def plot_profiles(profile: pd.DataFrame, S: float, flips, pos, neg, title_note=""):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=profile["strike"], y=profile["NetGEX_smooth"],
                             name="Net GEX (сглаж.)", mode="lines"))
    fig.add_trace(go.Scatter(x=profile["strike"], y=profile["Magnet_smooth"],
                             name="Magnet (взвеш., сглаж.)", mode="lines"))
    # Нулевая линия
    fig.add_hline(y=0, line_dash="dot", line_color="gray", annotation_text="Zero Gamma")
    # Spot
    fig.add_vline(x=S, line_dash="dot", line_color="red", annotation_text=f"Spot {S:.2f}")

    # Отметим флипы
    for x in flips:
        fig.add_vline(x=x, line_dash="dash", line_color="#f1c40f", annotation_text="Gamma Flip")

    # Отметим топ-магниты
    for (k,v,a) in pos:
        fig.add_scatter(x=[k], y=[v], mode="markers", marker=dict(size=10), name=f"+Magnet {k}")
    for (k,v,a) in neg:
        fig.add_scatter(x=[k], y=[v], mode="markers", marker=dict(size=10), name=f"-Magnet {k}")

    fig.update_layout(title=f"Профиль GEX/Magnet {title_note}", xaxis_title="Strike", yaxis_title="Value", height=520)
    return fig

# ---------- Основной поток ----------
if btn_load:
    if not RAPIDAPI_HOST or not RAPIDAPI_KEY:
        st.error("Укажи RapidAPI Host и Key (в сайдбаре).")
        st.stop()
    try:
        raw = fetch_chain_raw(ticker)
        shaped = ensure_chain_shape(raw)
        quote = shaped["quote"]
        exp_dates = shaped.get("expirationDates") or [c.get("expiration") for c in shaped["chains"]]
        exp_dates = [e for e in exp_dates if isinstance(e, int)]
        exp_dates = sorted(list(set(exp_dates)))
        if not exp_dates:
            st.error("Не удалось получить список экспираций.")
            st.stop()
        # Выбор ближайшей
        human = [time.strftime("%Y-%m-%d", time.gmtime(e)) for e in exp_dates]
        idx = st.selectbox("Выбери ближайшую экспирацию", list(range(len(exp_dates))), 
                           format_func=lambda i: human[i], index=0)
        calc = st.button("Рассчитать уровни (эта + 7 следующих)")
        if calc:
            picked = exp_dates[idx: idx+8]  # ближайшая + 7 следующих
            all_rows = []
            S_ref = None
            for e in picked:
                dat = fetch_specific_expiry(ticker, e)
                quote_i = dat["quote"]
                chain_i = dat["chain"]
                df_i, S_i = compute_chain_gex(chain_i, quote_i)
                df_i["expiry"] = e
                # возьмем volume, если есть
                # (в некоторых данных volume лежит в самих рядах; для простоты оставим как есть)
                all_rows.append(df_i)
                if S_ref is None and S_i is not None:
                    S_ref = S_i
            if not all_rows or S_ref is None:
                st.error("Недостаточно данных для расчёта.")
                st.stop()

            df_all = pd.concat(all_rows, ignore_index=True)

            # Добавим weights
            df_w = weight_scheme(df_all, S=S_ref, h=H_EXP, kappa=KAPPA)

            # Построим профили
            prof = build_profiles(df_w, S=S_ref, smooth_window=SMOOTH_WINDOW)

            # Найдём уровни
            flips, pos, neg = find_levels(prof)

            # График
            col1, col2 = st.columns([2,1])
            with col1:
                fig = plot_profiles(prof, S=S_ref, flips=flips, pos=pos, neg=neg,
                                    title_note=f"({ticker}, {time.strftime('%Y-%m-%d', time.gmtime(picked[0]))} +7)")
                st.plotly_chart(fig, use_container_width=True)

            # Таблица ключевых уровней
            def rows_from_mags(mags, sign_label):
                out = []
                for (k,v,a) in mags:
                    out.append({"Strike": float(k), "Magnet (взвеш.)": float(v), "Сила |Magnet|": float(a), "Сторона": sign_label})
                return out

            rows = rows_from_mags(pos, "+") + rows_from_mags(neg, "-")
            levels_df = pd.DataFrame(rows).sort_values("Сила |Magnet|", ascending=False)
            with col2:
                st.subheader("Ключевые уровни (магниты)")
                st.dataframe(levels_df, use_container_width=True)
                st.download_button("Скачать уровни (CSV)",
                                   data=levels_df.to_csv(index=False).encode("utf-8"),
                                   file_name=f"{ticker}_magnet_levels.csv",
                                   mime="text/csv")

            # Текстовый «чек-лист» на день
            st.subheader("Интрадей план (по методике)")
            plan = []
            if flips:
                flip_zone = f"{min(flips):.2f}–{max(flips):.2f}" if len(flips)>1 else f"{flips[0]:.2f}"
                plan.append(f"Нулевая гамма (flip): {flip_zone}. Ниже — режим mean-revert, выше — breakout.")
            if pos:
                plan.append(f"Главные магниты (+): {', '.join(str(round(k)) for (k,_,__) in pos[:3])}. Торговать возвраты к ним при MG>0.")
            if neg:
                plan.append(f"Главные магниты (–): {', '.join(str(round(k)) for (k,_,__) in neg[:3])}. При закреплении над flip — цели по пробою.")
            st.markdown("- " + "\n- ".join(plan))

    except Exception as e:
        st.exception(e)
