# streamlit_app.py
import os
import time
import json
import math
import requests
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# =============== Настройки страницы ===============
st.set_page_config(page_title="GEX Levels & Magnet Profile", layout="wide")
st.title("GEX Levels & Magnet Profile (по комбинированной методике)")

# =============== Константы методики ===============
SECONDS_PER_YEAR = 31557600.0
DEFAULT_R = 0.01   # r
DEFAULT_Q = 0.00   # q
H_EXP = 7.0        # полупериод для W_exp
KAPPA = 1.0        # параметр достижимости
SMOOTH_WINDOW = 3  # сглаживание по страйку
TOP_N_LEVELS = 5   # сколько топ-магнитов показывать на сторону

# =============== Чтение Secrets / UI ввода ===============
host_default = st.secrets.get("RAPIDAPI_HOST", "")
key_default  = st.secrets.get("RAPIDAPI_KEY", "")

with st.sidebar:
    st.header("Провайдер (RapidAPI)")
    RAPIDAPI_HOST = st.text_input("X-RapidAPI-Host", host_default, placeholder="yahoo-finance15.p.rapidapi.com")
    RAPIDAPI_KEY  = st.text_input("X-RapidAPI-Key", key_default, type="password")
    st.caption("Можно хранить здесь или в Secrets Streamlit (рекомендовано).")
    st.divider()
    ticker = st.text_input("Тикер", value="SPY")
    colb1, colb2 = st.columns(2)
    btn_load = colb1.button("Загрузить экспирации")
    # дополнительные настройки (по желанию можно скрыть)
    with st.expander("Параметры методики", expanded=False):
        H_EXP = st.slider("h (вес экспирации, дней)", 3.0, 14.0, H_EXP, 0.5)
        KAPPA = st.slider("κ (достижимость)", 0.5, 2.0, KAPPA, 0.1)
        SMOOTH_WINDOW = st.select_slider("Сглаживание по страйку (кол-во)", options=[1,3,5,7], value=SMOOTH_WINDOW)

# =============== Хелперы API (универсальные маршруты) ===============
def api_headers():
    return {"X-RapidAPI-Key": RAPIDAPI_KEY, "X-RapidAPI-Host": RAPIDAPI_HOST}

def _try_get(url: str):
    r = requests.get(url, headers=api_headers(), timeout=25)
    if r.status_code == 200:
        try:
            return r.json()
        except Exception as e:
            raise requests.HTTPError(f"Bad JSON for {url}") from e
    raise requests.HTTPError(f"{r.status_code} {url}\n{r.text[:400]}")

def fetch_chain_raw(symbol: str):
    """
    Пробуем несколько популярных путей у провайдера 'Yahoo Finance 15 via RapidAPI'
    чтобы получить список экспираций/цепочку.
    """
    base = f"https://{RAPIDAPI_HOST}"
    candidates = [
        f"{base}/api/yahoo/option/{symbol}",
        f"{base}/api/yahoo/options/{symbol}",
        f"{base}/api/yahoo/option/{symbol}?date=0",
        f"{base}/api/yahoo/options/{symbol}?date=0",
        f"{base}/api/yahoo/v2/option/{symbol}",
        f"{base}/api/yahoo/v2/options/{symbol}",
    ]
    errors = []
    for url in candidates:
        try:
            return _try_get(url)
        except Exception as e:
            errors.append(str(e))
            continue
    raise RuntimeError(
        "Не удалось получить список экспираций у провайдера.\n"
        "Проверь host/ключ. Диагностика (первые 2 попытки):\n\n" + "\n\n".join(errors[:2])
    )

def ensure_chain_shape(raw: dict):
    """
    Приводим JSON к унифицированному виду:
    { 'quote': {...}, 'expirationDates': [epoch,...],
      'chains': [{ 'expiration': epoch, 'calls': [...], 'puts': [...] }, ...] }
    """
    quote = raw.get("quote", {})
    expirationDates = raw.get("expirationDates", [])
    chains = []

    if "chains[0]" in raw and isinstance(raw["chains[0]"], dict):
        c0 = raw["chains[0]"]
        chains.append({
            "expiration": c0.get("expiration"),
            "calls": c0.get("calls", []),
            "puts":  c0.get("puts",  [])
        })

    if isinstance(raw.get("options"), list):
        for ch in raw["options"]:
            if isinstance(ch, dict):
                chains.append({
                    "expiration": ch.get("expiration"),
                    "calls": ch.get("calls", []),
                    "puts":  ch.get("puts",  [])
                })

    if isinstance(raw.get("chains"), list):
        for ch in raw["chains"]:
            if isinstance(ch, dict):
                chains.append({
                    "expiration": ch.get("expiration"),
                    "calls": ch.get("calls", []),
                    "puts":  ch.get("puts",  [])
                })

    # дедуп и сортировка
    seen = set()
    norm = []
    for ch in chains:
        exp = ch.get("expiration")
        sig = (exp, len(ch.get("calls", [])) + len(ch.get("puts", [])))
        if exp is None or sig in seen:
            continue
        seen.add(sig)
        norm.append(ch)
    norm.sort(key=lambda x: x.get("expiration", 0))

    return {"quote": quote, "expirationDates": expirationDates, "chains": norm}

def fetch_specific_expiry(symbol: str, epoch: int):
    """
    Тянем конкретную дату (на большинстве маршрутов обязательный параметр ?date=)
    """
    base = f"https://{RAPIDAPI_HOST}"
    candidates = [
        f"{base}/api/yahoo/option/{symbol}?date={epoch}",
        f"{base}/api/yahoo/options/{symbol}?date={epoch}",
        f"{base}/api/yahoo/v2/option/{symbol}?date={epoch}",
        f"{base}/api/yahoo/v2/options/{symbol}?date={epoch}",
    ]
    errors = []
    for url in candidates:
        try:
            raw = _try_get(url)
            shaped = ensure_chain_shape(raw)
            for ch in shaped["chains"]:
                if ch.get("expiration") == epoch:
                    return {"quote": shaped.get("quote", {}), "chain": ch}
            # fallback — возвращаем первую серию, если метки нет
            if shaped["chains"]:
                return {"quote": shaped.get("quote", {}), "chain": shaped["chains"][0]}
        except Exception as e:
            errors.append(str(e))
            continue
    raise RuntimeError(
        "Не удалось получить цепочку для выбранной даты.\n" + "\n\n".join(errors[:2])
    )

# =============== Математика (BSM Gamma / профили) ===============
def bsm_gamma(S, K, sigma, tau, r=DEFAULT_R, q=DEFAULT_Q):
    if tau <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0.0
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
    phi = np.exp(-0.5 * d1**2) / np.sqrt(2 * np.pi)
    gamma = phi * np.exp(-q * tau) / (S * sigma * np.sqrt(tau))
    return float(gamma)

def compute_chain_gex(chain: dict, quote: dict):
    """
    Возвращает DF:
      ['strike','type','oi','iv','tau','gex_signed']
    gex_signed: call -> +, put -> -
    """
    S = quote.get("regularMarketPrice")
    t0 = quote.get("regularMarketTime")
    exp = chain.get("expiration")
    calls = chain.get("calls", [])
    puts  = chain.get("puts", [])
    if S is None or t0 is None or exp is None:
        raise RuntimeError("Недостаточно данных 'quote/expiration' в ответе провайдера.")
    tau = max((exp - t0) / SECONDS_PER_YEAR, 0.0)
    rows = []
    for row in calls:
        K   = row.get("strike")
        oi  = row.get("openInterest", 0) or 0
        iv  = row.get("impliedVolatility", 0) or 0
        cs  = (row.get("contractSize") or "REGULAR").upper()
        mult = 100 if cs == "REGULAR" else 100
        gamma = bsm_gamma(S, K, iv, tau)
        gex   = oi * gamma * mult * S
        rows.append({"strike":K, "type":"call", "oi":oi, "iv":iv, "tau":tau, "gex_signed": gex})
    for row in puts:
        K   = row.get("strike")
        oi  = row.get("openInterest", 0) or 0
        iv  = row.get("impliedVolatility", 0) or 0
        cs  = (row.get("contractSize") or "REGULAR").upper()
        mult = 100 if cs == "REGULAR" else 100
        gamma = bsm_gamma(S, K, iv, tau)
        gex   = oi * gamma * mult * S
        rows.append({"strike":K, "type":"put", "oi":oi, "iv":iv, "tau":tau, "gex_signed": -gex})
    df = pd.DataFrame(rows)
    return df, S

def weight_scheme(df_all_exp: pd.DataFrame, S: float, h=H_EXP, kappa=KAPPA):
    """
    Добавляет веса W_exp, W_liq, W_dist.
    """
    df = df_all_exp.copy()
    df["DTE"]   = df["tau"] * 365.0
    df["W_exp"] = 2.0 ** (-df["DTE"] / h)

    # Ликвидность по экспирации
    oi_by_exp = df.groupby("expiry")["oi"].sum().rename("exp_oi")
    df = df.merge(oi_by_exp, on="expiry", how="left")
    total_oi = float(df["oi"].sum()) or 1.0
    df["OI_share"] = df["exp_oi"] / total_oi
    # если volume в исходных рядах нет — работаем без него
    if "volume" in df.columns:
        vol_by_exp = df.groupby("expiry")["volume"].sum().rename("exp_vol")
        df = df.merge(vol_by_exp, on="expiry", how="left")
        total_vol = float(df["exp_vol"].sum()) or 0.0
        df["Vol_share"] = np.where(total_vol > 0, df["exp_vol"] / total_vol, 0.0)
    else:
        df["Vol_share"] = 0.0
    df["W_liq"] = np.sqrt(df["OI_share"].clip(lower=0)) * np.sqrt(1.0 + df["Vol_share"].clip(lower=0))

    # Достижимость (через ln(K/S) в единицах ожидаемого движения)
    sig = df["iv"].clip(lower=1e-6)
    root_tau = np.sqrt(df["tau"].clip(lower=1e-9))
    denom = 2.0 * (kappa**2) * (sig * root_tau)**2
    log_term = np.log(np.maximum(df["strike"], 1e-6) / max(S, 1e-6))
    df["W_dist"] = np.exp(- (log_term**2) / np.maximum(denom, 1e-12))

    return df

def build_profiles(df_w: pd.DataFrame, S: float, smooth_window=SMOOTH_WINDOW):
    """
    Строим NetGEX_raw(K) и Magnet(K) = sum( gex_signed * W_prod / S )
    """
    raw = df_w.groupby("strike")["gex_signed"].sum().rename("NetGEX_raw").reset_index()
    df_w["W_prod"] = df_w["W_exp"] * df_w["W_liq"] * df_w["W_dist"]
    df_w["contrib"] = df_w["gex_signed"] * df_w["W_prod"] / S
    magnet = df_w.groupby("strike")["contrib"].sum().rename("Magnet").reset_index()
    prof = raw.merge(magnet, on="strike", how="outer").sort_values("strike")
    prof["NetGEX_smooth"] = prof["NetGEX_raw"].rolling(smooth_window, center=True, min_periods=1).mean()
    prof["Magnet_smooth"] = prof["Magnet"].rolling(smooth_window, center=True, min_periods=1).mean()
    return prof

def find_levels(profile: pd.DataFrame):
    """
    Возвращает:
      flips — точки пересечения нуля (gamma flip),
      pos/neg — топ локальные экстремумы по |Magnet_smooth| (плюс/минус).
    """
    prof = profile.dropna(subset=["Magnet_smooth"]).copy()
    strikes = prof["strike"].values
    vals = prof["Magnet_smooth"].values

    # пересечения нулевой линии
    flips = []
    for i in range(1, len(vals)):
        y0, y1 = vals[i-1], vals[i]
        if y0 == 0: 
            continue
        if (y0 > 0 and y1 < 0) or (y0 < 0 and y1 > 0):
            x0, x1 = strikes[i-1], strikes[i]
            x_cross = x0 + (x1 - x0) * (-y0) / (y1 - y0) if (y1 - y0) != 0 else (x0 + x1)/2
            flips.append(x_cross)

    # локальные экстремумы по |Magnet|
    mags = []
    absvals = np.abs(vals)
    for i in range(1, len(vals)-1):
        if absvals[i] >= absvals[i-1] and absvals[i] >= absvals[i+1]:
            mags.append((strikes[i], vals[i], absvals[i]))
    pos = sorted([(k,v,a) for (k,v,a) in mags if v>0], key=lambda x: x[2], reverse=True)[:TOP_N_LEVELS]
    neg = sorted([(k,v,a) for (k,v,a) in mags if v<0], key=lambda x: x[2], reverse=True)[:TOP_N_LEVELS]
    return flips, pos, neg

def plot_profiles(profile: pd.DataFrame, S: float, flips, pos, neg, title_note=""):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=profile["strike"], y=profile["NetGEX_smooth"],
                             name="Net GEX (сглаж.)", mode="lines"))
    fig.add_trace(go.Scatter(x=profile["strike"], y=profile["Magnet_smooth"],
                             name="Magnet (взвеш., сглаж.)", mode="lines"))
    fig.add_hline(y=0, line_dash="dot", line_color="gray", annotation_text="Zero Gamma")
    fig.add_vline(x=S, line_dash="dot", line_color="red", annotation_text=f"Spot {S:.2f}")
    for x in flips:
        fig.add_vline(x=x, line_dash="dash", line_color="#f1c40f", annotation_text="Gamma Flip")
    for (k,v,a) in pos:
        fig.add_scatter(x=[k], y=[v], mode="markers", marker=dict(size=10), name=f"+Magnet {k}")
    for (k,v,a) in neg:
        fig.add_scatter(x=[k], y=[v], mode="markers", marker=dict(size=10), name=f"-Magnet {k}")
    fig.update_layout(title=f"Профиль GEX/Magnet {title_note}", xaxis_title="Strike", yaxis_title="Value", height=520)
    return fig

# =============== Основной поток ===============
if btn_load:
    if not RAPIDAPI_HOST or not RAPIDAPI_KEY:
        st.error("Укажи RapidAPI Host и Key (в сайдбаре или в Secrets).")
        st.stop()
    try:
        raw = fetch_chain_raw(ticker.strip().upper())
        shaped = ensure_chain_shape(raw)
        quote = shaped.get("quote", {})
        exp_dates = shaped.get("expirationDates") or [c.get("expiration") for c in shaped["chains"]]
        exp_dates = [e for e in exp_dates if isinstance(e, int)]
        exp_dates = sorted(list(set(exp_dates)))
        if not exp_dates:
            st.error("Не удалось получить список экспираций (пустой ответ).")
            st.stop()

        human = [time.strftime("%Y-%m-%d", time.gmtime(e)) for e in exp_dates]
        picked_idx = st.selectbox("Выбери ближайшую экспирацию", list(range(len(exp_dates))),
                                  format_func=lambda i: human[i], index=0)
        if st.button("Рассчитать уровни (эта + 7 следующих)"):
            picked = exp_dates[picked_idx: picked_idx + 8]  # ближайшая +7
            all_rows = []
            S_ref = None
            for e in picked:
                dat = fetch_specific_expiry(ticker.strip().upper(), e)
                quote_i = dat["quote"]
                chain_i = dat["chain"]
                df_i, S_i = compute_chain_gex(chain_i, quote_i)
                df_i["expiry"] = e
                all_rows.append(df_i)
                if S_ref is None and S_i is not None:
                    S_ref = S_i
            if not all_rows or S_ref is None:
                st.error("Недостаточно данных для расчёта.")
                st.stop()

            df_all = pd.concat(all_rows, ignore_index=True)
            df_w = weight_scheme(df_all, S=S_ref, h=H_EXP, kappa=KAPPA)
            prof = build_profiles(df_w, S=S_ref, smooth_window=SMOOTH_WINDOW)
            flips, pos, neg = find_levels(prof)

            col1, col2 = st.columns([2,1])
            with col1:
                title_note = f"({ticker.strip().upper()}, {time.strftime('%Y-%m-%d', time.gmtime(picked[0]))} +7)"
                fig = plot_profiles(prof, S=S_ref, flips=flips, pos=pos, neg=neg, title_note=title_note)
                st.plotly_chart(fig, use_container_width=True)

            def rows_from_mags(mags, sign_label):
                return [{"Strike": float(k), "Magnet (взвеш.)": float(v), "Сила |Magnet|": float(a), "Сторона": sign_label}
                        for (k,v,a) in mags]

            rows = rows_from_mags(pos, "+") + rows_from_mags(neg, "-")
            levels_df = pd.DataFrame(rows).sort_values("Сила |Magnet|", ascending=False)

            with col2:
                st.subheader("Ключевые уровни (магниты)")
                st.dataframe(levels_df, use_container_width=True)
                st.download_button("Скачать уровни (CSV)",
                                   data=levels_df.to_csv(index=False).encode("utf-8"),
                                   file_name=f"{ticker.strip().upper()}_magnet_levels.csv",
                                   mime="text/csv")

            st.subheader("Интрадей план (по методике исследования)")
            checklist = []
            if flips:
                flip_zone = f"{min(flips):.2f}–{max(flips):.2f}" if len(flips) > 1 else f"{flips[0]:.2f}"
                checklist.append(f"Нулевая гамма (flip): {flip_zone}. Ниже — mean-revert, выше — breakout.")
            if pos:
                checklist.append("Главные магниты (+): " + ", ".join(str(round(k)) for (k,_,__) in pos[:3]))
            if neg:
                checklist.append("Главные магниты (–): " + ", ".join(str(round(k)) for (k,_,__) in neg[:3]))
            if not checklist:
                checklist.append("Магниты не выявлены (проверь ликвидность серии или увеличь сглаживание).")
            st.markdown("- " + "\n- ".join(checklist))

    except Exception as e:
        st.error(str(e))
        st.info("Если видишь 404/403 — проверь host/ключ и попробуй другой маршрут. Код уже перебирает популярные пути.")
