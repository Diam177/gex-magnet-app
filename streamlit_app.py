# streamlit_app.py
import time
import requests
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ====================== Страница ======================
st.set_page_config(page_title="GEX Levels & Magnet Profile", layout="wide")
st.title("GEX Levels & Magnet Profile (по комбинированной методике)")

# ====================== Константы методики ======================
SECONDS_PER_YEAR = 31557600.0
DEFAULT_R = 0.01
DEFAULT_Q = 0.00
H_EXP = 7.0
KAPPA = 1.0
SMOOTH_WINDOW = 3
TOP_N_LEVELS = 5

# ====================== Секреты / ввод ======================
host_default = st.secrets.get("RAPIDAPI_HOST", "")
key_default  = st.secrets.get("RAPIDAPI_KEY", "")

with st.sidebar:
    st.header("Провайдер (RapidAPI)")
    RAPIDAPI_HOST = st.text_input("X-RapidAPI-Host", host_default, placeholder="yahoo-finance15.p.rapidapi.com")
    RAPIDAPI_KEY  = st.text_input("X-RapidAPI-Key",  key_default, type="password")
    st.caption("Рекомендуется хранить ключи в Secrets Streamlit.")
    st.divider()
    ticker = st.text_input("Тикер", value="SPY").strip().upper()
    btn_load = st.button("Загрузить экспирации")
    with st.expander("Параметры методики", expanded=False):
        H_EXP   = st.slider("h (вес экспирации, дней)", 3.0, 14.0, H_EXP, 0.5)
        KAPPA   = st.slider("κ (достижимость)", 0.5, 2.0, KAPPA, 0.1)
        SMOOTH_WINDOW = st.select_slider("Сглаживание по страйку", options=[1,3,5,7], value=SMOOTH_WINDOW)

# ====================== API: YH Finance (steadyapi) ======================
def api_headers():
    return {"X-RapidAPI-Key": RAPIDAPI_KEY, "X-RapidAPI-Host": RAPIDAPI_HOST}

def BASE_URL():
    return f"https://{RAPIDAPI_HOST}/api/v1/markets/options"

def _try_get(url: str, params: dict | None = None):
    r = requests.get(url, headers=api_headers(), params=params or {}, timeout=25)
    if r.status_code == 200:
        try:
            return r.json()
        except Exception:
            raise requests.HTTPError(f"Bad JSON from {url}")
    raise requests.HTTPError(f"{r.status_code} {url}\n{r.text[:400]}")

# --- основной список дат (требует display=straddle) ---
def fetch_chain_raw(symbol: str):
    return _try_get(BASE_URL(), params={"ticker": symbol, "display": "straddle"})

# --- небольшой извлекатель цены/времени из options-ответа ---
def _quote_from_options_body(raw: dict) -> dict:
    q = {}
    body = raw.get("body")
    if isinstance(body, list) and body:
        b0 = body[0]
        # цена
        cand_price = []
        if isinstance(b0.get("quote"), dict):
            cand_price += [b0["quote"].get(k) for k in ("regularMarketPrice","last","underlyingPrice","price")]
        cand_price += [b0.get(k) for k in ("regularMarketPrice","last","underlyingPrice","price")]
        for v in cand_price:
            if isinstance(v, (int,float)):
                q["regularMarketPrice"] = float(v); break
        # время
        cand_time = []
        if isinstance(b0.get("quote"), dict):
            cand_time += [b0["quote"].get(k) for k in ("regularMarketTime","time","timestamp")]
        cand_time += [b0.get(k) for k in ("regularMarketTime","time","timestamp")]
        for v in cand_time:
            if isinstance(v, (int,float)):
                q["regularMarketTime"] = int(v); break
    return q

# --- унификация под нашу схему ---
def ensure_chain_shape(raw: dict):
    """
    { 'quote': {...}, 'expirationDates': [epoch,...],
      'chains': [{'expiration': epoch, 'calls': [...], 'puts': [...]}] }
    """
    quote = {}
    expirationDates = []

    # даты: body[0].expirationDates
    body = raw.get("body")
    if isinstance(body, list) and body:
        b0 = body[0]
        exp_list = b0.get("expirationDates") or b0.get("expirations") or b0.get("dates")
        if isinstance(exp_list, list):
            expirationDates = [int(x) for x in exp_list if isinstance(x, (int, float))]

    # quote из options-ответа (что удастся)
    quote.update(_quote_from_options_body(raw))

    # цепочки: если провайдер сразу положил calls/puts в body[n]
    chains = []
    if isinstance(body, list):
        for obj in body:
            if not isinstance(obj, dict): 
                continue
            calls = obj.get("calls") or (obj.get("options") or {}).get("calls")
            puts  = obj.get("puts")  or (obj.get("options") or {}).get("puts")
            exp   = obj.get("expiration") or obj.get("expiry") or obj.get("date")
            if isinstance(calls, list) and isinstance(puts, list) and exp is not None:
                chains.append({"expiration": int(exp), "calls": calls, "puts": puts})

    # дедуп/сортировка
    seen, norm = set(), []
    for ch in chains:
        exp = ch.get("expiration")
        sig = (exp, len(ch.get("calls", [])) + len(ch.get("puts", [])))
        if exp is None or sig in seen: 
            continue
        seen.add(sig); norm.append(ch)
    norm.sort(key=lambda x: x.get("expiration", 0))

    return {"quote": quote, "expirationDates": expirationDates, "chains": norm}

# --- котировочный фолбэк, если в options нет цены/времени ---
def fetch_quote(symbol: str) -> dict:
    base = f"https://{RAPIDAPI_HOST}"
    candidates = [
        (f"{base}/api/v1/markets/quotes", {"tickers": symbol}),
        (f"{base}/api/v1/market/quotes",  {"tickers": symbol}),
        (f"{base}/api/v1/markets/quotes/real-time", {"symbols": symbol}),
    ]
    for url, params in candidates:
        try:
            raw = _try_get(url, params=params)
            b = raw.get("body")
            if isinstance(b, list) and b:
                out = {}
                p = b[0].get("regularMarketPrice") or b[0].get("price") or b[0].get("last")
                t = b[0].get("regularMarketTime")  or b[0].get("time")  or b[0].get("timestamp")
                if isinstance(p, (int,float)): out["regularMarketPrice"] = float(p)
                if isinstance(t, (int,float)): out["regularMarketTime"]  = int(t)
                if out: return out
        except Exception:
            continue
    return {}

# --- конкретная дата (options + quote-фолбэк) ---
def fetch_specific_expiry(symbol: str, epoch: int):
    raw = _try_get(BASE_URL(), params={"ticker": symbol, "expiration": int(epoch), "display": "straddle"})
    shaped = ensure_chain_shape(raw)

    chain = None
    for ch in shaped["chains"]:
        if ch.get("expiration") == int(epoch):
            chain = ch; break
    if chain is None:
        body = raw.get("body")
        if isinstance(body, list) and body:
            b0 = body[0]
            if isinstance(b0.get("calls"), list) and isinstance(b0.get("puts"), list):
                chain = {"expiration": int(epoch), "calls": b0.get("calls", []), "puts": b0.get("puts", [])}
    if chain is None:
        chain = {"expiration": int(epoch), "calls": [], "puts": []}

    quote = dict(shaped.get("quote", {}))
    if "regularMarketPrice" not in quote or "regularMarketTime" not in quote:
        quote.update({k:v for k,v in fetch_quote(symbol).items() if k not in quote})

    return {"quote": quote, "chain": chain}

# ====================== Математика ======================
def bsm_gamma(S, K, sigma, tau, r=DEFAULT_R, q=DEFAULT_Q):
    if tau <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0.0
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
    phi = np.exp(-0.5 * d1**2) / np.sqrt(2 * np.pi)
    gamma = phi * np.exp(-q * tau) / (S * sigma * np.sqrt(tau))
    return float(gamma)

def compute_chain_gex(chain: dict, quote: dict):
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
        gamma = bsm_gamma(S, K, iv, tau)
        gex   = oi * gamma * 100 * S
        rows.append({"strike":K, "type":"call", "oi":oi, "iv":iv, "tau":tau, "gex_signed": gex})
    for row in puts:
        K   = row.get("strike")
        oi  = row.get("openInterest", 0) or 0
        iv  = row.get("impliedVolatility", 0) or 0
        gamma = bsm_gamma(S, K, iv, tau)
        gex   = oi * gamma * 100 * S
        rows.append({"strike":K, "type":"put", "oi":oi, "iv":iv, "tau":tau, "gex_signed": -gex})

    return pd.DataFrame(rows), S

def weight_scheme(df_all_exp: pd.DataFrame, S: float, h=H_EXP, kappa=KAPPA):
    df = df_all_exp.copy()
    df["DTE"]   = df["tau"] * 365.0
    df["W_exp"] = 2.0 ** (-df["DTE"] / h)

    oi_by_exp = df.groupby("expiry")["oi"].sum().rename("exp_oi")
    df = df.merge(oi_by_exp, on="expiry", how="left")
    total_oi = float(df["oi"].sum()) or 1.0
    df["OI_share"] = df["exp_oi"] / total_oi

    sig = df["iv"].clip(lower=1e-6)
    root_tau = np.sqrt(df["tau"].clip(lower=1e-9))
    denom = 2.0 * (kappa**2) * (sig * root_tau)**2
    log_term = np.log(np.maximum(df["strike"], 1e-6) / max(S, 1e-6))
    df["W_dist"] = np.exp(- (log_term**2) / np.maximum(denom, 1e-12))

    return df

def build_profiles(df_w: pd.DataFrame, S: float, smooth_window=SMOOTH_WINDOW):
    raw = df_w.groupby("strike")["gex_signed"].sum().rename("NetGEX_raw").reset_index()
    df_w["W_liq"] = np.sqrt(df_w["OI_share"].clip(lower=0))
    df_w["W_prod"] = df_w["W_exp"] * df_w["W_liq"] * df_w["W_dist"]
    df_w["contrib"] = df_w["gex_signed"] * df_w["W_prod"] / S
    magnet = df_w.groupby("strike")["contrib"].sum().rename("Magnet").reset_index()
    prof = raw.merge(magnet, on="strike", how="outer").sort_values("strike")
    prof["NetGEX_smooth"] = prof["NetGEX_raw"].rolling(smooth_window, center=True, min_periods=1).mean()
    prof["Magnet_smooth"] = prof["Magnet"].rolling(smooth_window, center=True, min_periods=1).mean()
    return prof

def find_levels(profile: pd.DataFrame):
    prof = profile.dropna(subset=["Magnet_smooth"]).copy()
    strikes = prof["strike"].values
    vals = prof["Magnet_smooth"].values

    flips = []
    for i in range(1, len(vals)):
        y0, y1 = vals[i-1], vals[i]
        if y0 == 0: 
            continue
        if (y0 > 0 and y1 < 0) or (y0 < 0 and y1 > 0):
            x0, x1 = strikes[i-1], strikes[i]
            x_cross = x0 + (x1 - x0) * (-y0) / (y1 - y0) if (y1 - y0) != 0 else (x0 + x1)/2
            flips.append(x_cross)

    mags, absvals = [], np.abs(vals)
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

# ====================== Основной поток ======================
if btn_load:
    if not RAPIDAPI_HOST or not RAPIDAPI_KEY:
        st.error("Укажи RapidAPI Host и Key (в сайдбаре или в Secrets).")
        st.stop()
    try:
        raw = fetch_chain_raw(ticker)

        # Debug-экспандер — удобно, если провайдер поменяет формат
        with st.expander("Debug: сырой ответ провайдера", expanded=False):
            if isinstance(raw, dict):
                st.write("Ключи:", list(raw.keys())[:20])
            st.json(raw)

        shaped = ensure_chain_shape(raw)
        quote = shaped.get("quote", {})
        exp_dates = shaped.get("expirationDates") or [c.get("expiration") for c in shaped["chains"]]
        exp_dates = sorted({int(e) for e in exp_dates if isinstance(e, (int, float))})
        if not exp_dates:
            st.error("Не удалось получить список экспираций (пустой ответ).")
            st.stop()

        human = [time.strftime("%Y-%m-%d", time.gmtime(e)) for e in exp_dates]
        picked_idx = st.selectbox("Выбери ближайшую экспирацию", list(range(len(exp_dates))),
                                  format_func=lambda i: human[i], index=0)

        if st.button("Рассчитать уровни (эта + 7 следующих)"):
            picked = exp_dates[picked_idx: picked_idx + 8]
            all_rows, S_ref = [], None
            for e in picked:
                dat = fetch_specific_expiry(ticker, e)
                df_i, S_i = compute_chain_gex(dat["chain"], dat["quote"])
                df_i["expiry"] = int(e)
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
                title_note = f"({ticker}, {time.strftime('%Y-%m-%d', time.gmtime(picked[0]))} +7)"
                fig = plot_profiles(prof, S=S_ref, flips=flips, pos=pos, neg=neg, title_note=title_note)
                st.plotly_chart(fig, use_container_width=True)

            def rows_from_mags(mags, label):
                return [{"Strike": float(k), "Magnet (взвеш.)": float(v), "Сила |Magnet|": float(a), "Сторона": label}
                        for (k,v,a) in mags]
            levels_df = pd.DataFrame(rows_from_mags(pos, "+") + rows_from_mags(neg, "-")) \
                        .sort_values("Сила |Magnet|", ascending=False)

            with col2:
                st.subheader("Ключевые уровни (магниты)")
                st.dataframe(levels_df, use_container_width=True)
                st.download_button("Скачать уровни (CSV)",
                                   data=levels_df.to_csv(index=False).encode("utf-8"),
                                   file_name=f"{ticker}_magnet_levels.csv",
                                   mime="text/csv")

            st.subheader("Интрадей план (по методике исследования)")
            checklist = []
            if flips:
                flip_zone = f"{min(flips):.2f}–{max(flips):.2f}" if len(flips) > 1 else f"{flips[0]:.2f}"
                checklist.append(f"Нулевая гамма (flip): {flip_zone}. Выше — breakout, ниже — mean-revert.")
            if pos:
                checklist.append("Главные магниты (+): " + ", ".join(str(round(k)) for (k,_,__) in pos[:3]))
            if neg:
                checklist.append("Главные магниты (–): " + ", ".join(str(round(k)) for (k,_,__) in neg[:3]))
            if not checklist:
                checklist.append("Магниты не выявлены — увеличь сглаживание или проверь ликвидность.")
            st.markdown("- " + "\n- ".join(checklist))

    except Exception as e:
        st.error(str(e))
        st.info("Если снова изменится формат ответа — раскрой Debug и пришли верхнюю часть JSON, я подгоню парсер.")
