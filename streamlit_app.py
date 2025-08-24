# streamlit_app.py
import time, json
import requests
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ============== Страница ==============
st.set_page_config(page_title="GEX Levels & Magnet Profile", layout="wide")
st.title("GEX Levels & Magnet Profile (по комбинированной методике)")

# ============== Константы методики ==============
SECONDS_PER_YEAR = 31557600.0
DEFAULT_R = 0.01
DEFAULT_Q = 0.00
H_EXP = 7.0
KAPPA = 1.0
SMOOTH_WINDOW = 3
TOP_N_LEVELS = 5

# ============== Секреты / ввод ==============
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
        st.session_state.setdefault("H_EXP", H_EXP)
        st.session_state.setdefault("KAPPA", KAPPA)
        st.session_state.setdefault("SMOOTH_WINDOW", SMOOTH_WINDOW)
        st.session_state["H_EXP"] = st.slider("h (вес экспирации, дней)", 3.0, 14.0, st.session_state["H_EXP"], 0.5)
        st.session_state["KAPPA"] = st.slider("κ (достижимость)", 0.5, 2.0, st.session_state["KAPPA"], 0.1)
        st.session_state["SMOOTH_WINDOW"] = st.select_slider("Сглаживание по страйку", options=[1,3,5,7], value=st.session_state["SMOOTH_WINDOW"])

def H(): return float(st.session_state["H_EXP"])
def K(): return float(st.session_state["KAPPA"])
def SMOOTH(): return int(st.session_state["SMOOTH_WINDOW"])

# ============== RapidAPI YH Finance (steadyapi) ==============
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

def _normalize_top(raw):
    """Если верхний уровень — список, оборачиваем в dict."""
    return {"body": raw} if isinstance(raw, list) else raw

# --- получить список экспираций; сначала display=list, затем straddle ---
def fetch_chain_raw(symbol: str):
    url = BASE_URL()
    for disp in ("list", "straddle"):
        try:
            raw = _normalize_top(_try_get(url, params={"ticker": symbol, "display": disp}))
            # наличие тела
            body = raw.get("body")
            if isinstance(body, list) and body:
                return raw
        except Exception:
            continue
    # Последняя попытка — без display
    return _normalize_top(_try_get(url, params={"ticker": symbol}))

# --- котировочный фолбэк, если из options не достали цену/время ---
def fetch_quote(symbol: str) -> dict:
    base = f"https://{RAPIDAPI_HOST}"
    candidates = [
        (f"{base}/api/v1/markets/quotes", {"tickers": symbol}),
        (f"{base}/api/v1/market/quotes",  {"tickers": symbol}),
        (f"{base}/api/v1/markets/quotes/real-time", {"symbols": symbol}),
    ]
    for url, params in candidates:
        try:
            raw = _normalize_top(_try_get(url, params=params))
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

# --- извлечь quote из options-сообщения (что найдём) ---
def _quote_from_options_body(raw: dict) -> dict:
    q = {}
    body = raw.get("body")
    if isinstance(body, list) and body:
        b0 = body[0]
        cand_price = []
        if isinstance(b0.get("quote"), dict):
            cand_price += [b0["quote"].get(k) for k in ("regularMarketPrice","last","underlyingPrice","price")]
        cand_price += [b0.get(k) for k in ("regularMarketPrice","last","underlyingPrice","price")]
        for v in cand_price:
            if isinstance(v, (int,float)):
                q["regularMarketPrice"] = float(v); break
        cand_time = []
        if isinstance(b0.get("quote"), dict):
            cand_time += [b0["quote"].get(k) for k in ("regularMarketTime","time","timestamp")]
        cand_time += [b0.get(k) for k in ("regularMarketTime","time","timestamp")]
        for v in cand_time:
            if isinstance(v, (int,float)):
                q["regularMarketTime"] = int(v); break
    return q

# --- универсальная вытяжка expiry-дат и возможных calls/puts/straddles ---
def ensure_chain_shape(raw: dict | list):
    raw = _normalize_top(raw)
    quote = _quote_from_options_body(raw)
    expirationDates = []
    chains = []

    body = raw.get("body")
    if isinstance(body, list) and body:
        b0 = body[0]
        # Список дат
        exps = b0.get("expirationDates") or b0.get("expirations") or b0.get("dates")
        if isinstance(exps, list):
            expirationDates = [int(x) for x in exps if isinstance(x, (int,float))]

        # Варианты расположения опционной цепочки
        # 1) Прямо в body[n]
        if isinstance(b0.get("calls"), list) and isinstance(b0.get("puts"), list) and b0.get("expiration"):
            chains.append({"expiration": int(b0["expiration"]),
                           "calls": b0["calls"], "puts": b0["puts"]})
        # 2) Внутри options[]
        if isinstance(b0.get("options"), list) and b0["options"]:
            o0 = b0["options"][0]
            # 2a) options[0].calls / options[0].puts
            c1, p1 = o0.get("calls"), o0.get("puts")
            if isinstance(c1, list) and isinstance(p1, list) and o0.get("expirationDate"):
                chains.append({"expiration": int(o0["expirationDate"]), "calls": c1, "puts": p1})
            # 2b) options[0].list[0].calls/puts
            if isinstance(o0.get("list"), list) and o0["list"]:
                l0 = o0["list"][0]
                if isinstance(l0.get("calls"), list) and isinstance(l0.get("puts"), list) and o0.get("expirationDate"):
                    chains.append({"expiration": int(o0["expirationDate"]), "calls": l0["calls"], "puts": l0["puts"]})
            # 2c) options[0].straddles[] → сконвертировать в calls/puts
            if isinstance(o0.get("straddles"), list) and o0["straddles"]:
                calls, puts = [], []
                for s in o0["straddles"]:
                    call = s.get("call") or {}
                    put  = s.get("put")  or {}
                    if call:
                        calls.append({
                            "strike": call.get("strike") or s.get("strike"),
                            "openInterest": call.get("openInterest", 0),
                            "impliedVolatility": call.get("impliedVolatility", 0),
                        })
                    if put:
                        puts.append({
                            "strike": put.get("strike") or s.get("strike"),
                            "openInterest": put.get("openInterest", 0),
                            "impliedVolatility": put.get("impliedVolatility", 0),
                        })
                if (calls or puts) and o0.get("expirationDate"):
                    chains.append({"expiration": int(o0["expirationDate"]), "calls": calls, "puts": puts})

    # дедуп и сортировка
    seen, out = set(), []
    for ch in chains:
        exp = ch.get("expiration")
        sig = (exp, len(ch.get("calls", [])) + len(ch.get("puts", [])))
        if exp is None or sig in seen:
            continue
        seen.add(sig); out.append(ch)
    out.sort(key=lambda x: x.get("expiration", 0))

    return {"quote": quote, "expirationDates": expirationDates, "chains": out}

# --- конкретная дата: пробуем display=list, затем straddle; конвертируем при необходимости ---
def fetch_specific_expiry(symbol: str, epoch: int):
    url = BASE_URL()
    last_raw = None
    for disp in ("list", "straddle", None):
        try:
            params = {"ticker": symbol, "expiration": int(epoch)}
            if disp: params["display"] = disp
            raw = _normalize_top(_try_get(url, params=params))
            last_raw = raw
            shaped = ensure_chain_shape(raw)
            # выбрать подходящую цепочку
            chain = None
            for ch in shaped["chains"]:
                if ch.get("expiration") == int(epoch):
                    chain = ch; break
            if chain:
                quote = dict(shaped.get("quote", {}))
                if "regularMarketPrice" not in quote or "regularMarketTime" not in quote:
                    quote.update({k:v for k,v in fetch_quote(symbol).items() if k not in quote})
                return {"quote": quote, "chain": chain}
        except Exception:
            continue
    # fallback — пусто, но с котировкой
    quote = _quote_from_options_body(last_raw or {}) or fetch_quote(symbol) or {}
    return {"quote": quote, "chain": {"expiration": int(epoch), "calls": [], "puts": []}}

# ============== Математика ==============
def bsm_gamma(S, K, sigma, tau, r=DEFAULT_R, q=DEFAULT_Q):
    if tau <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0.0
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
    phi = np.exp(-0.5 * d1**2) / np.sqrt(2 * np.pi)
    return float(phi * np.exp(-q * tau) / (S * sigma * np.sqrt(tau)))

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

def weight_scheme(df_all_exp: pd.DataFrame, S: float, h=H(), kappa=K()):
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

def build_profiles(df_w: pd.DataFrame, S: float, smooth_window=SMOOTH()):
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

# ============== Основной поток ==============
if btn_load:
    if not RAPIDAPI_HOST or not RAPIDAPI_KEY:
        st.error("Укажи RapidAPI Host и Key (в сайдбаре или в Secrets).")
        st.stop()
    try:
        raw = fetch_chain_raw(ticker)

        # Debug-экспандер + кнопка «Скачать»
        with st.expander("Debug: сырой ответ провайдера", expanded=False):
            st.write("Ключи верхнего уровня:", list(raw.keys()) if isinstance(raw, dict) else type(raw).__name__)
            st.json(raw)
            st.download_button("Скачать Debug JSON",
                               data=json.dumps(raw, ensure_ascii=False, indent=2),
                               file_name=f"debug_{ticker}.json",
                               mime="application/json")

        shaped = ensure_chain_shape(raw)
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
                st.error("Недостаточно данных для расчёта (нет цен/цепочек).")
                st.stop()

            df_all = pd.concat(all_rows, ignore_index=True)
            df_w = weight_scheme(df_all, S=S_ref, h=H(), kappa=K())
            prof = build_profiles(df_w, S=S_ref, smooth_window=SMOOTH())
            flips, pos, neg = find_levels(prof)

            col1, col2 = st.columns([2,1])
            with col1:
                title_note = f"({ticker}, {time.strftime('%Y-%m-%d', time.gmtime(picked[0]))} +7)"
                st.plotly_chart(plot_profiles(prof, S=S_ref, flips=flips, pos=pos, neg=neg, title_note=title_note),
                                use_container_width=True)

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
        st.info("Если снова изменится формат ответа — скачай Debug JSON и пришли его. Кнопка — в экспандере выше.")
