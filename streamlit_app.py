# streamlit_app.py
import time, json
import requests
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# -------------------- UI --------------------
st.set_page_config(page_title="GEX Levels & Magnet Profile", layout="wide")
st.title("GEX Levels & Magnet Profile (по комбинированной методике)")

# методика — базовые константы
SECONDS_PER_YEAR = 31557600.0
DEFAULT_R = 0.01
DEFAULT_Q = 0.00
TOP_N_LEVELS = 5

# значения по умолчанию
def_val = lambda k, v: st.secrets.get(k, v)
host_default = def_val("RAPIDAPI_HOST", "")
key_default  = def_val("RAPIDAPI_KEY", "")

with st.sidebar:
    st.header("Провайдер (RapidAPI)")
    RAPIDAPI_HOST = st.text_input("X-RapidAPI-Host", host_default, placeholder="yahoo-finance15.p.rapidapi.com")
    RAPIDAPI_KEY  = st.text_input("X-RapidAPI-Key",  key_default, type="password")
    st.caption("Рекомендуется хранить ключи в Secrets Streamlit.")
    st.divider()
    ticker = st.text_input("Тикер", value="SPY").strip().upper()
    btn_load = st.button("Загрузить экспирации")
    with st.expander("Параметры методики", expanded=False):
        H_EXP   = st.slider("h (вес экспирации, дней)", 3.0, 14.0, 7.0, 0.5)
        KAPPA   = st.slider("κ (достижимость)", 0.5, 2.0, 1.0, 0.1)
        SMOOTH  = st.select_slider("Сглаживание по страйку", options=[1,3,5,7], value=3)

# -------------------- RapidAPI helpers --------------------
def api_headers():
    return {"X-RapidAPI-Key": RAPIDAPI_KEY, "X-RapidAPI-Host": RAPIDAPI_HOST}

def base_url():
    return f"https://{RAPIDAPI_HOST}/api/v1/markets/options"

def _get(url, params=None):
    r = requests.get(url, headers=api_headers(), params=params or {}, timeout=25)
    r.raise_for_status()
    return r.json()

def _norm_top(raw):
    return {"body": raw} if isinstance(raw, list) else raw

def _to_sec(x):
    """нормализуем секунды/миллисекунды"""
    if x is None: return None
    x = int(x)
    return x//1000 if x > 10**12 else x

def fetch_options_listing(symbol: str):
    url = base_url()
    for disp in ("list", "straddle", None):
        try:
            params = {"ticker": symbol}
            if disp: params["display"] = disp
            return _norm_top(_get(url, params))
        except Exception:
            continue
    # последняя попытка — без хост-роутинга (бывает капризный)
    return _norm_top(_get(url, {"ticker": symbol}))

def fetch_quote(symbol: str) -> dict:
    base = f"https://{RAPIDAPI_HOST}"
    candidates = [
        (f"{base}/api/v1/markets/quotes", {"tickers": symbol}),
        (f"{base}/api/v1/market/quotes",  {"tickers": symbol}),
        (f"{base}/api/v1/markets/quotes/real-time", {"symbols": symbol}),
    ]
    for url, p in candidates:
        try:
            raw = _norm_top(_get(url, p))
            b = raw.get("body")
            if isinstance(b, list) and b:
                q = {}
                p_ = b[0].get("regularMarketPrice") or b[0].get("price") or b[0].get("last")
                t_ = b[0].get("regularMarketTime")  or b[0].get("time")  or b[0].get("timestamp")
                if isinstance(p_, (int,float)): q["regularMarketPrice"] = float(p_)
                if t_ is not None: q["regularMarketTime"] = _to_sec(t_)
                if q: return q
        except Exception:
            continue
    return {}

def quote_from_options(raw: dict) -> dict:
    q = {}
    body = raw.get("body")
    if isinstance(body, list) and body:
        b0 = body[0]
        qnode = b0.get("quote") if isinstance(b0.get("quote"), dict) else b0
        for key in ("regularMarketPrice", "last", "underlyingPrice", "price"):
            v = qnode.get(key)
            if isinstance(v, (int,float)):
                q["regularMarketPrice"] = float(v); break
        for key in ("regularMarketTime", "time", "timestamp"):
            v = qnode.get(key)
            if v is not None:
                q["regularMarketTime"] = _to_sec(v); break
    return q

def ensure_shape(raw: dict):
    """Возвращает {expirationDates, chains[], quote} с chains: {expiration, calls[], puts[]}."""
    raw = _norm_top(raw)
    body = raw.get("body")
    quote = quote_from_options(raw)

    expirationDates, chains = [], []
    if isinstance(body, list) and body:
        b0 = body[0]
        # список дат
        exps = b0.get("expirationDates") or b0.get("expirations") or b0.get("dates")
        if isinstance(exps, list):
            expirationDates = [_to_sec(x) for x in exps if isinstance(x, (int,float))]

        # разные места хранения
        def add_chain(exp, calls, puts):
            if exp is None: return
            chains.append({"expiration": _to_sec(exp), "calls": calls or [], "puts": puts or []})

        # напрямую
        if isinstance(b0.get("calls"), list) and isinstance(b0.get("puts"), list) and b0.get("expiration"):
            add_chain(b0["expiration"], b0["calls"], b0["puts"])

        # через options[]
        if isinstance(b0.get("options"), list) and b0["options"]:
            o0 = b0["options"][0]
            if isinstance(o0.get("calls"), list) and isinstance(o0.get("puts"), list) and o0.get("expirationDate"):
                add_chain(o0["expirationDate"], o0["calls"], o0["puts"])
            if isinstance(o0.get("list"), list) and o0["list"]:
                l0 = o0["list"][0]
                if isinstance(l0.get("calls"), list) and isinstance(l0.get("puts"), list) and o0.get("expirationDate"):
                    add_chain(o0["expirationDate"], l0["calls"], l0["puts"])
            if isinstance(o0.get("straddles"), list) and o0["straddles"] and o0.get("expirationDate"):
                calls, puts = [], []
                for s in o0["straddles"]:
                    c, p = s.get("call") or {}, s.get("put") or {}
                    if c: calls.append({"strike": c.get("strike") or s.get("strike"),
                                        "openInterest": c.get("openInterest", 0),
                                        "impliedVolatility": c.get("impliedVolatility", 0)})
                    if p: puts .append({"strike": p.get("strike") or s.get("strike"),
                                        "openInterest": p.get("openInterest", 0),
                                        "impliedVolatility": p.get("impliedVolatility", 0)})
                add_chain(o0["expirationDate"], calls, puts)

    # дедуп и сортировка
    uniq = {}
    for ch in chains:
        exp = ch["expiration"]
        key = (exp, len(ch["calls"]), len(ch["puts"]))
        uniq[key] = ch
    chains = sorted(uniq.values(), key=lambda z: z["expiration"])
    return {"expirationDates": [e for e in expirationDates if e], "chains": chains, "quote": quote}

def fetch_expiry(symbol: str, epoch: int):
    url = base_url()
    last = None
    for disp in ("list", "straddle", None):
        try:
            p = {"ticker": symbol, "expiration": epoch}
            if disp: p["display"] = disp
            raw = _norm_top(_get(url, p))
            last = raw
            shaped = ensure_shape(raw)
            # возьмём цепочку с совпадающей датой
            for ch in shaped["chains"]:
                if ch["expiration"] == epoch:
                    q = shaped.get("quote", {})
                    # фолбэк на отдельный quote
                    if "regularMarketPrice" not in q or "regularMarketTime" not in q:
                        q2 = fetch_quote(symbol)
                        q.update({k:v for k,v in q2.items() if k not in q})
                    # последний фолбэк на текущее время
                    if "regularMarketTime" not in q or q["regularMarketTime"] is None:
                        q["regularMarketTime"] = int(time.time())
                    return {"quote": q, "chain": ch}
        except Exception:
            continue
    q = quote_from_options(last or {}) or fetch_quote(symbol) or {"regularMarketTime": int(time.time())}
    return {"quote": q, "chain": {"expiration": epoch, "calls": [], "puts": []}}

# -------------------- математика --------------------
def bsm_gamma(S, K, sigma, tau, r=DEFAULT_R, q=DEFAULT_Q):
    if not (S and K and sigma and tau) or S<=0 or K<=0 or sigma<=0 or tau<=0:
        return 0.0
    d1 = (np.log(S/K) + (r - q + 0.5*sigma**2)*tau) / (sigma*np.sqrt(tau))
    phi = np.exp(-0.5*d1**2)/np.sqrt(2*np.pi)
    return float(phi*np.exp(-q*tau)/(S*sigma*np.sqrt(tau)))

def compute_chain_gex(chain: dict, quote: dict):
    S = quote.get("regularMarketPrice")
    t0 = quote.get("regularMarketTime")
    exp = chain.get("expiration")
    if S is None or t0 is None or exp is None:
        return pd.DataFrame(), None
    tau = max((_to_sec(exp) - _to_sec(t0)) / SECONDS_PER_YEAR, 0.0)

    rows = []
    for typ, rows_src, sign in (("call", chain.get("calls", []), +1),
                                ("put",  chain.get("puts",  []), -1)):
        for r in rows_src or []:
            K  = r.get("strike")
            oi = r.get("openInterest", 0) or 0
            iv = r.get("impliedVolatility", 0) or 0
            if not (K and iv) or K<=0: 
                continue
            gamma = bsm_gamma(S, K, float(iv), tau)
            gex   = oi * gamma * 100 * S * sign
            rows.append({"strike": float(K), "type": typ, "oi": float(oi), "iv": float(iv),
                         "tau": float(tau), "gex_signed": float(gex)})
    df = pd.DataFrame(rows)
    return df, float(S)

def weight_and_profile(df_all: pd.DataFrame, S: float, h_days: float, kappa: float, smooth: int):
    if df_all.empty: 
        return pd.DataFrame()
    df = df_all.copy()
    df["DTE"]   = df["tau"] * 365.0
    df["W_exp"] = 2.0 ** (-df["DTE"]/h_days)

    oi_by_exp = df.groupby("expiry")["oi"].sum().rename("exp_oi")
    df = df.merge(oi_by_exp, on="expiry", how="left")
    total_oi = float(df["oi"].sum()) or 1.0
    df["OI_share"] = df["exp_oi"]/total_oi

    sig = df["iv"].clip(lower=1e-6)
    root_tau = np.sqrt(df["tau"].clip(lower=1e-9))
    denom = 2.0*(kappa**2)*(sig*root_tau)**2
    logt = np.log(np.maximum(df["strike"], 1e-6)/max(S,1e-6))
    df["W_dist"] = np.exp(-(logt**2)/np.maximum(denom, 1e-12))

    df["W_liq"] = np.sqrt(df["OI_share"].clip(lower=0))
    df["W_prod"] = df["W_exp"]*df["W_liq"]*df["W_dist"]
    df["contrib"] = df["gex_signed"]*df["W_prod"]/S

    raw = df.groupby("strike")["gex_signed"].sum().rename("NetGEX_raw").reset_index()
    mag = df.groupby("strike")["contrib"].sum().rename("Magnet").reset_index()
    prof = raw.merge(mag, on="strike", how="outer").sort_values("strike")
    prof["NetGEX_smooth"] = prof["NetGEX_raw"].rolling(smooth, center=True, min_periods=1).mean()
    prof["Magnet_smooth"] = prof["Magnet"].rolling(smooth, center=True, min_periods=1).mean()
    return prof

def find_levels(profile: pd.DataFrame):
    if profile.empty: return [], [], []
    prof = profile.dropna(subset=["Magnet_smooth"]).copy()
    strikes = prof["strike"].values
    vals = prof["Magnet_smooth"].values
    flips = []
    for i in range(1, len(vals)):
        y0, y1 = vals[i-1], vals[i]
        if (y0>0 and y1<0) or (y0<0 and y1>0):
            x0,x1 = strikes[i-1], strikes[i]
            x = x0 + (x1-x0)*(-y0)/(y1-y0) if (y1-y0)!=0 else (x0+x1)/2
            flips.append(float(x))
    mags, absv = [], np.abs(vals)
    for i in range(1,len(vals)-1):
        if absv[i]>=absv[i-1] and absv[i]>=absv[i+1]:
            mags.append((float(strikes[i]), float(vals[i]), float(absv[i])))
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
    for x in flips: fig.add_vline(x=x, line_dash="dash", line_color="#f1c40f", annotation_text="Gamma Flip")
    for (k,v,a) in pos: fig.add_scatter(x=[k], y=[v], mode="markers", marker=dict(size=10), name=f"+Magnet {k}")
    for (k,v,a) in neg: fig.add_scatter(x=[k], y=[v], mode="markers", marker=dict(size=10), name=f"-Magnet {k}")
    fig.update_layout(title=f"Профиль GEX/Magnet {title_note}", xaxis_title="Strike", yaxis_title="Value", height=520)
    return fig

# -------------------- Загрузка экспираций --------------------
if btn_load:
    try:
        raw = fetch_options_listing(ticker)
        with st.expander("Debug: сырой ответ провайдера", expanded=False):
            st.json(raw)
            st.download_button("Скачать Debug JSON",
                               data=json.dumps(raw, ensure_ascii=False, indent=2),
                               file_name=f"debug_{ticker}.json",
                               mime="application/json")
        shaped = ensure_shape(raw)
        exp_dates = shaped.get("expirationDates") or [c.get("expiration") for c in shaped["chains"]]
        exp_dates = sorted({e for e in exp_dates if e})
        if not exp_dates:
            st.error("Не удалось получить список экспираций (пустой ответ).")
            st.stop()

        human = [time.strftime("%Y-%m-%d", time.gmtime(e)) for e in exp_dates]
        idx = st.selectbox("Выбери ближайшую экспирацию", list(range(len(exp_dates))),
                           format_func=lambda i: human[i], index=0)

        # Кнопка расчёта
        if st.button("Рассчитать уровни (эта + 7 следующих)"):
            picked = exp_dates[idx: idx+8]
            log_box = st.empty()
            progress = st.progress(0)
            all_rows, S_ref = [], None
            per_exp_info = []

            with st.spinner("Тянем цепочки и считаем Гамму..."):
                for j, e in enumerate(picked, start=1):
                    dat = fetch_expiry(ticker, e)
                    df_i, S_i = compute_chain_gex(dat["chain"], dat["quote"])
                    calls_n = len(dat["chain"].get("calls", []))
                    puts_n  = len(dat["chain"].get("puts",  []))
                    per_exp_info.append(f"{time.strftime('%Y-%m-%d', time.gmtime(e))}: calls={calls_n}, puts={puts_n}, rows={len(df_i)}")
                    if not df_i.empty:
                        df_i["expiry"] = int(e)
                        all_rows.append(df_i)
                        if S_ref is None and S_i is not None:
                            S_ref = S_i
                    progress.progress(j/len(picked))
                    log_box.write("\n".join(per_exp_info))

            if not all_rows:
                st.warning("По выбранному окну экспираций цепочки пустые (нет строк для расчёта). "
                           "Попробуй другую дату или тикер.")
                st.stop()

            if S_ref is None:
                q = fetch_quote(ticker)
                S_ref = q.get("regularMarketPrice")
            if S_ref is None:
                st.error("Не удалось определить Spot (regularMarketPrice).")
                st.stop()

            df_all = pd.concat(all_rows, ignore_index=True)
            prof   = weight_and_profile(df_all, S=S_ref, h_days=H_EXP, kappa=KAPPA, smooth=SMOOTH)
            if prof.empty:
                st.warning("Профиль пуст. Возможно, фильтрация/веса занулили вклад. Увеличь сглаживание или проверь ликвидность.")
                st.stop()

            flips, pos, neg = find_levels(prof)

            c1, c2 = st.columns([2,1])
            with c1:
                title = f"({ticker}, {time.strftime('%Y-%m-%d', time.gmtime(picked[0]))} +7)"
                st.plotly_chart(plot_profiles(prof, S=S_ref, flips=flips, pos=pos, neg=neg, title_note=title),
                                use_container_width=True)

            def rows(mags, label):
                return [{"Strike": k, "Magnet": v, "|Magnet|": a, "Side": label} for (k,v,a) in mags]
            levels = pd.DataFrame(rows(pos, "+") + rows(neg, "-")).sort_values("|Magnet|", ascending=False)

            with c2:
                st.subheader("Ключевые уровни (магниты)")
                st.dataframe(levels, use_container_width=True)
                st.download_button("Скачать уровни (CSV)",
                                   data=levels.to_csv(index=False).encode("utf-8"),
                                   file_name=f"{ticker}_magnet_levels.csv",
                                   mime="text/csv")

    except Exception as e:
        st.error(f"Ошибка: {e}")
        st.info("Открой «Debug: сырой ответ провайдера» и скачай JSON — пришли мне файл, если ошибка повторится.")
