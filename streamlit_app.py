# streamlit_app.py
import time, json
import requests
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ========== UI ==========
st.set_page_config(page_title="GEX Levels & Magnet Profile", layout="wide")
st.title("GEX Levels & Magnet Profile (по комбинированной методике)")

SECONDS_PER_YEAR = 31557600.0
DEFAULT_R = 0.01
DEFAULT_Q = 0.00
TOP_N_LEVELS = 5

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
        H_EXP   = st.slider("h (вес экспирации, дней)", 3.0, 14.0, 7.0, 0.5)
        KAPPA   = st.slider("κ (достижимость)", 0.5, 2.0, 1.0, 0.1)
        SMOOTH  = st.select_slider("Сглаживание по страйку (оконный размер)", options=[1,3,5,7], value=3)


# === Session state init for stable UI across reruns ===
if "exp_dates" not in st.session_state:
    st.session_state.exp_dates = []
    st.session_state.raw_listing = None
    st.session_state.ticker_loaded = None
    st.session_state.exp_idx = 0  # selected expiry index


# ========== RapidAPI helpers ==========
def api_headers():
    return {"X-RapidAPI-Key": RAPIDAPI_KEY, "X-RapidAPI-Host": RAPIDAPI_HOST}

def base_options_url():
    return f"https://{RAPIDAPI_HOST}/api/v1/markets/options"

def _get(url, params=None):
    r = requests.get(url, headers=api_headers(), params=params or {}, timeout=25)
    r.raise_for_status()
    return r.json()

def _norm_top(raw):
    # если верхний уровень — список, упаковываем в {"body": list}
    return {"body": raw} if isinstance(raw, list) else raw

def _to_sec(x):
    if x is None: return None
    x = int(x)
    return x//1000 if x > 10**12 else x  # ms -> s

def fetch_options_listing(symbol: str):
    url = base_options_url()
    # YH Finance (Steady/Yahoo-Rapid) часто поддерживает display=list/straddle
    for disp in ("list", "straddle", None):
        try:
            params = {"ticker": symbol}
            if disp: params["display"] = disp
            return _norm_top(_get(url, params))
        except Exception:
            continue
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
            body = raw.get("body")
            if isinstance(body, list) and body:
                node = body[0]
                price = node.get("regularMarketPrice") or node.get("price") or node.get("last")
                ts    = node.get("regularMarketTime")  or node.get("time")  or node.get("timestamp")
                q = {}
                if isinstance(price, (int,float)): q["regularMarketPrice"] = float(price)
                if ts is not None: q["regularMarketTime"] = _to_sec(ts)
                if q: return q
        except Exception:
            continue
    return {}

def quote_from_options(raw: dict) -> dict:
    """Пытаемся достать quote прямо из options-листа (как в твоём debug)."""
    q = {}
    body = raw.get("body")
    if isinstance(body, list) and body:
        b0 = body[0]
        # иногда quote лежит прямо в этом объекте
        qnode = b0.get("quote")
        if isinstance(qnode, dict):
            for key in ("regularMarketPrice","last","underlyingPrice","price"):
                v = qnode.get(key)
                if isinstance(v,(int,float)): q["regularMarketPrice"]=float(v); break
            for key in ("regularMarketTime","time","timestamp"):
                v = qnode.get(key)
                if v is not None: q["regularMarketTime"]=_to_sec(v); break
        # fallback — попробуем взять из самого b0
        if "regularMarketPrice" not in q:
            for key in ("regularMarketPrice","last","underlyingPrice","price"):
                v = b0.get(key)
                if isinstance(v,(int,float)): q["regularMarketPrice"]=float(v); break
        if "regularMarketTime" not in q:
            for key in ("regularMarketTime","time","timestamp"):
                v = b0.get(key)
                if v is not None: q["regularMarketTime"]=_to_sec(v); break
    return q

def ensure_shape(raw: dict):
    """
    Возвращает нормализованную структуру:
    {
      "expirationDates": [int(sec), ...],
      "chains": [ {"expiration": sec, "calls": [...], "puts":[...]}, ... ],
      "quote": {...}
    }
    """
    raw = _norm_top(raw)
    body = raw.get("body")
    quote = quote_from_options(raw)

    expirationDates, chains = [], []

    if isinstance(body, list) and body:
        b0 = body[0]

        # даты экспираций (по твоему debug лежат в body[0].expirationDates)
        exps = b0.get("expirationDates") or b0.get("expirations") or b0.get("dates")
        if isinstance(exps, list):
            expirationDates = [_to_sec(x) for x in exps if isinstance(x,(int,float))]

        # список опционов лежит в body[0].options — САМЫЙ ВАЖНЫЙ МОМЕНТ
        options_list = b0.get("options", [])
        if isinstance(options_list, list):
            for opt in options_list:
                exp = _to_sec(opt.get("expirationDate"))
                calls = opt.get("calls", []) or []
                puts  = opt.get("puts",  []) or []
                if exp:
                    chains.append({"expiration": exp, "calls": calls, "puts": puts})

    # дедуп и сортировка
    seen = set(); uniq = []
    for ch in chains:
        key = (ch["expiration"], len(ch["calls"]), len(ch["puts"]))
        if key not in seen:
            seen.add(key); uniq.append(ch)
    chains = sorted(uniq, key=lambda z: z["expiration"])

    return {"expirationDates": [e for e in expirationDates if e], "chains": chains, "quote": quote}

def fetch_expiry(symbol: str, epoch: int):
    url = base_options_url()
    last = None
    for disp in ("list", "straddle", None):
        try:
            p = {"ticker": symbol, "expiration": epoch}
            if disp: p["display"] = disp
            raw = _norm_top(_get(url, p))
            last = raw
            shaped = ensure_shape(raw)
            # берём цепочку с совпадающей датой
            for ch in shaped["chains"]:
                if ch["expiration"] == epoch:
                    q = shaped.get("quote", {})
                    if "regularMarketPrice" not in q or "regularMarketTime" not in q:
                        q2 = fetch_quote(symbol)
                        q.update({k:v for k,v in q2.items() if k not in q})
                    if "regularMarketTime" not in q or q["regularMarketTime"] is None:
                        q["regularMarketTime"] = int(time.time())
                    return {"quote": q, "chain": ch}
        except Exception:
            continue
    # fallback: вернём пустую цепочку, но с попыткой достать котировку
    q = quote_from_options(last or {}) or fetch_quote(symbol) or {"regularMarketTime": int(time.time())}
    return {"quote": q, "chain": {"expiration": epoch, "calls": [], "puts": []}}

# ========== Математика ==========
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
            gamma = bsm_gamma(S, float(K), float(iv), tau)
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

    oi_by_exp = df.groupby("expiry")["oi"].sum().rename("exp_oi") if "expiry" in df.columns else pd.Series(dtype=float)
    if not oi_by_exp.empty:
        df = df.merge(oi_by_exp, on="expiry", how="left")
        total_oi = float(df["oi"].sum()) or 1.0
        df["OI_share"] = (df["exp_oi"]/total_oi).fillna(0.0)
    else:
        df["OI_share"] = 0.0

    sig = df["iv"].clip(lower=1e-6)
    root_tau = np.sqrt(df["tau"].clip(lower=1e-9))
    denom = 2.0*(kappa**2)*(sig*root_tau)**2
    logt = np.log(np.maximum(df["strike"], 1e-6)/max(S,1e-6))
    df["W_dist"] = np.exp(-(logt**2)/np.maximum(denom, 1e-12))

    df["W_liq"]  = np.sqrt(df["OI_share"].clip(lower=0))
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

# ========== Интерэктив: загрузка и расчёт ==========
if btn_load:
    try:
        raw = fetch_options_listing(ticker)

        # Debug-раздел + возможность скачать JSON
        with st.expander("Debug: сырой ответ провайдера", expanded=False):
            st.json(raw)
            st.download_button(
                "Скачать Debug JSON",
                data=json.dumps(raw, ensure_ascii=False, indent=2),
                file_name=f"debug_{ticker}.json",
                mime="application/json"
            )

        shaped = ensure_shape(raw)
        exp_dates = shaped.get("expirationDates")
        # fallback: если expirationDates нет, пробуем взять из chains
        if not exp_dates:
            exp_dates = sorted({ ch.get("expiration") for ch in shaped.get("chains", []) if ch.get("expiration") })
        if not exp_dates:
            st.error("Не удалось получить список экспираций (пустой ответ).")
            st.stop()


        # Save to session state and show instruction
        st.session_state.raw_listing = raw
        st.session_state.exp_dates = exp_dates
        st.session_state.ticker_loaded = ticker
        st.session_state.exp_idx = 0
        st.success(f"Экспирации загружены для {ticker} — выбери дату слева и нажми «Рассчитать уровни…».")

    except Exception as e:
        st.error(f"Ошибка загрузки экспираций: {e}")


# === Standalone selection and calculation section (visible whenever expiries are loaded) ===
exp_dates = st.session_state.get("exp_dates", [])
if exp_dates:
    try:
        human = [time.strftime("%Y-%m-%d", time.gmtime(int(e))) for e in exp_dates]
    except Exception:
        human = [str(e) for e in exp_dates]
    st.session_state.exp_idx = st.selectbox(
        "Выбери ближайшую экспирацию",
        list(range(len(exp_dates))),
        format_func=lambda i: human[i],
        index=st.session_state.get("exp_idx", 0),
        key="exp_select_box"
    )

    
if st.button("Рассчитать уровни (эта + 7 следующих)", key="calc_levels_btn"):
    try:
        idx = st.session_state.exp_idx
        picked = exp_dates[idx: idx + 8]
        log_box = st.empty()
        progress = st.progress(0.0)
        all_rows, S_ref = [], None
        per_exp_info = []
        df_selected = None  # цепочка для выбранной (первой) экспирации

        with st.spinner("Тянем цепочки и считаем Гамму..."):
            for j, e in enumerate(picked, start=1):
                dat = fetch_expiry(st.session_state.ticker_loaded or ticker, int(e))
                df_i, S_i = compute_chain_gex(dat["chain"], dat["quote"])
                if j == 1:
                    df_selected = df_i.copy()
                calls_n = len(dat["chain"].get("calls", []))
                puts_n  = len(dat["chain"].get("puts",  []))
                per_exp_info.append(f"{time.strftime('%Y-%m-%d', time.gmtime(int(e)))}: calls={calls_n}, puts={puts_n}, rows={len(df_i)}")
                if not df_i.empty:
                    df_i["expiry"] = int(e)
                    all_rows.append(df_i)
                    if S_ref is None and S_i is not None:
                        S_ref = S_i
                progress.progress(j / max(1, len(picked)))
                log_box.write("\n".join(per_exp_info))

        if not all_rows:
            st.warning("По выбранному окну экспираций цепочки пустые (нет строк для расчёта). Попробуй другую дату или тикер.")
            st.stop()

        if S_ref is None:
            q = fetch_quote(st.session_state.ticker_loaded or ticker)
            S_ref = q.get("regularMarketPrice")
        if S_ref is None:
            st.error("Не удалось определить Spot (regularMarketPrice).")
            st.stop()

        df_all = pd.concat(all_rows, ignore_index=True)
        prof   = weight_and_profile(df_all, S=S_ref, h_days=H_EXP, kappa=KAPPA, smooth=SMOOTH)
        if prof.empty:
            st.warning("Профиль пуст. Проверь ликвидность и параметры весов/сглаживания.")
            st.stop()

        flips, pos, neg = find_levels(prof)

        c1, c2 = st.columns([2, 1])
        with c1:
            title = f"({st.session_state.ticker_loaded or ticker}, c {time.strftime('%Y-%m-%d', time.gmtime(int(picked[0])))} по {time.strftime('%Y-%m-%d', time.gmtime(int(picked[-1])))} )"
            st.plotly_chart(
                plot_profiles(prof, S=S_ref, flips=flips, pos=pos, neg=neg, title_note=title),
                use_container_width=True
            )

        def rows(mags, label):
            return [{"Strike": k, "Magnet": v, "|Magnet|": a, "Side": label} for (k, v, a) in mags]
        levels = pd.DataFrame(rows(pos, "+") + rows(neg, "-")).sort_values("|Magnet|", ascending=False)

        with c2:
            st.subheader("Ключевые уровни (магниты)")
            st.dataframe(levels, use_container_width=True)
            st.download_button(
                "Скачать уровни (CSV)",
                data=levels.to_csv(index=False).encode("utf-8"),
                file_name=f"{st.session_state.ticker_loaded or ticker}_magnet_levels.csv",
                mime="text/csv"
            )

        # --- Таблица Net GEX по страйкам для выбранной экспирации ---
        if df_selected is not None and not df_selected.empty:
            df_p = df_selected.copy()
            df_p["type"] = df_p["type"].str.lower().map({"call":"call","put":"put"})
            gex_by = df_p.groupby(["strike","type"])["gex_signed"].sum().unstack(fill_value=0.0)
            for col in ("call","put"):
                if col not in gex_by.columns:
                    gex_by[col] = 0.0
            gex_by = gex_by.rename(columns={"call":"GEX_call","put":"GEX_put"})
            gex_by["Net_GEX"] = gex_by["GEX_call"] + gex_by["GEX_put"]
            gex_by["|Net_GEX|"] = gex_by["Net_GEX"].abs()
            gex_table = gex_by.reset_index().sort_values("strike")
            st.subheader("Net GEX по страйкам для выбранной экспирации")
            st.dataframe(gex_table, use_container_width=True)
            st.download_button(
                "Скачать Net GEX по страйкам (CSV)",
                data=gex_table.to_csv(index=False).encode("utf-8"),
                file_name=f"{st.session_state.ticker_loaded or ticker}_{time.strftime('%Y-%m-%d', time.gmtime(int(picked[0])))}_netgex_by_strike.csv",
                mime="text/csv"
            )

# --- Диаграмма Net GEX по страйкам (как на примере MaxPower) ---
try:
    # агрегаты по OI/Volume и Net GEX
    df_p = df_selected.copy()
    if "volume" not in df_p.columns:
        df_p["volume"] = 0.0  # на случай отсутствия поля в данных
    grp_gex = df_p.groupby(["strike","type"])["gex_signed"].sum().unstack(fill_value=0.0)
    grp_oi  = df_p.groupby(["strike","type"])["oi"].sum().unstack(fill_value=0.0)
    grp_vol = df_p.groupby(["strike","type"])["volume"].sum().unstack(fill_value=0.0)

    for g in (grp_gex, grp_oi, grp_vol):
        for col in ("call","put"):
            if col not in g.columns:
                g[col] = 0.0

    bar_df = grp_gex.rename(columns={"call":"GEX_call","put":"GEX_put"})
    bar_df["Net_GEX"] = bar_df["GEX_call"] + bar_df["GEX_put"]
    oi_df  = grp_oi.rename(columns={"call":"Call_OI","put":"Put_OI"})
    vol_df = grp_vol.rename(columns={"call":"Call_Volume","put":"Put_Volume"})

    merged = bar_df.join(oi_df).join(vol_df).reset_index().sort_values("strike")
    x = merged["strike"].values
    y = merged["Net_GEX"].values

    # подготовим customdata для всплывающей подсказки
    import numpy as np
    custom = np.stack([
        merged["strike"].values,
        merged["Call_OI"].values,
        merged["Put_OI"].values,
        merged["Call_Volume"].values,
        merged["Put_Volume"].values,
        merged["Net_GEX"].values
    ], axis=1)

    import plotly.graph_objects as go
    fig = go.Figure()

    # отдельные следы для положительных и отрицательных значений (цвета как на примере)
    pos_mask = (y >= 0)
    neg_mask = ~pos_mask
    fig.add_bar(x=x[pos_mask], y=y[pos_mask], name="Net GEX +", customdata=custom[pos_mask],
                hovertemplate=(
                    "Strike: %{customdata[0]:.0f}<br>" +
                    "Call OI: %{customdata[1]:,.0f}<br>" +
                    "Put OI: %{customdata[2]:,.0f}<br>" +
                    "Call Volume: %{customdata[3]:,.0f}<br>" +
                    "Put Volume: %{customdata[4]:,.0f}<br>" +
                    "Net GEX: %{customdata[5]:,.1f}<extra></extra>"
                ))
    fig.add_bar(x=x[neg_mask], y=y[neg_mask], name="Net GEX -", customdata=custom[neg_mask],
                hovertemplate=(
                    "Strike: %{customdata[0]:.0f}<br>" +
                    "Call OI: %{customdata[1]:,.0f}<br>" +
                    "Put OI: %{customdata[2]:,.0f}<br>" +
                    "Call Volume: %{customdata[3]:,.0f}<br>" +
                    "Put Volume: %{customdata[4]:,.0f}<br>" +
                    "Net GEX: %{customdata[5]:,.1f}<extra></extra>"
                ))

    # вертикальная линия спота и подпись
    spot_x = float(S_ref)
    fig.add_vline(x=spot_x, line_width=2, line_dash="solid")
    fig.add_annotation(x=spot_x, y=1.02, yref="paper", showarrow=False,
                       text=f"Price: {spot_x:.2f}")

    fig.update_layout(
        barmode="relative",
        showlegend=False,
        margin=dict(l=40, r=20, t=30, b=40),
        xaxis_title="Strike",
        yaxis_title="Net GEX",
    )
    st.plotly_chart(fig, use_container_width=True)
except Exception as _e:
    st.info(f"Не удалось построить диаграмму Net GEX по страйкам: {_e}")


    except Exception as e:
        st.error(f"Ошибка расчёта уровней: {e}")
