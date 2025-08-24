# streamlit_app.py (v3_AG_vol)
import time, json, io, zipfile
import requests
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# --- helper for scaling series to the right y-axis height ---
def _scale_to_y2(arr, top):
    import numpy as _np
    if arr is None:
        return None
    try:
        arr = _np.asarray(arr, dtype=float)
    except Exception:
        return arr
    if arr.size == 0:
        return arr
    m = float(_np.nanmax(arr))
    if not _np.isfinite(m) or m <= 0 or top is None:
        return arr
    try:
        top_val = float(top)
    except Exception:
        return arr
    if not _np.isfinite(top_val) or top_val <= 0:
        return arr
    return arr * (top_val / m)



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
        ALPHA_PCT = st.slider("Порог значимости |Net GEX| от max, %", 0.0, 10.0, 2.0, 0.5)

# ========== RapidAPI helpers ==========
def api_headers():
    return {"X-RapidAPI-Key": RAPIDAPI_KEY, "X-RapidAPI-Host": RAPIDAPI_HOST}

def base_options_url():
    return f"https://{RAPIDAPI_HOST}/api/v1/markets/options"

def _get(url, params):
    r = requests.get(url, headers=api_headers(), params=params, timeout=30)
    r.raise_for_status()
    return r.json()

def _norm_top(raw):
    return {"body": raw} if isinstance(raw, list) else raw

def _to_sec(x):
    if x is None: return None
    try:
        x = int(x)
    except Exception:
        return None
    return x//1000 if x > 10**12 else x  # ms -> s

def fetch_options_listing(symbol: str):
    url = base_options_url()
    for disp in ("list", "straddle", None):
        try:
            params = {"ticker": symbol}
            if disp:
                params["display"] = disp
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
            body = raw.get("body") or raw
            if isinstance(body, list) and body:
                q = body[0]
            elif isinstance(body, dict):
                q = body.get("quotes") or body.get("quote") or body
                if isinstance(q, list) and q:
                    q = q[0]
            else:
                q = {}
            out = {}
            for k in ("regularMarketPrice","regularMarketTime"):
                if isinstance(q, dict) and k in q:
                    out[k] = q[k]
            return out
        except Exception:
            continue
    return {}

def quote_from_options(raw: dict) -> dict:
    raw = _norm_top(raw)
    body = raw.get("body")
    q = {}
    if isinstance(body, list) and body:
        b0 = body[0]
        if isinstance(b0, dict):
            q0 = b0.get("quote") or {}
            if isinstance(q0, dict):
                for k in ("regularMarketPrice","regularMarketTime"):
                    if k in q0: q[k] = q0[k]
    return q

def ensure_shape(raw: dict):
    raw = _norm_top(raw)
    body = raw.get("body")
    quote = quote_from_options(raw)

    expirationDates, chains = [], []

    if isinstance(body, list) and body:
        b0 = body[0]
        exps = b0.get("expirationDates") or b0.get("expirations") or b0.get("dates")
        if isinstance(exps, list):
            expirationDates = [_to_sec(x) for x in exps if _to_sec(x)]
        options_list = b0.get("options", [])
        if isinstance(options_list, list):
            for opt in options_list:
                exp = _to_sec(opt.get("expirationDate"))
                calls = opt.get("calls", []) or []
                puts  = opt.get("puts",  []) or []
                if exp:
                    chains.append({"expiration": exp, "calls": calls, "puts": puts})

    if not expirationDates:
        expirationDates = sorted({ ch.get("expiration") for ch in chains if ch.get("expiration") })

    return {"expirationDates": [e for e in expirationDates if e], "chains": chains, "quote": quote}

def fetch_expiry(symbol: str, epoch: int):
    url = base_options_url()
    last = None
    for disp in ("list", "straddle", None):
        try:
            p = {"ticker": symbol, "expiration": epoch}
            if disp:
                p["display"] = disp
            raw = _norm_top(_get(url, p))
            last = raw
            shaped = ensure_shape(raw)
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
    if last:
        shaped = ensure_shape(last)
        if shaped.get("chains"):
            ch = shaped["chains"][0]
            q = shaped.get("quote", {})
            if "regularMarketTime" not in q: q["regularMarketTime"] = int(time.time())
            return {"quote": q, "chain": ch}
    return {"quote": {}, "chain": {"expiration": epoch, "calls": [], "puts": []}}

# ========== Математика/методика ==========
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
            vol = r.get("volume", 0) or 0
            if not (K and iv) or K<=0:
                continue
            gamma = bsm_gamma(S, float(K), float(iv), tau)
            gex   = oi * gamma * 100 * S * sign
            rows.append({"strike": float(K), "type": typ, "oi": float(oi), "iv": float(iv),
                         "volume": float(vol),
                         "tau": float(tau), "gex_signed": float(gex)})
    df = pd.DataFrame(rows)
    return df, float(S)

def weight_and_profile(df: pd.DataFrame, S: float, h_days: float, kappa: float, smooth: int):
    if df.empty: return pd.DataFrame()
    df = df.copy()
    df["W_exp"] = np.exp(-df["tau"]*365.0/h_days)

    oi_by_exp = df.groupby("tau")["oi"].sum().rename("exp_oi").reset_index()
    if not oi_by_exp.empty:
        df = df.merge(oi_by_exp, on="tau", how="left")
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
    if prof.empty: return [], [], []
    strikes = prof["strike"].values
    vals = prof["Magnet_smooth"].values
    flips = []
    for i in range(1, len(vals)):
        y0, y1 = vals[i-1], vals[i]
        if (y0 <= 0 and y1 > 0) or (y0 >= 0 and y1 < 0):
            flips.append(float(strikes[i]))
    abs_mag = np.abs(vals)
    order = np.argsort(-abs_mag)
    pos = [(float(strikes[i]), float(vals[i]), float(abs_mag[i])) for i in order if vals[i] > 0][:TOP_N_LEVELS]
    neg = [(float(strikes[i]), float(vals[i]), float(abs_mag[i])) for i in order if vals[i] < 0][:TOP_N_LEVELS]
    return flips, pos, neg

def plot_profiles(profile: pd.DataFrame, S: float, flips, pos, neg, title_note=""):
    prof = profile.copy()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=prof["strike"], y=prof["Magnet_smooth"], name="Magnet", mode="lines"))
    fig.add_trace(go.Scatter(x=prof["strike"], y=prof["NetGEX_smooth"], name="Net GEX (сглаж.)", mode="lines"))
    for f in flips or []:
        fig.add_vline(x=float(f), line_width=1, line_dash="dot", line_color="#888")
    fig.add_vline(x=float(S), line_width=2, line_dash="solid", line_color="#FFA500")
    fig.update_layout(
        title=title_note, showlegend=True,
        margin=dict(l=40,r=20,t=30,b=40),
        xaxis_title="Strike", yaxis_title="Value",
        dragmode=False
    )
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True, tickformat=",")
    return fig

# ========== Интерэктив: загрузка и расчёт ==========
if btn_load:
    try:
        raw = fetch_options_listing(ticker)

        with st.expander("Debug: сырой ответ провайдера", expanded=False):
            st.json(raw)
            st.download_button(
                "Скачать Debug JSON",
                data=json.dumps(raw, ensure_ascii=False, indent=2),
                file_name=f"debug_{ticker}.json",
                mime="application/json"
            )

        shaped = ensure_shape(raw)
        exp_dates = shaped.get("expirationDates", [])
        st.session_state.raw_listing = raw
        st.session_state.exp_dates = exp_dates
        st.session_state.ticker_loaded = ticker
        st.session_state.exp_idx = 0
        st.success(f"Экспирации загружены для {ticker} — выбери дату ниже и нажми «Рассчитать уровни…».")

        if not exp_dates:
            exp_dates = sorted({ ch.get("expiration") for ch in shaped.get("chains", []) if ch.get("expiration") })
            st.session_state.exp_dates = exp_dates
        if not exp_dates:
            st.error("Не удалось получить список экспираций (пустой ответ).")
            st.stop()

    except Exception as e:
        st.error(f"Ошибка: {e}")
        st.info("Открой «Debug: сырой ответ провайдера» и скачай JSON — пришли мне файл, если ошибка повторится.")

exp_dates = st.session_state.get("exp_dates", [])
if exp_dates:
    human = [time.strftime("%Y-%m-%d", time.gmtime(e)) for e in exp_dates]
    idx = st.selectbox("Выбери ближайшую экспирацию",
                       list(range(len(exp_dates))),
                       format_func=lambda i: human[i],
                       index=st.session_state.get("exp_idx", 0),
                       key="expiry_select_idx")
    st.session_state.exp_idx = idx

    show_ag = st.toggle("Показать абсолютную гамму (AG)", value=True)

    if st.button("Рассчитать уровни (эта + 7 следующих)", key="calc_levels_btn"):
        try:
            idx = st.session_state.get("exp_idx", 0)
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
                    calls_n = len(dat["chain"].get("calls", []))
                    puts_n  = len(dat["chain"].get("puts",  []))
                    per_exp_info.append(
                        f"{time.strftime('%Y-%m-%d', time.gmtime(int(e)))}: "
                        f"calls={calls_n}, puts={puts_n}, rows={len(df_i)}"
                    )
                    if j == 1:
                        df_selected = df_i.copy()
                    if S_i is not None:
                        S_ref = S_i
                    if not df_i.empty:
                        df_i["expiry"] = int(e)
                        all_rows.append(df_i)
                    progress.progress(j / max(1, len(picked)))
                    log_box.write("\n".join(per_exp_info))

            if not all_rows:
                st.warning("По выбранному окну экспираций данные пустые (нет строк для расчёта). Попробуй другую дату или тикер.")
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

            c_top, c_chart = st.columns([1, 2])

            if df_selected is not None and not df_selected.empty:
                df_p = df_selected.copy()
                # объёмы точно в числовой тип
                if "volume" not in df_p.columns:
                    df_p["volume"] = 0.0
                df_p["volume"] = pd.to_numeric(df_p["volume"], errors="coerce").fillna(0.0)
                df_p["type"] = df_p["type"].str.lower().map({"call":"call","put":"put"})

                grp_gex = df_p.groupby(["strike","type"])["gex_signed"].sum().unstack(fill_value=0.0)
                grp_oi  = df_p.groupby(["strike","type"])["oi"].sum().unstack(fill_value=0.0)
                grp_vol = df_p.groupby(["strike","type"])["volume"].sum().unstack(fill_value=0.0)
                for g in (grp_gex, grp_oi, grp_vol):
                    for col in ("call","put"):
                        if col not in g.columns:
                            g[col] = 0.0

                bar_df = grp_gex.rename(columns={"call":"GEX_call","put":"GEX_put"})
                bar_df["Net_GEX"] = bar_df["GEX_call"] + bar_df["GEX_put"]
                bar_df["AG"] = bar_df["GEX_call"].abs() + bar_df["GEX_put"].abs()

                oi_df  = grp_oi.rename(columns={"call":"Call_OI","put":"Put_OI"})
                vol_df = grp_vol.rename(columns={"call":"Call_Volume","put":"Put_Volume"})
                merged = bar_df.join(oi_df).join(vol_df).reset_index().sort_values("strike")

                gex_table = merged[["strike", "GEX_call", "GEX_put", "Net_GEX", "AG", "Call_OI", "Put_OI", "Call_Volume", "Put_Volume"]].copy()
                gex_table["|Net_GEX|"] = gex_table["Net_GEX"].abs()

                with c_top:
                    st.subheader("Net GEX / Absolute Gamma по страйкам")
                    st.dataframe(gex_table, use_container_width=True, height=260)
                    st.download_button(
                        "Скачать Net GEX по страйкам (CSV)",
                        data=gex_table.to_csv(index=False).encode("utf-8"),
                        file_name=f"{st.session_state.ticker_loaded or ticker}_{time.strftime('%Y-%m-%d', time.gmtime(int(picked[0])))}_netgex_by_strike.csv",
                        mime="text/csv"
                    )

                # --- Диаграмма Net GEX по страйкам + AG overlay ---
                x = merged["strike"].to_numpy()
                y = merged["Net_GEX"].to_numpy()
                ag = merged["AG"].to_numpy()

                custom = np.stack([
                    merged["strike"].to_numpy(),
                    merged["Call_OI"].to_numpy(),
                    merged["Put_OI"].to_numpy(),
                    merged["Call_Volume"].to_numpy(),
                    merged["Put_Volume"].to_numpy(),
                    merged["Net_GEX"].to_numpy(),
                    merged["AG"].to_numpy()
                ], axis=1)

                abs_y = np.abs(y)
                if abs_y.size:
                    max_abs = float(abs_y.max())
                    alpha = float(ALPHA_PCT)/100.0
                    sig = abs_y >= (alpha * max_abs)
                    if sig.any():
                        idxs = np.where(sig)[0]
                        i0, i1 = max(0, int(idxs.min())-3), min(len(x)-1, int(idxs.max())+3)
                        x, y, ag, custom = x[i0:i1+1], y[i0:i1+1], ag[i0:i1+1], custom[i0:i1+1]

                tickvals = x.tolist()
                ticktext = [str(int(v)) if float(v).is_integer() else f"{v:g}" for v in x]

                pos_mask = (y >= 0)
                neg_mask = ~pos_mask

                fig = go.Figure()
                fig.add_bar(
                    x=x[pos_mask], y=y[pos_mask], name="Net GEX +", customdata=custom[pos_mask],
                    marker_color="#33B5FF",
                    hovertemplate=(
                        "Strike: %{customdata[0]:.0f}<br>"
                        "Call OI: %{customdata[1]:,.0f}<br>"
                        "Put OI: %{customdata[2]:,.0f}<br>"
                        "Call Volume: %{customdata[3]:,.0f}<br>"
                        "Put Volume: %{customdata[4]:,.0f}<br>"
                        "Net GEX: %{customdata[5]:,.1f}<extra></extra>"
                    )
                )
                fig.add_bar(
                    x=x[neg_mask], y=y[neg_mask], name="Net GEX -", customdata=custom[neg_mask],
                    marker_color="#FF3B30",
                    hovertemplate=(
                        "Strike: %{customdata[0]:.0f}<br>"
                        "Call OI: %{customdata[1]:,.0f}<br>"
                        "Put OI: %{customdata[2]:,.0f}<br>"
                        "Call Volume: %{customdata[3]:,.0f}<br>"
                        "Put Volume: %{customdata[4]:,.0f}<br>"
                        "Net GEX: %{customdata[5]:,.1f}<extra></extra>"
                    )
                )

                if show_ag and ag.size:
                    fig.add_trace(go.Scatter(
                        x=x, y=ag, yaxis="y2", name="AG",
                        mode="lines+markers",
                        line=dict(color="#B366FF"),
                        fill="tozeroy",
                        fillcolor="rgba(179,102,255,0.25)",
                        line_shape="spline",
                        hovertemplate=(
                            "Strike: %{customdata[0]:.1f}<br>"
                            "Call OI: %{customdata[1]:,.0f}<br>"
                            "Put OI: %{customdata[2]:,.0f}<br>"
                            "Call Volume: %{customdata[3]:,.0f}<br>"
                            "Put Volume: %{customdata[4]:,.0f}<br>"
                            "AG: %{customdata[6]:,.1f}<extra></extra>"
                        ),
                        customdata=custom
                    ))

                # --- Настройка правой оси: по умолчанию OI/Volume; если AG выбран и кратно больше — AG ---
                y2_ag_max = float(np.max(ag)) if (show_ag and ag.size) else 0.0
                _cand = []
                for _col in ["Call_OI", "Put_OI", "Call_Volume", "Put_Volume"]:
                    if _col in merged.columns:
                        _a = merged[_col].to_numpy()
                        if _a.size:
                            _cand.append(_a)
                y2_oi_vol_max = float(np.max([np.max(a) for a in _cand])) if _cand else 0.0
                # если AG хотя бы в 3 раза больше — выбираем шкалу AG
                _use_ag_axis = (y2_ag_max > 0 and y2_ag_max >= 3.0 * max(y2_oi_vol_max, 1e-9))
                y2max = y2_ag_max if _use_ag_axis else y2_oi_vol_max
                y2_title = "AG" if _use_ag_axis else "OI / Volume"

                # масштабы осей перед оверлеями
                ymax = float(np.abs(y).max()) if y.size else 0.0
                y2_candidates = []
                if ag.size:
                    y2_candidates.append(ag)
                for _col in ["Call_OI", "Put_OI", "Call_Volume", "Put_Volume"]:
                    if _col in merged.columns:
                        _arr = merged[_col].to_numpy()
                        if _arr.size:
                            y2_candidates.append(_arr)
                y2max = float(np.max([np.max(a) for a in y2_candidates])) if y2_candidates else 0.0

                # Оверлеи по правой оси: OI & Volume
                call_oi  = merged["Call_OI"].to_numpy()     if "Call_OI"     in merged.columns else None
                put_oi   = merged["Put_OI"].to_numpy()      if "Put_OI"      in merged.columns else None
                call_vol = merged["Call_Volume"].to_numpy() if "Call_Volume" in merged.columns else None
                put_vol  = merged["Put_Volume"].to_numpy()  if "Put_Volume"  in merged.columns else None
                if call_oi is not None:
                    fig.add_trace(go.Scatter(
                        x=x, y=call_oi, yaxis="y2", name="Call OI",
                        mode="lines", line=dict(width=2, color="#1ABC9C")
                    ))
                if put_oi is not None:
                    fig.add_trace(go.Scatter(
                        x=x, y=put_oi, yaxis="y2", name="Put OI",
                        mode="lines", line=dict(width=2, color="#F39C12")
                    ))
                if call_vol is not None:
                    fig.add_trace(go.Scatter(
                        x=x, y=call_vol, yaxis="y2", name="Call Volume",
                        mode="lines", line=dict(width=1, dash="dot", color="#95A5A6")
                    ))
                if put_vol is not None:
                    fig.add_trace(go.Scatter(
                        x=x, y=put_vol, yaxis="y2", name="Put Volume",
                        mode="lines", line=dict(width=1, dash="dot", color="#E91E63")
                    ))


                
                # масштабы осей перед оверлеями

                spot_x = float(S_ref)
                fig.add_vline(x=spot_x, line_width=2, line_dash="solid", line_color="#FFA500")
                xmin, xmax = float(np.min(x)), float(np.max(x))
                mid = 0.5*(xmin + xmax)
                _xanchor, _xshift = ('left', 8) if spot_x <= mid else ('right', -8)
                fig.add_annotation(x=spot_x, y=1.02, xref="x", yref="paper", showarrow=False,
                                   xanchor=_xanchor, xshift=_xshift,
                                   text=f"Price: {spot_x:.2f}", font=dict(color="#FFA500"))

                ymax = float(np.abs(y).max()) if y.size else 0.0
                y2_candidates = []
                if ag.size:
                    y2_candidates.append(ag)
                for _col in ["Call_OI", "Put_OI", "Call_Volume", "Put_Volume"]:
                    if _col in merged.columns:
                        _arr = merged[_col].to_numpy()
                        if _arr.size:
                            y2_candidates.append(_arr)
                y2max = float(np.max([np.max(a) for a in y2_candidates])) if y2_candidates else 0.0


                fig.update_layout(
                    barmode="relative",
                    showlegend=True,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    margin=dict(l=40, r=40, t=40, b=40),
                    xaxis_title="Strike",
                    yaxis_title="Net GEX",
                    dragmode=False,
                    yaxis2=dict(title=y2_title, overlaying="y", side="right", showgrid=False, tickformat=","),
                )
                fig.update_xaxes(tickmode="array", tickvals=tickvals, ticktext=ticktext, tickangle=0)
                fig.update_xaxes(fixedrange=True)
                if ymax > 0:
                    fig.update_yaxes(range=[-1.2*ymax, 1.2*ymax])
                if y2max > 0:
                    fig.update_layout(yaxis2=dict(range=[0, 1.2*y2max], title=y2_title, overlaying="y", side="right", showgrid=False, tickformat=","))

                fig.update_yaxes(fixedrange=True, tickformat=",")
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False, "scrollZoom": False})

                # Кнопка для скачивания debug.zip
                try:
                    buf = io.BytesIO()
                    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
                        if st.session_state.get("raw_listing"):
                            zf.writestr("raw_listing.json", json.dumps(st.session_state.raw_listing, ensure_ascii=False, indent=2))
                        if S_ref is not None:
                            zf.writestr("quote.json", json.dumps({"regularMarketPrice": float(S_ref)}, ensure_ascii=False, indent=2))
                        try:
                            zf.writestr("netgex_by_strike.csv", gex_table.to_csv(index=False))
                        except Exception:
                            pass
                    st.download_button(
                        "Скачать debug.zip",
                        data=buf.getvalue(),
                        file_name=f"{st.session_state.ticker_loaded or ticker}_debug.zip",
                        mime="application/zip"
                    )
                except Exception:
                    pass

            tk = st.session_state.ticker_loaded or ticker
            title = (
                f"({tk}, c {time.strftime('%Y-%m-%d', time.gmtime(int(picked[0])))} "
                f"по {time.strftime('%Y-%m-%d', time.gmtime(int(picked[-1])))} )"
            )
            prof_fig = plot_profiles(prof, S=S_ref, flips=flips, pos=pos, neg=neg, title_note=title)
            st.plotly_chart(prof_fig, use_container_width=True)

        except Exception as e:
            st.error(f"Ошибка расчёта уровней: {e}")
else:
    st.info("Чтобы начать, введите тикер и нажмите «Загрузить экспирации».")
