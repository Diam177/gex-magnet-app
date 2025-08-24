
import io
import json
import math
from typing import Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st


st.set_page_config(page_title="Net GEX • Power Zone", layout="wide")
st.title("Net GEX • Power Zone & OI/Volume Overlays")


def _to_float_series(x):
    return pd.to_numeric(pd.Series(x), errors="coerce").fillna(0.0)


def smooth_series(y: pd.Series, window: int = 3) -> np.ndarray:
    s = pd.Series(y, dtype=float)
    sm = s.rolling(window=window, center=True, min_periods=1).mean()
    return sm.values


def compute_power_zone(df: pd.DataFrame, spot: float,
                       a0: float = 0.2349, a1: float = 0.1266, beta: float = 0.0134) -> pd.Series:
    """
    Power Zone = AG * (a0 + a1 * Volume/OI) * exp(-beta * |K - Spot|).
    Volume = Call_Volume + Put_Volume; OI = Call_OI + Put_OI.
    """
    ag = _to_float_series(df.get("AG", 0))
    call_oi = _to_float_series(df.get("Call_OI", 0))
    put_oi = _to_float_series(df.get("Put_OI", 0))
    call_vol = _to_float_series(df.get("Call_Volume", 0))
    put_vol = _to_float_series(df.get("Put_Volume", 0))

    vol_sum = call_vol + put_vol
    oi_sum = call_oi + put_oi
    ratio = pd.Series(np.divide(vol_sum, oi_sum, out=np.zeros_like(vol_sum), where=oi_sum != 0.0))

    liq_factor = a0 + a1 * ratio
    proximity = np.exp(-beta * np.abs(_to_float_series(df["strike"]) - float(spot)))
    power_zone = ag * liq_factor * proximity
    return pd.Series(power_zone, name="Power_Zone")


def add_overlay(fig, x, y, name, color_rgba, hover_suffix="", secondary_y=True, smooth_window=3):
    y_smooth = smooth_series(pd.Series(y, dtype=float), window=smooth_window)
    fill_color = color_rgba.replace(" 1.0)", " 0.35)")
    fig.add_trace(
        go.Scatter(
            x=x, y=y_smooth,
            mode="lines+markers",
            name=name,
            line=dict(width=2, color=color_rgba),
            marker=dict(size=5, color=color_rgba),
            fill="tozeroy",
            fillcolor=fill_color,
            hovertemplate="%{x}<br>%{y:,.1f}" + hover_suffix + "<extra>" + name + "</extra>",
        ),
        secondary_y=secondary_y
    )


def build_chart(df: pd.DataFrame, spot: float,
                show_pz: bool, show_call_oi: bool, show_put_oi: bool,
                show_call_vol: bool, show_put_vol: bool) -> go.Figure:

    net = _to_float_series(df.get("Net_GEX", 0))
    strikes = df["strike"].astype(float)

    pos_mask = net >= 0
    neg_mask = ~pos_mask

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Negative bars (red)
    fig.add_trace(go.Bar(
        x=strikes[neg_mask], y=net[neg_mask],
        name="Net GEX−", marker_color="rgba(214, 39, 40, 0.85)",
        hovertemplate=(
            "Strike: %{x}<br>"
            "Call OI: %{customdata[0]:,.0f}<br>"
            "Put OI: %{customdata[1]:,.0f}<br>"
            "Call Volume: %{customdata[2]:,.0f}<br>"
            "Put Volume: %{customdata[3]:,.0f}<br>"
            "Net GEX: %{y:,.1f}<extra></extra>"
        ),
        customdata=np.c_[
            _to_float_series(df["Call_OI"])[neg_mask],
            _to_float_series(df["Put_OI"])[neg_mask],
            _to_float_series(df["Call_Volume"])[neg_mask],
            _to_float_series(df["Put_Volume"])[neg_mask],
        ]
    ), secondary_y=False)

    # Positive bars (blue)
    fig.add_trace(go.Bar(
        x=strikes[pos_mask], y=net[pos_mask],
        name="Net GEX+", marker_color="rgba(82, 182, 255, 0.9)",
        hovertemplate=(
            "Strike: %{x}<br>"
            "Call OI: %{customdata[0]:,.0f}<br>"
            "Put OI: %{customdata[1]:,.0f}<br>"
            "Call Volume: %{customdata[2]:,.0f}<br>"
            "Put Volume: %{customdata[3]:,.0f}<br>"
            "Net GEX: %{y:,.1f}<extra></extra>"
        ),
        customdata=np.c_[
            _to_float_series(df["Call_OI"])[pos_mask],
            _to_float_series(df["Put_OI"])[pos_mask],
            _to_float_series(df["Call_Volume"])[pos_mask],
            _to_float_series(df["Put_Volume"])[pos_mask],
        ]
    ), secondary_y=False)

    # overlays
    COL_PZ      = "rgba(255, 215,   0, 1.0)"
    COL_CALLVOL = "rgba( 31, 119, 180, 1.0)"
    COL_PUTVOL  = "rgba(255, 127,  14, 1.0)"
    COL_CALL_OI = "rgba( 44, 160,  44, 1.0)"
    COL_PUT_OI  = "rgba(214,  39,  40, 1.0)"

    if show_pz and "Power_Zone" in df.columns:
        add_overlay(fig, strikes, df["Power_Zone"], "Power Zone", COL_PZ, hover_suffix="")

    if show_call_vol:
        add_overlay(fig, strikes, _to_float_series(df["Call_Volume"]), "Call Volume", COL_CALLVOL, hover_suffix=" (vol)")

    if show_put_vol:
        add_overlay(fig, strikes, _to_float_series(df["Put_Volume"]), "Put Volume", COL_PUTVOL, hover_suffix=" (vol)")

    if show_call_oi:
        add_overlay(fig, strikes, _to_float_series(df["Call_OI"]), "Call OI", COL_CALL_OI, hover_suffix=" (OI)")

    if show_put_oi:
        add_overlay(fig, strikes, _to_float_series(df["Put_OI"]), "Put OI", COL_PUT_OI, hover_suffix=" (OI)")

    # vertical price line
    fig.add_vline(x=spot, line_color="#f5a623", line_width=2)
    yref = (net.max() if net.max() > 0 else 0) * 1.15 + (abs(net.min()) if net.min() < 0 else 0) * 0.02
    fig.add_annotation(
        x=spot, y=yref,
        text=f"Price: {spot:.2f}",
        showarrow=False,
        font=dict(color="#f5a623"),
        xanchor="left", xshift=12
    )

    fig.update_layout(
        barmode="relative",
        margin=dict(l=40, r=40, t=30, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis_title="Strike",
        yaxis_title="Net GEX",
        template="plotly_dark"
    )
    fig.update_yaxes(secondary_y=True, showgrid=False, tickformat="~s")
    return fig


def make_demo_df(center: float = 645.0) -> Tuple[pd.DataFrame, float]:
    strikes = np.arange(center - 19, center + 20, 1.0)
    rng = np.random.default_rng(42)
    net = np.where(strikes < center - 5, -rng.normal(6000, 2500, len(strikes)),
                   rng.normal(9000, 4500, len(strikes)))
    net = pd.Series(net).rolling(3, center=True, min_periods=1).mean().values

    ag = 1.2e6 * np.exp(-0.12 * (strikes - center)**2) + \
         3.2e5 * np.exp(-0.7 * (strikes - (center+5))**2) + \
         4.0e5 * np.exp(-0.5 * (strikes - (center-15))**2)

    call_oi = np.abs(200 + 7000 * np.exp(-0.35 * (strikes - (center+5))**2)).astype(int)
    put_oi  = np.abs(200 + 6000 * np.exp(-0.35 * (strikes - (center-15))**2)).astype(int)

    call_vol = np.abs(200 + 90000 * np.exp(-0.22 * (strikes - (center+5))**2)).astype(int)
    put_vol  = np.abs(200 + 70000 * np.exp(-0.22 * (strikes - (center-2))**2)).astype(int)

    df = pd.DataFrame({
        "strike": strikes,
        "Net_GEX": net,
        "AG": ag,
        "Call_OI": call_oi,
        "Put_OI": put_oi,
        "Call_Volume": call_vol,
        "Put_Volume": put_vol
    })
    return df, float(center)


st.sidebar.header("Данные")
mode = st.sidebar.radio("Источник данных", ["Загрузить CSV", "Демо-данные"], index=1)

spot_price = None
df = None

if mode == "Загрузить CSV":
    uploaded = st.sidebar.file_uploader("CSV по страйкам", type=["csv"])
    spot_price = st.sidebar.number_input("Цена базового актива (Spot)", value=645.31, step=0.01, format="%.2f")
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
        except Exception as e:
            st.sidebar.error(f"Не удалось прочитать CSV: {e}")
else:
    df, spot_price = make_demo_df(center=645.0)
    spot_price = st.sidebar.number_input("Цена базового актива (Spot)", value=float(spot_price), step=0.01, format="%.2f")

if df is None or df.empty:
    st.info("Загрузите CSV файл с колонками: strike, Net_GEX, AG, Call_OI, Put_OI, Call_Volume, Put_Volume — "
            "или используйте демо-данные в сайдбаре.")
    st.stop()

for col in ["strike", "Net_GEX", "AG", "Call_OI", "Put_OI", "Call_Volume", "Put_Volume"]:
    if col not in df.columns:
        df[col] = 0.0

df["strike"] = pd.to_numeric(df["strike"], errors="coerce")
df = df.dropna(subset=["strike"]).sort_values("strike").reset_index(drop=True)

A0 = st.sidebar.number_input("A0", value=0.2349, step=0.01, format="%.4f")
A1 = st.sidebar.number_input("A1", value=0.1266, step=0.01, format="%.4f")
BETA = st.sidebar.number_input("Beta", value=0.0134, step=0.001, format="%.4f")

df["Power_Zone"] = compute_power_zone(df, spot=float(spot_price), a0=A0, a1=A1, beta=BETA)

st.subheader("Настройки оверлеев")
c1, c2, c3, c4, c5 = st.columns(5)
show_pz = c1.toggle("Power Zone", value=True)
show_call_oi = c2.toggle("Call OI", value=False)
show_put_oi = c3.toggle("Put OI", value=False)
show_call_vol = c4.toggle("Call Volume", value=False)
show_put_vol = c5.toggle("Put Volume", value=False)

fig = build_chart(
    df, spot=float(spot_price),
    show_pz=show_pz, show_call_oi=show_call_oi, show_put_oi=show_put_oi,
    show_call_vol=show_call_vol, show_put_vol=show_put_vol
)
st.plotly_chart(fig, use_container_width=True, theme="streamlit")

st.subheader("Таблица по страйкам")
cols_to_show = ["strike", "Net_GEX", "AG", "Call_OI", "Put_OI", "Call_Volume", "Put_Volume", "Power_Zone"]
st.dataframe(df[cols_to_show], use_container_width=True)

csv_bytes = df[cols_to_show].to_csv(index=False).encode("utf-8")
st.download_button("Скачать CSV", data=csv_bytes, file_name="net_gex_with_power_zone.csv", mime="text/csv")
