"""
============================================================
 US AIRLINE ON-TIME PERFORMANCE — MARCUS REID DASHBOARD v2
============================================================
Redesigned following course modules 3-5 and rubric:
  - Module 3: Pre-attentive attributes, Gestalt, cognitive load
  - Module 4: Chart-question alignment, honest axes
  - Module 5: Grammar of Graphics, color semantics, heatmap geom

Changes from v1:
  - KPIs: large number + semantic color + delta + badge
  - "Why" → Donut (5 categories, clear proportions)
  - "When" → Heatmap hour×day (replaces line chart)
  - "Delay Propagation" → FRI carrier ranking (actionable)
  - "Predictive" → Horizontal bars with benchmark line
  - Full dark theme, colorblind-safe palette
  - Marcus narrative alert bar

Run:  python3 03_dashboard.py
Open: http://127.0.0.1:8050
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, State, ALL
import dash_bootstrap_components as dbc

DATA_DIR = Path("data")

# ── LOAD DATA ─────────────────────────────────────────────
def load_data():
    d = {}
    try:
        d["clean"]       = pd.read_parquet(DATA_DIR / "bts_2023_clean.parquet")
        d["carrier_kpi"] = pd.read_parquet(DATA_DIR / "agg_carrier_kpi.parquet")
        d["heatmap"]     = pd.read_parquet(DATA_DIR / "agg_heatmap.parquet")
        d["monthly"]     = pd.read_parquet(DATA_DIR / "agg_monthly.parquet")
        d["airports"]    = pd.read_parquet(DATA_DIR / "agg_airport_stats.parquet")
        d["causes"]      = pd.read_parquet(DATA_DIR / "agg_delay_causes.parquet")
    except FileNotFoundError as e:
        print(f"⚠ Warning: Data file not found. Using mock data. {e}")
        d = generate_mock_data()
    return d

def generate_mock_data():
    """Generate realistic mock data for visualization."""
    np.random.seed(42)
    n_rows = 5000
    
    carriers = ["Delta Air Lines", "United Airlines", "American Airlines", "Southwest Airlines", "JetBlue Airways"]
    airports_list = ["JFK", "LAX", "ORD", "ATL", "DFW", "LAS", "DEN", "SFO", "SEA", "MIA"]
    
    clean_df = pd.DataFrame({
        "YEAR": np.random.choice([2023], n_rows),
        "MONTH": np.random.choice(range(1, 13), n_rows),
        "DAY_OF_WEEK": np.random.choice(range(0, 7), n_rows),
        "HOUR_DEP": np.random.choice(range(6, 21), n_rows),
        "CARRIER_NAME": np.random.choice(carriers, n_rows),
        "OP_CARRIER": np.random.choice(["DL", "UA", "AA", "WN", "B6"], n_rows),
        "ORIGIN": np.random.choice(airports_list, n_rows),
        "DEST": np.random.choice(airports_list, n_rows),
        "ON_TIME": np.random.choice([0, 1], n_rows, p=[0.31, 0.69]),
        "DEP_DELAY": np.random.normal(5, 20, n_rows),
        "ARR_DELAY": np.random.normal(2.4, 15, n_rows),
        "CANCELLED": np.random.choice([0, 1], n_rows, p=[1, 0]),
        "LATE_AIRCRAFT_DELAY": np.random.exponential(3, n_rows),
        "CARRIER_DELAY": np.random.exponential(2.5, n_rows),
        "NAS_DELAY": np.random.exponential(1.8, n_rows),
        "WEATHER_DELAY": np.random.exponential(1.2, n_rows),
        "SECURITY_DELAY": np.random.exponential(0.3, n_rows),
    })
    
    carrier_kpi = clean_df.groupby("CARRIER_NAME").agg(
        on_time_pct=("ON_TIME", "mean"),
        avg_arr_delay=("ARR_DELAY", "mean"),
        cancel_rate=("CANCELLED", "mean"),
    ).reset_index()
    
    heatmap_df = clean_df[["DAY_OF_WEEK", "HOUR_DEP", "ARR_DELAY"]].copy()
    
    monthly_df = clean_df.groupby("MONTH")["ARR_DELAY"].mean().reset_index()
    
    airport_stats = []
    for apt in airports_list:
        subset = clean_df[(clean_df["ORIGIN"] == apt) | (clean_df["DEST"] == apt)]
        airport_stats.append({
            "iata": apt,
            "city": apt,
            "lat": np.random.uniform(25, 48),
            "lon": np.random.uniform(-125, -70),
            "avg_delay": subset["ARR_DELAY"].mean() if len(subset) > 0 else 2.4,
        })
    airports_df = pd.DataFrame(airport_stats)
    
    causes_df = clean_df[["LATE_AIRCRAFT_DELAY", "CARRIER_DELAY", "NAS_DELAY", "WEATHER_DELAY", "SECURITY_DELAY"]].copy()
    
    return {
        "clean": clean_df,
        "carrier_kpi": carrier_kpi,
        "heatmap": heatmap_df,
        "monthly": monthly_df,
        "airports": airports_df,
        "causes": causes_df,
    }

DATA = load_data()

# ── PALETTE (colorblind-safe, module 3) ───────────────────
C = {
    "bg":       "#0D1117",
    "surface":  "#161B22",
    "surface2": "#1C2230",
    "border":   "rgba(255,255,255,0.08)",
    "text":     "#E6EDF3",
    "muted":    "#7D8590",
    "green":    "#2EA043",
    "amber":    "#D29922",
    "red":      "#DA3633",
    "blue":     "#2F81F7",
    "teal":     "#1B7F79",
}

# ── HELPERS ────────────────────────────────────────────────

def normalize(s):
    """Min-max normalize."""
    return (s - s.min()) / (s.max() - s.min() + 1e-9)

def compute_metrics(df, df_with_cancelled=None):
    """Compute KPI metrics."""
    if df.empty:
        return {"ontime": 0.0, "delay": 0.0, "cancel": 0.0, "fri": 0.0}
    ontime = round(df["ON_TIME"].mean() * 100, 1)
    delay  = round(df["ARR_DELAY"].mean(), 1)

    # Cancel rate desde agg_carrier_kpi que sí tiene el dato real
    try:
        kpi_data = pd.read_parquet(DATA_DIR / "agg_carrier_kpi.parquet")
        cancel = round(
            (kpi_data["cancel_rate"] * kpi_data["flights"]).sum() /
            kpi_data["flights"].sum() * 100, 1
        )
    except:
        cancel = 2.4  # valor real BTS 2023

    fri = round(
        0.50 * (ontime / 100) +
        0.30 * (1 - min(1, delay / 30)) +
        0.20 * (1 - min(1, cancel / 10)),
        3
    ) * 100
    # Avg 150 passengers per flight, delay in minutes → hours lost
    total_delay_hours = round((df["ARR_DELAY"][df["ARR_DELAY"] > 0].sum() * 150) / 60 / 1_000_000, 1)
    return {"ontime": ontime, "delay": delay, "cancel": cancel, "fri": round(fri, 1), "passenger_hours": total_delay_hours}

def kpi_badge(label, color):
    """Create semantic badge."""
    return html.Span(label, style={
        "fontSize": "10px", "fontWeight": "600", "padding": "2px 10px",
        "borderRadius": "20px", "textTransform": "uppercase", "letterSpacing": ".05em",
        "background": f"rgba({','.join(str(int(color.lstrip('#')[i:i+2], 16)) for i in (0, 2, 4))},.14)",
        "color": color, "marginTop": "6px", "display": "inline-block"
    })

def get_panel(title, subtitle, content):
    """Reusable panel component."""
    return html.Div([
        html.Div(title, style={"fontSize": "13px", "fontWeight": "600", "color": C["text"], "marginBottom": "2px"}),
        html.Div(subtitle, style={"fontSize": "11px", "color": C["muted"], "marginBottom": "12px"}),
        html.Div(content)
    ], style={
        "background": C["surface"],
        "border": f"1px solid {C['border']}",
        "borderRadius": "14px",
        "padding": "18px",
    })

# ── KPI STRIP ──────────────────────────────────────────────

def build_kpi_strip(metrics):
    """Build KPI strip with semantic colors."""
    
    def get_color(key, val):
        if key == "ontime":
            return C["green"] if val >= 80 else C["amber"] if val >= 65 else C["red"]
        if key == "delay":
            return C["green"] if val <= 8 else C["amber"] if val <= 15 else C["red"]
        if key == "cancel":
            return C["green"] if val <= 2 else C["amber"] if val <= 5 else C["red"]
        if key == "fri":
            return C["green"] if val >= 70 else C["amber"] if val >= 55 else C["red"]
        return C["text"]

    def get_badge(key, val):
        if key == "ontime":
            return ("Excellent", C["green"]) if val >= 80 else ("Moderate", C["amber"]) if val >= 65 else ("Poor", C["red"])
        if key == "delay":
            return ("Low Impact", C["green"]) if val <= 8 else ("Moderate", C["amber"]) if val <= 15 else ("High Impact", C["red"])
        if key == "cancel":
            return ("Low Risk", C["green"]) if val <= 2 else ("Moderate", C["amber"]) if val <= 5 else ("High Risk", C["red"])
        if key == "fri":
            return ("Reliable", C["green"]) if val >= 70 else ("Moderate", C["amber"]) if val >= 55 else ("Unreliable", C["red"])
        return ("—", C["muted"])

    kpi_defs = [
        ("ontime", "ON-TIME PERFORMANCE", f"{metrics['ontime']}%", "System-wide punctuality", "Arrival < 15 min late"),
        ("delay", "AVG. ARRIVAL DELAY", f"{metrics['delay']} min", "Average late arrival", "Cancelled flights excluded"),
        ("cancel", "CANCELLATION RATE", f"{metrics['cancel']}%", "Flights cancelled", "Separate operational risk"),
        ("fri", "FLIGHT RELIABILITY INDEX", f"{metrics['fri']}", "Composite 0–100 score", "50% OT + 30% delay + 20% cancel"),
    ]

    hero_val = metrics.get("passenger_hours", 0)
    hero_color = C["green"] if hero_val < 0.5 else C["amber"] if hero_val < 1.0 else C["red"]
    hero_card = dbc.Col(html.Div([
        html.Div(style={"height":"4px","borderRadius":"10px 10px 0 0","background":hero_color}),
        html.Div([
            html.P("PASSENGER HOURS LOST", style={"fontSize":"11px","color":C["muted"],
                   "fontWeight":"600","letterSpacing":".06em","textTransform":"uppercase","margin":"0 0 10px"}),
            html.Div(f"{hero_val}M hours", style={"fontSize":"48px","fontWeight":"700",
                     "color":hero_color,"letterSpacing":"-0.04em","lineHeight":"1","marginBottom":"8px"}),
            html.P("Total delay × 150 passengers avg ÷ 60", style={"fontSize":"11px","color":C["muted"],"margin":"0 0 8px"}),
            html.Span("HUMAN IMPACT", style={"fontSize":"10px","fontWeight":"600","padding":"2px 8px",
                      "borderRadius":"20px","textTransform":"uppercase","background":f"rgba(218,54,51,.15)",
                      "color":hero_color}),
        ], style={"padding":"18px 20px"}),
    ], style={"background":C["surface"],"borderRadius":"0 0 12px 12px",
              "border":f"1px solid {C['border']}","borderTop":"none","minHeight":"148px"}),
    md=12, style={"marginBottom":"12px"})

    cols = []
    for key, label, value, subtitle, badge_text in kpi_defs:
        color = get_color(key, metrics[key])
        badge_label, badge_color = get_badge(key, metrics[key])
        cols.append(dbc.Col(html.Div([
            html.Div(style={"height": "4px", "borderRadius": "4px 4px 0 0", "background": color, "marginBottom": "0"}),
            html.Div([
                html.P(label, style={"fontSize": "10px", "color": C["muted"], "fontWeight": "600", "letterSpacing": ".08em", "textTransform": "uppercase", "margin": "0 0 8px"}),
                html.Div(value, style={"fontSize": "28px", "fontWeight": "700", "color": color, "lineHeight": "1", "marginBottom": "4px"}),
                html.P(subtitle, style={"fontSize": "10px", "color": C["muted"], "margin": "0 0 8px"}),
                kpi_badge(badge_label, badge_color),
            ], style={"padding": "14px 16px"}),
        ], style={"background": C["surface"], "borderRadius": "0 0 12px 12px", "border": f"1px solid {C['border']}", "borderTop": "none"}), md=3))

    return html.Div([
        dbc.Row([hero_card], className="g-3"),
        dbc.Row(cols, className="g-3 mb-3"),
    ])

# ── MARCUS ALERT ───────────────────────────────────────────

def marcus_alert(df, selected_airline, selected_airport):
    """Decision insight bar."""
    if df.empty:
        return html.Div("No insights available for this selection.", style={"color": C["muted"], "fontSize": "12px"})

    if selected_airline and selected_airline != "All":
        best_text = f"The current data recommends {selected_airline} for this route."
    else:
        if "OP_CARRIER" in df.columns and "ON_TIME" in df.columns:
            MAJOR_CARRIERS = ["DL", "AA", "UA", "WN", "AS", "B6", "NK", "F9"]
            CARRIER_FRI = (
                df.groupby(["OP_CARRIER", "CARRIER_NAME"])
                .agg(FRI=("ON_TIME", lambda x: x.mean() * 100))
                .reset_index()
                .sort_values("FRI", ascending=False)
            )
            top_fri = CARRIER_FRI[CARRIER_FRI["OP_CARRIER"].isin(MAJOR_CARRIERS)]
            top_carrier = top_fri.iloc[0]["CARRIER_NAME"] if not top_fri.empty else "Delta Air Lines"
            top_score = top_fri.iloc[0]["FRI"] if not top_fri.empty else 80.0
            second_carrier = top_fri.iloc[1]["CARRIER_NAME"] if len(top_fri) > 1 else "American Airlines"
            second_score = top_fri.iloc[1]["FRI"] if len(top_fri) > 1 else 70.0
            gap = top_score - second_score
            best_text = (
                f"Real data insight: Sunday and Friday evenings (19–22h) are the worst slots "
                f"— avg delay up to 19 min with 50k+ flights affected. "
                f"Best option: fly Tuesday or Thursday before 9am — avg delay is negative (arrives early). "
                f"Action: Book {top_carrier} on an early weekday morning for maximum reliability."
            )
        elif "CARRIER_NAME" in df.columns and "ON_TIME" in df.columns:
            carriers = df.groupby("CARRIER_NAME").agg(on_time_pct=("ON_TIME", "mean")).reset_index()
            carriers = carriers.sort_values("on_time_pct", ascending=False)
            if len(carriers) >= 2:
                best_carrier = carriers.iloc[0]["CARRIER_NAME"]
                best_pct = carriers.iloc[0]["on_time_pct"] * 100
                second_pct = carriers.iloc[1]["on_time_pct"] * 100
                best_text = f"Data favors {best_carrier}: {best_pct:.1f}% on-time vs {second_pct:.1f}% for the next-best carrier."
            else:
                best_text = "Carrier reliability is stable across the selected dataset."
        else:
            best_text = "Airline selection is stable."

    slot_text = "Monday 6–9am from ORD"
    if selected_airport and selected_airport != "All":
        slot_text = f"Peak hub analysis at {selected_airport} during Monday 6–9am."

    return html.Div([
        html.Div([
            html.Span("Decision insight", style={"color": C["amber"], "fontSize": "11px", "fontWeight": "700", "marginRight": "10px"}),
            html.Span(best_text, style={"color": C["text"], "fontSize": "12px"}),
        ], style={"display": "flex", "justifyContent": "center", "alignItems": "center", "gap": "10px"}),
        html.Div(slot_text, style={"color": C["muted"], "fontSize": "11px", "marginTop": "8px"}),
    ], style={"background": C["surface"], "border": f"1px solid {C['border']}", "borderRadius": "14px", "padding": "16px", "display": "inline-block"})

# ── MAP ────────────────────────────────────────────────────

def build_airport_map(df, selected_airport):
    """Build airport hotspot map."""
    if df.empty or DATA["airports"].empty:
        fig = go.Figure()
        fig.add_annotation(text="No airport data available", showarrow=False, font=dict(color=C["text"]))
        fig.update_layout(paper_bgcolor=C["bg"], plot_bgcolor=C["bg"], height=380)
        return fig

    airport_data = pd.concat([
        df[["ORIGIN", "ARR_DELAY"]].rename(columns={"ORIGIN": "iata"}) if "ORIGIN" in df.columns else pd.DataFrame(),
        df[["DEST", "ARR_DELAY"]].rename(columns={"DEST": "iata"}) if "DEST" in df.columns else pd.DataFrame(),
    ], ignore_index=True)

    if airport_data.empty:
        fig = go.Figure()
        fig.add_annotation(text="No data", showarrow=False, font=dict(color=C["text"]))
        fig.update_layout(paper_bgcolor=C["bg"], plot_bgcolor=C["bg"], height=380)
        return fig

    stats = airport_data.groupby("iata").agg(avg_delay=("ARR_DELAY", "mean"), flights=("iata", "size")).reset_index()
    stats = stats.merge(DATA["airports"][['iata', 'city', 'lat', 'lon']], on='iata', how='left').dropna(subset=['lat', 'lon'])

    if stats.empty:
        fig = go.Figure()
        fig.add_annotation(text="No geo data", showarrow=False, font=dict(color=C["text"]))
        fig.update_layout(paper_bgcolor=C["bg"], plot_bgcolor=C["bg"], height=380)
        return fig

    np.random.seed(42)
    stats = stats.copy()
    stats["lat_jitter"] = stats["lat"] + np.random.normal(0, 0.015, len(stats))
    stats["lon_jitter"] = stats["lon"] + np.random.normal(0, 0.015, len(stats))

    fig = go.Figure(go.Scattergeo(
        lon=stats["lon_jitter"],
        lat=stats["lat_jitter"],
        text=stats.apply(lambda r: f"<b>{r['iata']}</b><br>Avg delay: {r['avg_delay']:.1f} min<br>Flights: {int(r['flights']):,}", axis=1),
        mode="markers",
        marker=dict(
            size=stats["flights"] / stats["flights"].max() * 20 + 5,
            color=stats["avg_delay"],
            colorscale=[[0.0, "#0d4429"], [0.3, "#1e5f2a"], [0.5, "#8a7a12"], [0.7, "#b04c16"], [1.0, "#a10e1b"]],
            cmin=0,
            cmax=max(stats["avg_delay"].max(), 15),
            colorbar=dict(title=dict(text="Avg delay (min)", font=dict(color=C["muted"], size=10)), thickness=10, len=0.7),
            line=dict(width=1, color="rgba(255,255,255,.3)"),
            opacity=0.85
        ),
        hovertemplate="<b>%{text}</b><extra></extra>",
    ))

    highlight = selected_airport if selected_airport and selected_airport != "All" else "ORD"
    target = stats[stats["iata"] == highlight]
    if not target.empty:
        fig.add_trace(go.Scattergeo(
            lon=target["lon_jitter"],
            lat=target["lat_jitter"],
            mode="markers+text",
            marker=dict(size=24, color=C["amber"], symbol="star", line=dict(width=2, color="white")),
            text=highlight,
            textposition="top center",
            textfont=dict(size=11, color=C["amber"]),
            hoverinfo="skip",
            showlegend=False
        ))

    fig.update_layout(geo=dict(
        scope="usa",
        bgcolor="rgba(0,0,0,0)",
        showframe=False,
        showcoastlines=False,
        showland=True,
        landcolor="#161B22",
        showlakes=True,
        lakecolor="#0D1117",
        showsubunits=True,
        subunitcolor="rgba(255,255,255,0.08)",
        showcountries=False,
        projection_type="albers usa",
    ))
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor=C["bg"], plot_bgcolor=C["bg"], showlegend=False, height=380)
    return fig

# ── HEATMAP ────────────────────────────────────────────────

def build_heatmap(df):
    """Hourly congestion heatmap."""
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No hourly data", showarrow=False, font=dict(color=C["text"]))
        fig.update_layout(paper_bgcolor=C["bg"], plot_bgcolor=C["bg"], height=320)
        return fig

    df = df.copy()
    if "HOUR_DEP" in df.columns:
        df = df[df["HOUR_DEP"].between(6, 20)]
    if "DAY_OF_WEEK" in df.columns and "HOUR_DEP" in df.columns and "ARR_DELAY" in df.columns:
        day_names = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}
        df["day_name"] = df["DAY_OF_WEEK"].map(day_names)
        order = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

        pivot = df.groupby(["day_name", "HOUR_DEP"]).agg(avg_delay=("ARR_DELAY", "mean")).reset_index()
        pivot = pivot.pivot(index="day_name", columns="HOUR_DEP", values="avg_delay").reindex(order)
        pivot = pivot[[h for h in range(6, 21) if h in pivot.columns]]

        x = [f"{int(h):02d}:00" for h in pivot.columns]
        y = pivot.index.tolist()
        z = pivot.values

        fig = go.Figure(go.Heatmap(
            z=z, x=x, y=y,
            text=np.round(z, 0),
            texttemplate="%{text}",
            textfont=dict(size=9, color="rgba(230,237,243,.8)"),
            colorscale=[[0.0, "#0d4429"], [0.25, "#1e5f2a"], [0.5, "#8a7a12"], [0.75, "#b04c16"], [1.0, "#a10e1b"]],
            colorbar=dict(title=dict(text="Avg delay (min)", font=dict(color=C["muted"], size=9)), thickness=8, len=0.7),
            hovertemplate="<b>%{y} %{x}</b><br>Avg delay: %{z:.1f} min<extra></extra>",
        ))

        # Calcular worst y best slot dinámicamente desde los datos filtrados
        heatmap_agg = (
            df.groupby(["day_name", "HOUR_DEP"])
            .agg(avg_delay=("ARR_DELAY", "mean"), flight_count=("ARR_DELAY", "count"))
            .reset_index()
        )
        df_vol = heatmap_agg[heatmap_agg["flight_count"] > 100]
        if not df_vol.empty:
            worst_row = df_vol.loc[df_vol["avg_delay"].idxmax()]
            best_row  = df_vol.loc[df_vol["avg_delay"].idxmin()]

            pivot_cols = list(pivot.columns)
            pivot_idx  = list(pivot.index)

            worst_x = pivot_cols.index(worst_row["HOUR_DEP"]) if worst_row["HOUR_DEP"] in pivot_cols else None
            worst_y = pivot_idx.index(worst_row["day_name"]) if worst_row["day_name"] in pivot_idx else None
            best_x  = pivot_cols.index(best_row["HOUR_DEP"]) if best_row["HOUR_DEP"] in pivot_cols else None
            best_y  = pivot_idx.index(best_row["day_name"]) if best_row["day_name"] in pivot_idx else None

            if worst_x is not None and worst_y is not None:
                fig.add_shape(type="rect",
                    x0=worst_x-0.5, x1=worst_x+0.5,
                    y0=worst_y-0.5, y1=worst_y+0.5,
                    xref="x", yref="y",
                    line=dict(color="#FF4444", width=2.5),
                    fillcolor="rgba(0,0,0,0)"
                )
                fig.add_annotation(
                    x=1.02, y=worst_y,
                    xref="paper", yref="y",
                    text="▲ worst",
                    showarrow=False,
                    font=dict(size=9, color="#FF4444"),
                    xanchor="left"
                )

            if best_x is not None and best_y is not None:
                fig.add_shape(type="rect",
                    x0=best_x-0.5, x1=best_x+0.5,
                    y0=best_y-0.5, y1=best_y+0.5,
                    xref="x", yref="y",
                    line=dict(color="#2EA043", width=2.5),
                    fillcolor="rgba(0,0,0,0)"
                )
                fig.add_annotation(
                    x=1.02, y=best_y,
                    xref="paper", yref="y",
                    text="▼ best",
                    showarrow=False,
                    font=dict(size=9, color="#2EA043"),
                    xanchor="left"
                )

        fig.update_layout(margin=dict(l=50, r=70, t=10, b=50), paper_bgcolor=C["bg"], plot_bgcolor=C["bg"], font=dict(color=C["text"], size=9), height=320)
        return fig
    else:
        fig = go.Figure()
        fig.add_annotation(text="Missing heatmap columns", showarrow=False, font=dict(color=C["text"]))
        fig.update_layout(paper_bgcolor=C["bg"], plot_bgcolor=C["bg"], height=320)
        return fig

# ── DONUT CHART ────────────────────────────────────────────

def build_donut(df):
    """Why delays occur - cause breakdown."""
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No cause data", showarrow=False, font=dict(color=C["text"]))
        fig.update_layout(paper_bgcolor=C["bg"], plot_bgcolor=C["bg"], height=380)
        return fig

    totals = {}
    if "LATE_AIRCRAFT_DELAY" in df.columns:
        totals["Late Aircraft"] = df["LATE_AIRCRAFT_DELAY"].sum()
    if "CARRIER_DELAY" in df.columns:
        totals["Airline Operations"] = df["CARRIER_DELAY"].sum()
    if "NAS_DELAY" in df.columns:
        totals["NAS / Air Traffic"] = df["NAS_DELAY"].sum()
    if "WEATHER_DELAY" in df.columns:
        totals["Weather"] = df["WEATHER_DELAY"].sum()
    if "SECURITY_DELAY" in df.columns:
        totals["Security"] = df["SECURITY_DELAY"].sum()

    if not totals:
        fig = go.Figure()
        fig.add_annotation(text="No cause columns", showarrow=False, font=dict(color=C["text"]))
        fig.update_layout(paper_bgcolor=C["bg"], plot_bgcolor=C["bg"], height=380)
        return fig

    labels = list(totals.keys())
    values = list(totals.values())
    colors_list = [C["red"], C["amber"], C["blue"], C["green"], C["teal"]]

    max_val = max(values) if values else 1
    pull = [0.1 if v == max_val else 0 for v in values]

    fig = go.Figure(go.Pie(
        labels=labels,
        values=values,
        hole=0.55,
        marker=dict(colors=colors_list[:len(labels)], line=dict(color=C["bg"], width=2)),
        textinfo="none",
        hovertemplate="<b>%{label}</b><br>%{percent}<br>%{value:,.0f} min<extra></extra>",
        pull=pull,
    ))

    main_pct = round(values[0] / sum(values) * 100 if sum(values) else 0)
    fig.add_annotation(
        text=f"<b>{main_pct}%</b><br><span style='font-size:9px'>{labels[0].upper()}</span>",
        x=0.5, y=0.5, showarrow=False,
        font=dict(color=colors_list[0], size=16, family="system-ui"),
        xanchor="center", yanchor="middle",
    )

    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor=C["bg"], plot_bgcolor=C["bg"], showlegend=True, height=380)
    return fig

# ── FRI CHART ──────────────────────────────────────────────

def build_fri_chart(df, selected_airline):
    """Carrier reliability ranking."""
    if DATA["carrier_kpi"].empty:
        fig = go.Figure()
        fig.add_annotation(text="No carrier data", showarrow=False, font=dict(color=C["text"]))
        fig.update_layout(paper_bgcolor=C["bg"], plot_bgcolor=C["bg"], height=320)
        return fig

    carriers = DATA["carrier_kpi"].copy()

    if "on_time_pct" not in carriers.columns or "avg_arr_delay" not in carriers.columns:
        fig = go.Figure()
        fig.add_annotation(text="Missing carrier columns", showarrow=False, font=dict(color=C["text"]))
        fig.update_layout(paper_bgcolor=C["bg"], plot_bgcolor=C["bg"], height=320)
        return fig

    # Compute FRI
    carriers["FRI"] = (
        0.50 * carriers["on_time_pct"] +
        0.30 * (1 - normalize(carriers["avg_arr_delay"])) * 100 +
        0.20 * (1 - carriers["cancel_rate"].fillna(0)) * 100
    ).round(1)
    carriers = carriers.sort_values("FRI", ascending=False)

    def fri_color(v):
        return C["green"] if v >= 70 else C["amber"] if v >= 55 else C["red"]

    colors = [fri_color(v) for v in carriers["FRI"]]
    if selected_airline and selected_airline != "All":
        colors = [C["blue"] if name == selected_airline else c for c, name in zip(colors, carriers["CARRIER_NAME"])]

    fig = go.Figure(go.Bar(
        x=carriers["FRI"],
        y=carriers["CARRIER_NAME"],
        orientation="h",
        marker=dict(color=colors, line=dict(color=colors, width=1.5)),
        text=carriers["FRI"].round(0),
        textposition="outside",
        textfont=dict(color=C["muted"], size=9),
        hovertemplate="<b>%{y}</b><br>FRI: %{x:.0f}/100<extra></extra>",
    ))

    industry_avg = carriers["FRI"].mean()
    fig.add_vline(x=70, line=dict(color=C["green"], width=1, dash="dot"))
    fig.add_vline(x=55, line=dict(color=C["red"], width=1, dash="dot"))
    fig.add_vline(x=industry_avg, line=dict(color=C["blue"], width=2, dash="dash"))

    fig.update_layout(
        xaxis=dict(title=dict(text="Flight Reliability Index (0–100)", font=dict(color=C["muted"], size=9)), tickfont=dict(color=C["muted"], size=8)),
        yaxis=dict(tickfont=dict(color=C["text"], size=9)),
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor=C["bg"],
        plot_bgcolor=C["bg"],
        font=dict(color=C["text"], family="system-ui", size=9),
        showlegend=False,
        height=320
    )
    return fig

# ── PREDICTIVE RISK ────────────────────────────────────────

def build_predictive_chart(df):
    """Top airports with delay risk."""
    if DATA["airports"].empty:
        fig = go.Figure()
        fig.add_annotation(text="No airport data", showarrow=False, font=dict(color=C["text"]))
        fig.update_layout(paper_bgcolor=C["bg"], plot_bgcolor=C["bg"], height=380)
        return fig

    stats = DATA["airports"].copy()
    if "avg_delay" not in stats.columns:
        fig = go.Figure()
        fig.add_annotation(text="Missing delay column", showarrow=False, font=dict(color=C["text"]))
        fig.update_layout(paper_bgcolor=C["bg"], plot_bgcolor=C["bg"], height=380)
        return fig

    stats = stats.sort_values("avg_delay", ascending=False).head(12)

    def risk_color(v):
        return C["red"] if v > 12 else C["amber"] if v >= 8 else C["green"]

    colors = [risk_color(v) for v in stats["avg_delay"]]

    fig = go.Figure(go.Bar(
        x=stats["avg_delay"],
        y=stats["iata"],
        orientation="h",
        marker=dict(color=colors, line=dict(color=colors, width=1)),
        text=[f"{v:.1f}" for v in stats["avg_delay"]],
        textposition="outside",
        textfont=dict(color=C["text"], size=9),
        hovertemplate="<b>%{y}</b><br>Avg delay: %{x:.1f} min<extra></extra>",
    ))

    sys_avg = stats["avg_delay"].mean()
    fig.add_vline(x=sys_avg, line=dict(color="rgba(255,255,255,.5)", width=2, dash="dash"))

    fig.update_layout(
        xaxis=dict(title=dict(text="Minutes", font=dict(color=C["muted"], size=8)), tickfont=dict(color=C["muted"], size=8)),
        yaxis=dict(tickfont=dict(color=C["text"], size=9)),
        margin=dict(l=0, r=60, t=0, b=0),
        paper_bgcolor=C["bg"],
        plot_bgcolor=C["bg"],
        font=dict(color=C["text"], family="system-ui", size=8),
        showlegend=False,
        height=380
    )
    return fig

# ── SCATTER PROPAGATION ────────────────────────────────────

def build_scatter_propagation(df):
    """DEP_DELAY vs ARR_DELAY scatter with regression line, colored by airline."""
    if df.empty or len(df) < 500:
        fig = go.Figure()
        fig.add_annotation(
            text="Not enough flights for propagation analysis",
            showarrow=False,
            font=dict(color=C["muted"], size=13)
        )
        fig.update_layout(
            paper_bgcolor=C["bg"], plot_bgcolor=C["bg"],
            height=420, margin=dict(l=60, r=20, t=10, b=50)
        )
        return fig
    if df.empty or "DEP_DELAY" not in df.columns or "ARR_DELAY" not in df.columns:
        fig = go.Figure()
        fig.add_annotation(text="No delay propagation data available", showarrow=False, font=dict(color=C["text"]))
        fig.update_layout(paper_bgcolor=C["bg"], plot_bgcolor=C["bg"], height=420)
        return fig

    sample = df[["DEP_DELAY", "ARR_DELAY", "CARRIER_NAME"]].dropna()
    sample = sample[(sample["DEP_DELAY"] >= -30) & (sample["DEP_DELAY"] <= 180)]
    sample = sample[(sample["ARR_DELAY"] >= -60) & (sample["ARR_DELAY"] <= 180)]
    if len(sample) > 50000:
        sample = sample.sample(50000, random_state=42)

    palette = [C["blue"], C["green"], C["amber"], C["red"], C["teal"],
               "#9B59B6", "#E67E22", "#1ABC9C", "#F39C12", "#2ECC71"]
    carriers_sorted = sorted(sample["CARRIER_NAME"].unique())
    color_map = {c: palette[i % len(palette)] for i, c in enumerate(carriers_sorted)}

    fig = go.Figure()
    for carrier, grp in sample.groupby("CARRIER_NAME"):
        fig.add_trace(go.Scattergl(
            x=grp["DEP_DELAY"],
            y=grp["ARR_DELAY"],
            mode="markers",
            name=carrier,
            marker=dict(color=color_map[carrier], size=3, opacity=0.35),
            hovertemplate=f"<b>{carrier}</b><br>Dep delay: %{{x:.0f}} min<br>Arr delay: %{{y:.0f}} min<extra></extra>",
        ))

    x_vals = sample["DEP_DELAY"].values
    y_vals = sample["ARR_DELAY"].values
    mask = np.isfinite(x_vals) & np.isfinite(y_vals)
    if mask.sum() > 2:
        m, b = np.polyfit(x_vals[mask], y_vals[mask], 1)
        x_range = np.array([np.percentile(x_vals[mask], 1), np.percentile(x_vals[mask], 99)])
        fig.add_trace(go.Scatter(
            x=x_range, y=m * x_range + b,
            mode="lines",
            name="Regression",
            line=dict(color="#FFFFFF", width=3, dash="dash"),
            opacity=0.9,
            hoverinfo="skip",
        ))

    fig.update_layout(
        xaxis=dict(
            range=[-30, 185],
            title=dict(text="Departure delay (min)", font=dict(color=C["muted"], size=9)),
            tickfont=dict(color=C["muted"], size=8),
            gridcolor=C["border"], zeroline=False,
        ),
        yaxis=dict(
            range=[-60, 185],
            title=dict(text="Arrival delay (min)", font=dict(color=C["muted"], size=9)),
            tickfont=dict(color=C["muted"], size=8),
            gridcolor=C["border"], zeroline=False,
        ),
        legend=dict(font=dict(color=C["muted"], size=8), bgcolor="rgba(0,0,0,0)", itemsizing="constant"),
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor=C["bg"],
        plot_bgcolor=C["bg"],
        font=dict(color=C["text"], family="system-ui", size=9),
        height=420,
    )
    return fig

# ── SEVERE DELAY CHART ─────────────────────────────────────

def build_severe_delay_chart(df):
    """Horizontal bar: % of flights with ARR_DELAY > 60 min by airline."""
    if df.empty or len(df) < 100:
        fig = go.Figure()
        fig.add_annotation(
            text="Not enough data for this selection",
            showarrow=False,
            font=dict(color=C["muted"], size=13),
            xanchor="center", yanchor="middle"
        )
        fig.update_layout(
            paper_bgcolor=C["bg"], plot_bgcolor=C["bg"],
            height=340, margin=dict(l=50, r=60, t=10, b=30)
        )
        return fig
    if "ARR_DELAY" not in df.columns or "CARRIER_NAME" not in df.columns:
        fig = go.Figure()
        fig.add_annotation(text="No severe delay data available", showarrow=False, font=dict(color=C["text"]))
        fig.update_layout(paper_bgcolor=C["bg"], plot_bgcolor=C["bg"], height=420)
        return fig

    group_cols = ["OP_CARRIER", "CARRIER_NAME"] if "OP_CARRIER" in df.columns else ["CARRIER_NAME"]
    severe = (
        df.groupby(group_cols)
        .apply(lambda x: (x["ARR_DELAY"] > 60).mean() * 100)
        .reset_index(name="severe_pct")
        .sort_values("severe_pct", ascending=True)
    )
    if len(severe) < 3:
        # Mostrar el carrier seleccionado vs sistema
        try:
            all_severe = (
                DATA["clean"]
                .groupby("CARRIER_NAME")
                .apply(lambda x: (x["ARR_DELAY"] > 60).mean() * 100)
                .reset_index(name="severe_rate")
            )
            system_avg = all_severe["severe_rate"].mean()
        except:
            system_avg = 7.0  # valor real BTS 2023

        carrier_name = severe.iloc[0]["CARRIER_NAME"] if not severe.empty else "Selected carrier"
        carrier_rate = severe.iloc[0]["severe_pct"] if not severe.empty else 0

        def severe_color(v):
            return C["red"] if v >= 6 else C["amber"] if v >= 3 else C["green"]

        col = severe_color(carrier_rate)
        r, g, b = int(col[1:3], 16), int(col[3:5], 16), int(col[5:7], 16)
        col_fill = f"rgba({r},{g},{b},0.45)"

        fig = go.Figure()

        # Barra del carrier seleccionado
        fig.add_trace(go.Bar(
            x=[carrier_rate],
            y=[carrier_name],
            orientation="h",
            marker=dict(color=col_fill, line=dict(color=col, width=1.5)),
            text=[f"{carrier_rate:.1f}%"],
            textposition="outside",
            textfont=dict(color=C["muted"], size=11),
            name=carrier_name,
        ))

        # Línea de sistema
        fig.add_vline(
            x=system_avg,
            line=dict(color="rgba(255,255,255,.4)", width=1.5, dash="dash"),
            annotation=dict(
                text=f"System avg {system_avg:.1f}%",
                font=dict(color=C["muted"], size=9),
                yref="paper", y=1.1, showarrow=False
            )
        )

        # Comparación textual
        diff = carrier_rate - system_avg
        comp_text = f"{abs(diff):.1f}pp {'above' if diff > 0 else 'below'} system average"
        comp_color = C["red"] if diff > 0 else C["green"]

        fig.add_annotation(
            x=0.5, y=-0.25,
            xref="paper", yref="paper",
            text=comp_text,
            showarrow=False,
            font=dict(color=comp_color, size=11)
        )

        fig.update_xaxes(
            range=[0, max(carrier_rate, system_avg) * 1.4],
            showgrid=False,
            title=dict(text="Flights with ARR_DELAY > 60 min (%)",
                       font=dict(color=C["muted"], size=10)),
            tickfont=dict(color=C["muted"], size=9)
        )
        fig.update_yaxes(showgrid=False, tickfont=dict(color=C["text"], size=11))
        fig.update_layout(
            paper_bgcolor=C["bg"], plot_bgcolor=C["bg"],
            margin=dict(l=130, r=60, t=30, b=50),
            height=200,
            showlegend=False,
            font=dict(color=C["text"], family="system-ui")
        )
        return fig

    def sev_color(v):
        return C["green"] if v < 3 else C["amber"] if v <= 6 else C["red"]

    colors = [sev_color(v) for v in severe["severe_pct"]]

    fig = go.Figure(go.Bar(
        x=severe["severe_pct"],
        y=severe["CARRIER_NAME"],
        orientation="h",
        marker=dict(color=colors, line=dict(color=colors, width=1)),
        text=[f"{v:.1f}%" for v in severe["severe_pct"]],
        textposition="outside",
        textfont=dict(color=C["muted"], size=9),
        hovertemplate="<b>%{y}</b><br>Severe delay rate: %{x:.1f}%<extra></extra>",
    ))

    fig.add_vline(x=3, line=dict(color=C["green"], width=1, dash="dot"))
    fig.add_vline(x=6, line=dict(color=C["red"], width=1, dash="dot"))
    system_avg = severe["severe_pct"].mean()
    fig.add_vline(
        x=system_avg,
        line=dict(color="rgba(255,255,255,.4)", width=1.5, dash="dash"),
        annotation=dict(
            text=f"System avg {system_avg:.1f}%",
            font=dict(color=C["muted"], size=9),
            yref="paper", y=1.02, showarrow=False,
        ),
    )

    fig.update_layout(
        xaxis=dict(
            title=dict(text="Flights with ARR_DELAY > 60 min (%)", font=dict(color=C["muted"], size=9)),
            tickfont=dict(color=C["muted"], size=8),
        ),
        yaxis=dict(tickfont=dict(color=C["text"], size=9)),
        margin=dict(l=0, r=60, t=0, b=0),
        paper_bgcolor=C["bg"],
        plot_bgcolor=C["bg"],
        font=dict(color=C["text"], family="system-ui", size=9),
        showlegend=False,
        height=420,
    )
    return fig

# ── SEASONALITY CHART ──────────────────────────────────────

def build_seasonality_chart(df):
    """Monthly on-time % line chart for top 5 carriers (DL, AA, UA, WN, AS)."""
    if df.empty or len(df) < 1000:
        fig = go.Figure()
        fig.add_annotation(
            text="Select 'All carriers' to see seasonal patterns",
            showarrow=False,
            font=dict(color=C["muted"], size=13)
        )
        fig.update_layout(
            paper_bgcolor=C["bg"], plot_bgcolor=C["bg"],
            height=360, margin=dict(l=60, r=20, t=10, b=50)
        )
        return fig
    TOP5_CODES = {"DL", "AA", "UA", "WN", "AS"}
    TOP5_NAMES = {"Delta Air Lines", "American Airlines", "United Airlines",
                  "Southwest Airlines", "Alaska Airlines"}
    MONTH_LABELS = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                    7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}

    if df.empty or "MONTH" not in df.columns or "ON_TIME" not in df.columns:
        fig = go.Figure()
        fig.add_annotation(text="No seasonality data available", showarrow=False, font=dict(color=C["text"]))
        fig.update_layout(paper_bgcolor=C["bg"], plot_bgcolor=C["bg"], height=360)
        return fig

    if "OP_CARRIER" in df.columns and df["OP_CARRIER"].isin(TOP5_CODES).any():
        sub = df[df["OP_CARRIER"].isin(TOP5_CODES)].copy()
        group_col = "OP_CARRIER"
    elif df["CARRIER_NAME"].isin(TOP5_NAMES).any():
        sub = df[df["CARRIER_NAME"].isin(TOP5_NAMES)].copy()
        group_col = "CARRIER_NAME"
    else:
        top5 = df["CARRIER_NAME"].value_counts().head(5).index.tolist()
        sub = df[df["CARRIER_NAME"].isin(top5)].copy()
        group_col = "CARRIER_NAME"

    monthly = (
        sub.groupby([group_col, "MONTH"])["ON_TIME"]
        .mean()
        .reset_index(name="on_time_pct")
    )
    monthly["on_time_pct"] *= 100

    CARRIER_COLORS = {
        "DL": "#2EA043",  # verde  — Delta
        "AA": "#2F81F7",  # azul   — American
        "UA": "#D29922",  # ámbar  — United
        "WN": "#DA3633",  # rojo   — Southwest
        "AS": "#1B7F79",  # teal   — Alaska
    }
    NAME_TO_CODE = {
        "Delta Air Lines": "DL", "American Airlines": "AA",
        "United Airlines": "UA", "Southwest Airlines": "WN", "Alaska Airlines": "AS",
    }
    FALLBACK_COLORS = [C["blue"], C["green"], C["amber"], C["red"], C["teal"]]
    carriers_list = sorted(monthly[group_col].unique())

    fig = go.Figure()
    for i, carrier in enumerate(carriers_list):
        code = carrier if group_col == "OP_CARRIER" else NAME_TO_CODE.get(carrier)
        color = CARRIER_COLORS.get(code, FALLBACK_COLORS[i % len(FALLBACK_COLORS)])
        grp = monthly[monthly[group_col] == carrier].sort_values("MONTH")
        fig.add_trace(go.Scatter(
            x=grp["MONTH"],
            y=grp["on_time_pct"],
            mode="lines+markers",
            name=carrier,
            line=dict(color=color, width=2),
            marker=dict(color=color, size=5),
            hovertemplate=f"<b>{carrier}</b><br>%{{x}}: %{{y:.1f}}%<extra></extra>",
        ))

    fig.update_layout(
        xaxis=dict(
            tickvals=list(MONTH_LABELS.keys()),
            ticktext=list(MONTH_LABELS.values()),
            tickfont=dict(color=C["muted"], size=8),
            gridcolor=C["border"],
        ),
        yaxis=dict(
            title=dict(text="On-time % (arr < 15 min late)", font=dict(color=C["muted"], size=9)),
            tickfont=dict(color=C["muted"], size=8),
            ticksuffix="%",
            gridcolor=C["border"],
        ),
        legend=dict(font=dict(color=C["muted"], size=9), bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor=C["bg"],
        plot_bgcolor=C["bg"],
        font=dict(color=C["text"], family="system-ui", size=9),
        height=360,
    )
    return fig

# ── APP SETUP ──────────────────────────────────────────────

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY], suppress_callback_exceptions=True)

app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            body { background: #0D1117; }
            .Select-control { background-color: #1C2230 !important; border: 1px solid rgba(255,255,255,0.12) !important; border-radius: 6px !important; }
            .Select-value-label { color: #E6EDF3 !important; }
            .Select-placeholder { color: #7D8590 !important; }
            .Select-input > input { color: #E6EDF3 !important; }
            .Select-menu-outer { background-color: #1C2230 !important; border: 1px solid rgba(255,255,255,0.12) !important; z-index: 9999 !important; }
            .VirtualizedSelectOption { background-color: #1C2230 !important; color: #E6EDF3 !important; }
            .VirtualizedSelectFocusedOption { background-color: #2F81F7 !important; color: #FFFFFF !important; }
            .Select-arrow { border-color: #7D8590 transparent transparent !important; }
            .is-open .Select-arrow { border-color: transparent transparent #7D8590 !important; }
            .Select.is-focused > .Select-control { border-color: #2F81F7 !important; box-shadow: none !important; }
            .dash-dropdown { min-width: 140px; }
            .dark-dropdown .Select-control { background-color: #1C2230 !important; }
            .dark-dropdown .Select-value-label { color: #E6EDF3 !important; }
            .dark-dropdown .Select-placeholder { color: #7D8590 !important; }
            .dark-dropdown .Select-menu-outer { background-color: #1C2230 !important; color: #E6EDF3 !important; }
            .dark-dropdown option { background-color: #1C2230 !important; color: #E6EDF3 !important; }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
        <script>
            document.addEventListener('keydown', function(e) {
                if (e.key === 'ArrowRight') {
                    var btn = document.getElementById('btn-next');
                    if (btn) btn.click();
                } else if (e.key === 'ArrowLeft') {
                    var btn = document.getElementById('btn-prev');
                    if (btn) btn.click();
                }
            });
        </script>
    </body>
</html>
'''

year_options = [{"label": str(y), "value": str(y)} for y in sorted(DATA["clean"]["YEAR"].unique())] if not DATA["clean"].empty else [{"label": "2023", "value": "2023"}]
carrier_options = [{"label": "All", "value": "All"}] + [{"label": c, "value": c} for c in sorted(DATA["clean"]["CARRIER_NAME"].unique())] if not DATA["clean"].empty else [{"label": "All", "value": "All"}]
airport_options = [{"label": "All", "value": "All"}] + [{"label": a, "value": a} for a in sorted(DATA["clean"]["ORIGIN"].unique())] if not DATA["clean"].empty else [{"label": "All", "value": "All"}]

dd_style = {"background": C["surface"], "color": C["text"], "border": f"1px solid {C['border']}", "minHeight": "36px", "fontSize": "13px"}

# ── SLIDE TITLES ───────────────────────────────────────────

SLIDE_TITLES = {
    0: "Cover",
    1: "01 — The System",
    2: "02 — Where it breaks",
    3: "03 — Why it breaks",
    4: "04 — When it breaks",
    5: "05 — Who to trust",
    6: "06 — The domino effect",
    7: "07 — The full picture",
    8: "08 — Marcus decides",
}
TOTAL_SLIDES = 8

MENU_ITEMS = [
    (1, "The System",        "KPIs and the scale of the problem"),
    (2, "Where it breaks",   "Airport congestion map"),
    (3, "Why it breaks",     "Delay cause breakdown"),
    (4, "When it breaks",    "Hour × weekday heatmap"),
    (5, "Who to trust",      "Flight Reliability Index ranking"),
    (6, "The domino effect", "Propagation and severe delays"),
    (7, "The full picture",  "Predictive risk and seasonality"),
    (8, "Marcus decides",    "Recommended actions"),
]

_DOT_ACTIVE   = {"width": "8px", "height": "8px", "borderRadius": "50%", "background": "#2F81F7",
                 "border": "none", "cursor": "pointer", "padding": "0", "transition": "all 0.2s", "flexShrink": "0"}
_DOT_INACTIVE = {"width": "6px", "height": "6px", "borderRadius": "50%", "background": "rgba(255,255,255,0.15)",
                 "border": "none", "cursor": "pointer", "padding": "0", "transition": "all 0.2s", "flexShrink": "0"}

# ── LAYOUT ─────────────────────────────────────────────────

app.layout = html.Div(style={"background": C["bg"], "minHeight": "100vh", "fontFamily": "system-ui,-apple-system,sans-serif"}, children=[
    # Topbar
    html.Div(style={
        "background": C["surface"], "borderBottom": f"1px solid {C['border']}",
        "padding": "0 28px", "height": "64px",
        "display": "flex", "alignItems": "center", "justifyContent": "space-between",
    }, children=[
        html.Div([
            html.Button("☰  Menu", id="menu-toggle", n_clicks=0, style={
                "background": "transparent", "color": "#7D8590",
                "border": "1px solid rgba(255,255,255,0.12)", "borderRadius": "8px",
                "padding": "7px 14px", "fontSize": "12px", "fontWeight": "600",
                "cursor": "pointer", "marginRight": "16px", "flexShrink": "0",
                "fontFamily": "system-ui,-apple-system,sans-serif",
            }),
            html.Span("✈ ", style={"fontSize": "20px", "color": "#2F81F7", "marginRight": "10px"}),
            html.Div([
                html.Div("Delay Risk & Operations Dashboard",
                         style={"fontSize": "16px", "fontWeight": "700", "color": "#E6EDF3",
                                "letterSpacing": "-0.3px", "lineHeight": "1.2"}),
                html.Div("Marcus Reid's data journey — 6.7M flights · BTS 2023",
                         style={"fontSize": "11px", "color": "#7D8590", "marginTop": "1px"}),
            ]),
        ], style={"display": "flex", "alignItems": "center"}),
        html.Div([
            html.Div([
                html.Div("Year", style={"fontSize": "11px", "color": "#58A6FF", "marginBottom": "4px", "fontWeight": "700", "letterSpacing": ".06em", "textTransform": "uppercase"}),
                dcc.Dropdown(id="year-filter", options=year_options, value=str(max(DATA["clean"]["YEAR"])) if not DATA["clean"].empty else "2023", clearable=False, style={"backgroundColor": "#0D1117", "color": "#58A6FF", "border": "2px solid #388BFD", "borderRadius": "8px", "fontSize": "13px", "minWidth": "130px", "fontWeight": "600"}, className="dark-dropdown")
            ]),
            html.Div([
                html.Div("Airline", style={"fontSize": "11px", "color": "#58A6FF", "marginBottom": "4px", "fontWeight": "700", "letterSpacing": ".06em", "textTransform": "uppercase"}),
                dcc.Dropdown(id="airline-filter", options=carrier_options, value="All", clearable=False, style={"backgroundColor": "#0D1117", "color": "#58A6FF", "border": "2px solid #388BFD", "borderRadius": "8px", "fontSize": "13px", "minWidth": "190px", "fontWeight": "600"}, className="dark-dropdown")
            ]),
            html.Div([
                html.Div("Hub Airport", style={"fontSize": "11px", "color": "#58A6FF", "marginBottom": "4px", "fontWeight": "700", "letterSpacing": ".06em", "textTransform": "uppercase"}),
                dcc.Dropdown(id="airport-filter", options=airport_options, value="All", clearable=False, style={"backgroundColor": "#0D1117", "color": "#58A6FF", "border": "2px solid #388BFD", "borderRadius": "8px", "fontSize": "13px", "minWidth": "190px", "fontWeight": "600"}, className="dark-dropdown")
            ]),
        ], style={"display": "flex", "alignItems": "center", "gap": "16px"}),
    ]),

    # Slide store
    dcc.Store(id="current-slide", data=0),

    # Collapsible side panel
    html.Div([
        html.Div("Chapters", style={
            "fontSize": "11px", "color": "#7D8590", "letterSpacing": "2px",
            "textTransform": "uppercase", "marginBottom": "16px",
            "paddingBottom": "8px", "borderBottom": "1px solid rgba(255,255,255,0.08)",
        }),
        *[html.Div([
            html.Span(f"{i+1:02d}", style={
                "fontSize": "10px", "color": "#2F81F7",
                "fontFamily": "monospace", "fontWeight": "700", "marginRight": "10px",
            }),
            html.Span(title, style={"fontSize": "13px", "color": "#E6EDF3"}),
        ], id=f"panel-slide-{i+1}", n_clicks=0, style={
            "padding": "10px 12px", "borderRadius": "6px", "cursor": "pointer",
            "marginBottom": "4px", "display": "flex", "alignItems": "center",
            "background": "transparent",
        }) for i, title in enumerate([
            "The System", "Where it breaks", "Why it breaks",
            "When it breaks", "Who to trust", "The domino effect",
            "The full picture", "Marcus decides",
        ])],
    ], id="side-panel", style={
        "position": "fixed", "top": "70px", "left": "0", "width": "220px",
        "height": "calc(100vh - 70px)", "background": "rgba(13,17,23,0.97)",
        "borderRight": "1px solid rgba(255,255,255,0.08)", "padding": "20px 16px",
        "zIndex": "1999", "transform": "translateX(-100%)", "transition": "transform 0.25s ease",
        "backdropFilter": "blur(10px)", "overflowY": "auto",
    }),

    # Slide content area
    html.Div(id="slide-content", style={
        "padding": "16px 28px 100px 28px",
        "overflowY": "auto",
        "minHeight": "calc(100vh - 64px - 60px)",
        "boxSizing": "border-box",
    }),

    # Progress dots (static — styles updated per slide by callback)
    html.Div([
        html.Button(style=_DOT_ACTIVE if i == 0 else _DOT_INACTIVE, id=f"dot-{i}", n_clicks=0)
        for i in range(9)
    ], style={
        "position": "fixed", "bottom": "62px", "left": "0", "right": "0",
        "display": "flex", "justifyContent": "center", "alignItems": "center",
        "paddingBottom": "4px",
        "gap": "8px", "zIndex": "999",
    }),

    # Fixed bottom navigation bar
    html.Div([
        html.Div(id="slide-counter", style={"fontSize": "11px", "color": "#7D8590", "minWidth": "60px"}),
        html.Div([
            html.Button("← Prev", id="btn-prev", n_clicks=0, style={
                "background": "transparent", "color": "#7D8590",
                "border": "1px solid rgba(255,255,255,0.12)",
                "borderRadius": "6px", "padding": "8px 20px",
                "fontSize": "12px", "cursor": "pointer",
                "fontFamily": "system-ui,-apple-system,sans-serif",
            }),
            html.Div(id="slide-indicator", style={
                "fontSize": "13px", "color": "#E6EDF3", "fontWeight": "500",
                "minWidth": "220px", "textAlign": "center",
            }),
            html.Button("Next →", id="btn-next", n_clicks=0, style={
                "background": "#2F81F7", "color": "white", "border": "none",
                "borderRadius": "6px", "padding": "8px 20px",
                "fontSize": "12px", "cursor": "pointer", "fontWeight": "600",
                "fontFamily": "system-ui,-apple-system,sans-serif",
            }),
        ], style={"display": "flex", "alignItems": "center", "gap": "16px"}),
        html.Button("⌂ Menu", id="btn-home", n_clicks=0, style={
            "background": "transparent", "color": "#7D8590",
            "border": "1px solid rgba(255,255,255,0.12)",
            "borderRadius": "6px", "padding": "8px 16px",
            "fontSize": "12px", "cursor": "pointer", "minWidth": "70px",
            "fontFamily": "system-ui,-apple-system,sans-serif",
        }),
    ], style={
        "position": "fixed", "bottom": "0", "left": "0", "right": "0",
        "display": "flex", "justifyContent": "space-between", "alignItems": "center",
        "padding": "12px 28px",
        "background": "rgba(13,17,23,0.97)",
        "borderTop": "1px solid rgba(255,255,255,0.08)",
        "zIndex": "1000", "backdropFilter": "blur(8px)",
    }),
])

# ── HELPER COMPONENTS ──────────────────────────────────────

def slide_question(parts, marcus_quote, icon=""):
    """
    parts: list of (text, highlight) tuples
    Example: [("How broken is the ", False), ("US airline system", True), ("?", False)]
    """
    title_spans = []
    if icon:
        title_spans.append(html.Span(icon + " ", style={"fontSize": "32px", "marginRight": "10px"}))
    for text, highlight in parts:
        title_spans.append(html.Span(text, style={
            "fontSize": "38px", "fontWeight": "800", "letterSpacing": "-1px", "lineHeight": "1.15",
            "whiteSpace": "pre",
            "color": "#2F81F7" if highlight else "#E6EDF3",
        }))
    return html.Div([
        html.Div(title_spans, style={"marginBottom": "14px", "display": "flex",
                                     "alignItems": "center", "flexWrap": "wrap"}),
        html.Div([
            html.Div([
                html.Div("👤", style={
                    "width": "36px", "height": "36px", "borderRadius": "50%",
                    "background": "rgba(210,153,34,0.15)", "border": "1px solid rgba(210,153,34,0.3)",
                    "display": "flex", "alignItems": "center", "justifyContent": "center",
                    "fontSize": "18px", "flexShrink": "0", "marginRight": "14px",
                }),
                html.Div([
                    html.Div("Marcus Reid  ·  Business Consultant  ·  Chicago, IL",
                             style={"fontSize": "10px", "color": "#7D8590", "fontWeight": "600",
                                    "letterSpacing": "0.06em", "textTransform": "uppercase",
                                    "marginBottom": "5px"}),
                    html.Div(f'"{marcus_quote}"',
                             style={"fontSize": "15px", "color": "#D29922", "fontStyle": "italic",
                                    "lineHeight": "1.5", "fontWeight": "500"}),
                ]),
            ], style={"display": "flex", "alignItems": "center"}),
        ], style={
            "background": "rgba(210,153,34,0.06)", "border": "1px solid rgba(210,153,34,0.15)",
            "borderRadius": "10px", "padding": "14px 18px", "marginBottom": "20px",
        }),
        html.Hr(style={"border": "none", "borderTop": "1px solid rgba(255,255,255,0.06)",
                       "margin": "0 0 20px"}),
    ])


def upsize_chart(fig, title_size=16, axis_size=12, tick_size=11):
    new_ann = []
    for a in fig.layout.annotations:
        d = a.to_plotly_json()
        f = d.get("font") or {}
        f["size"] = max(f.get("size") or 10, 11)
        d["font"] = f
        new_ann.append(d)
    fig.update_layout(
        title=dict(font=dict(size=title_size, color="#E6EDF3")),
        font=dict(size=tick_size),
        xaxis=dict(title=dict(font=dict(size=axis_size)), tickfont=dict(size=tick_size)),
        yaxis=dict(title=dict(font=dict(size=axis_size)), tickfont=dict(size=tick_size)),
        legend=dict(font=dict(size=tick_size)),
        annotations=new_ann if new_ann else fig.layout.annotations,
    )
    return fig


def _cover_slide():
    planes_data = [
        ("168px", "0.04", "25",   "10%", "5%",  None),
        ("112px", "0.06", "-15",  "60%", None,  "3%"),
        ("280px", "0.03", "45",   "30%", "60%", None),
        ("84px",  "0.07", "10",   "80%", "20%", None),
        ("210px", "0.025","-30",  "5%",  None,  "15%"),
        ("56px",  "0.08", "60",   "70%", "45%", None),
        ("140px", "0.035","20",   "45%", "8%",  None),
        ("98px",  "0.05", "-45",  "15%", None,  "35%"),
    ]
    plane_elements = [
        html.Div("✈", style={
            "position": "absolute", "fontSize": s, "opacity": o,
            "color": "#2F81F7", "transform": f"rotate({r}deg)",
            "pointerEvents": "none", "userSelect": "none",
            "top": t, "left": l if l else "auto", "right": ri if ri else "auto",
        }) for s, o, r, t, l, ri in planes_data
    ]

    chapter_items = [
        ("The System",      "Scale of the problem"),
        ("Where it breaks", "Airport congestion"),
        ("Why it breaks",   "Delay causes"),
        ("When it breaks",  "Hour × weekday"),
        ("Who to trust",    "FRI ranking"),
        ("Domino effect",   "Propagation & severity"),
        ("Full picture",    "Risk & seasonality"),
        ("Marcus decides",  "Recommended actions"),
    ]
    chapter_grid = html.Div([
        html.Div([
            html.Div(f"{i+1:02d}", style={"fontSize": "10px", "color": "#2F81F7",
                     "fontFamily": "monospace", "fontWeight": "700", "marginBottom": "6px"}),
            html.Div(title, style={"fontSize": "13px", "color": "#E6EDF3", "fontWeight": "500"}),
            html.Div(desc,  style={"fontSize": "11px", "color": "#7D8590", "marginTop": "3px"}),
        ], id={"type": "menu-nav", "index": i + 1}, n_clicks=0, style={
            "background": "rgba(22,27,34,0.8)", "border": "1px solid rgba(255,255,255,0.08)",
            "borderRadius": "8px", "padding": "16px 12px", "cursor": "pointer", "flex": "1",
            "minWidth": "0", "minHeight": "80px", "display": "flex", "flexDirection": "column",
            "justifyContent": "flex-start",
        })
        for i, (title, desc) in enumerate(chapter_items)
    ], style={"display": "flex", "gap": "8px", "width": "100%", "maxWidth": "1100px"})

    return html.Div([
        # Radial gradient overlay
        html.Div(style={
            "position": "absolute", "inset": "0", "pointerEvents": "none",
            "background": "radial-gradient(ellipse at 30% 50%, rgba(47,129,247,0.06) 0%, transparent 60%)",
            "zIndex": "0",
        }),
        *plane_elements,
        html.Div([
            html.Div("DATAFONOS · BIG DATA VISUALIZATION · BTS 2023", style={
                "fontSize": "10px", "color": "#7D8590", "letterSpacing": "3px",
                "marginBottom": "32px", "textAlign": "center",
            }),
            html.Div("Can Marcus", style={
                "fontSize": "88px", "fontWeight": "800", "color": "#E6EDF3",
                "lineHeight": "1", "textAlign": "center", "letterSpacing": "-4px",
            }),
            html.Div("Trust His Flight?", style={
                "fontSize": "88px", "fontWeight": "800", "color": "#2F81F7",
                "lineHeight": "1", "textAlign": "center", "letterSpacing": "-4px", "marginBottom": "28px",
            }),
            html.Div([
                html.Span("✈ ", style={"fontSize": "17px"}),
                html.Span(
                    '"I fly 80 times a year. I\'ve missed 3 connections this quarter. Tonight I look at the data."',
                    style={"fontStyle": "italic", "color": "#D29922"},
                ),
            ], style={
                "background": "rgba(210,153,34,0.08)", "border": "1px solid rgba(210,153,34,0.25)",
                "borderRadius": "12px", "padding": "20px 28px", "maxWidth": "640px",
                "margin": "0 auto 44px", "fontSize": "17px", "textAlign": "center", "color": "#D29922",
            }),
            html.Div([
                html.Div("256.8M hrs lost", style={
                    "background": "rgba(218,54,51,0.15)", "border": "1px solid rgba(218,54,51,0.3)",
                    "color": "#DA3633", "padding": "14px 28px", "borderRadius": "25px",
                    "fontSize": "15px", "fontWeight": "600",
                }),
                html.Div("79.4% on-time", style={
                    "background": "rgba(210,153,34,0.15)", "border": "1px solid rgba(210,153,34,0.3)",
                    "color": "#D29922", "padding": "14px 28px", "borderRadius": "25px",
                    "fontSize": "15px", "fontWeight": "600",
                }),
                html.Div("6.7M flights", style={
                    "background": "rgba(47,129,247,0.15)", "border": "1px solid rgba(47,129,247,0.3)",
                    "color": "#2F81F7", "padding": "14px 28px", "borderRadius": "25px",
                    "fontSize": "15px", "fontWeight": "600",
                }),
            ], style={"display": "flex", "gap": "12px", "justifyContent": "center", "marginBottom": "52px"}),
            chapter_grid,
        ], style={
            "position": "relative", "zIndex": "10", "padding": "60px 40px",
            "display": "flex", "flexDirection": "column", "alignItems": "center",
            "justifyContent": "center", "minHeight": "calc(100vh - 140px)",
        }),
    ], style={"position": "relative", "overflow": "hidden"})


def _slide8_content():
    return html.Div([
        html.Div([
            html.Div([
                html.Div([
                    html.Div(num, style={"fontSize": "10px", "fontWeight": "700", "color": col,
                                         "fontFamily": "monospace", "letterSpacing": "0.1em",
                                         "marginBottom": "10px"}),
                    html.Div(icon, style={"fontSize": "40px", "marginBottom": "14px", "lineHeight": "1"}),
                    html.Div(title, style={"fontSize": "17px", "fontWeight": "700",
                                           "color": "#E6EDF3", "marginBottom": "10px", "lineHeight": "1.3"}),
                    html.Div(desc, style={"fontSize": "12px", "color": "#7D8590", "lineHeight": "1.6"}),
                    html.Div(style={"height": "3px", "borderRadius": "3px",
                                    "background": col, "marginTop": "16px"}),
                ], style={
                    "padding": "28px 24px", "background": "#161B22", "borderRadius": "12px",
                    "border": "1px solid rgba(255,255,255,0.06)", "borderTop": f"4px solid {col}",
                    "height": "100%", "boxSizing": "border-box",
                }),
            ], style={"flex": "1"})
            for num, icon, title, desc, col in [
                ("01", "✅", "Book Delta or Alaska",
                 "FRI 82–84 · Most reliable major carriers · Best on-time rate + lowest severe delay rate",
                 "#2EA043"),
                ("02", "🌅", "Fly Tuesday or Thursday before 9am",
                 "Negative average delay · Aircraft starts day fresh · No accumulated rotation delays",
                 "#2F81F7"),
                ("03", "🚫", "Avoid JetBlue and Frontier",
                 "Severe delay rate >12% · Nearly 1 in 7 flights arrives 60+ min late",
                 "#DA3633"),
                ("04", "🌙", "Never book Sunday/Friday evenings",
                 "19–22h is the worst window · Up to 19 min average delay · 50k+ flights affected",
                 "#D29922"),
            ]
        ], style={"display": "flex", "gap": "16px", "marginBottom": "28px", "height": "220px"}),

        html.Div([
            html.Div([
                html.Span("✈", style={"fontSize": "14px", "color": "rgba(210,153,34,0.3)",
                                       "marginRight": "8px"}) for _ in range(12)
            ], style={"textAlign": "center", "marginBottom": "20px", "letterSpacing": "4px"}),

            html.Div('"Delta. Tuesday morning. Never Friday evening."',
                     style={"fontSize": "34px", "fontWeight": "800", "color": "#D29922",
                            "fontStyle": "italic", "textAlign": "center", "lineHeight": "1.4",
                            "letterSpacing": "-0.5px", "marginBottom": "14px"}),
            html.Div("— Marcus Reid · after analysing 6.7 million real flights · BTS 2023",
                     style={"fontSize": "12px", "color": "#7D8590", "textAlign": "center",
                            "marginBottom": "20px"}),

            html.Div([
                html.Div([
                    html.Div("6.7M", style={"fontSize": "22px", "fontWeight": "800",
                                             "color": "#2F81F7", "letterSpacing": "-1px"}),
                    html.Div("flights analysed", style={"fontSize": "11px", "color": "#7D8590"}),
                ], style={"textAlign": "center", "flex": "1"}),
                html.Div(style={"width": "1px", "background": "rgba(255,255,255,0.08)"}),
                html.Div([
                    html.Div("BTS 2023", style={"fontSize": "22px", "fontWeight": "800",
                                                 "color": "#2F81F7", "letterSpacing": "-1px"}),
                    html.Div("official data source", style={"fontSize": "11px", "color": "#7D8590"}),
                ], style={"textAlign": "center", "flex": "1"}),
                html.Div(style={"width": "1px", "background": "rgba(255,255,255,0.08)"}),
                html.Div([
                    html.Div("FRI", style={"fontSize": "22px", "fontWeight": "800",
                                            "color": "#2F81F7", "letterSpacing": "-1px"}),
                    html.Div("original KPI by Datafonos", style={"fontSize": "11px", "color": "#7D8590"}),
                ], style={"textAlign": "center", "flex": "1"}),
            ], style={"display": "flex", "gap": "0", "paddingTop": "16px",
                      "borderTop": "1px solid rgba(255,255,255,0.06)"}),
        ], style={
            "background": "linear-gradient(135deg, rgba(210,153,34,0.08) 0%, rgba(13,17,23,0) 60%)",
            "border": "1px solid rgba(210,153,34,0.2)",
            "borderRadius": "14px", "padding": "28px 36px",
        }),
    ], style={"padding": "0 0 100px"})


# ── CALLBACKS ──────────────────────────────────────────────

@app.callback(
    Output("current-slide", "data"),
    [Input("btn-prev", "n_clicks"),
     Input("btn-next", "n_clicks"),
     Input("btn-home", "n_clicks"),
     Input({"type": "menu-nav", "index": ALL}, "n_clicks")] +
    [Input(f"dot-{i}", "n_clicks") for i in range(9)] +
    [Input(f"panel-slide-{i}", "n_clicks") for i in range(1, 9)],
    State("current-slide", "data"),
    prevent_initial_call=True,
)
def navigate(*args):
    import json
    current = args[-1]
    ctx = dash.callback_context
    if not ctx.triggered:
        return current
    prop = ctx.triggered[0]["prop_id"]
    if "btn-next" in prop:
        return min(current + 1, TOTAL_SLIDES)
    if "btn-prev" in prop:
        return max(current - 1, 0)
    if "btn-home" in prop:
        return 0
    if prop.startswith("{"):
        id_dict = json.loads(prop.rsplit(".", 1)[0])
        return id_dict["index"]
    if "dot-" in prop:
        return int(prop.split(".")[0].split("-")[1])
    if "panel-slide-" in prop:
        return int(prop.split(".")[0].split("-")[2])
    return current


@app.callback(
    Output("side-panel", "style"),
    [Input("menu-toggle", "n_clicks")] +
    [Input(f"panel-slide-{i}", "n_clicks") for i in range(1, 9)],
    State("side-panel", "style"),
    prevent_initial_call=True,
)
def toggle_panel(*args):
    style = args[-1]
    ctx = dash.callback_context
    if not ctx.triggered:
        return style
    prop = ctx.triggered[0]["prop_id"]
    closed = {**style, "transform": "translateX(-100%)"}
    if "menu-toggle" in prop:
        is_open = style.get("transform", "translateX(-100%)") not in ("translateX(-100%)",)
        return {**style, "transform": "translateX(0px)" if not is_open else "translateX(-100%)"}
    return closed


@app.callback(
    [Output(f"dot-{i}", "style") for i in range(9)],
    Input("current-slide", "data"),
)
def update_dots(slide):
    return [_DOT_ACTIVE if i == slide else _DOT_INACTIVE for i in range(9)]


@app.callback(
    Output("data-ticker-s1", "children"),
    Input("ticker-interval-s1", "n_intervals"),
)
def update_ticker_s1(n):
    stats = [
        ("6,743,400", "flights analysed",          "#2F81F7"),
        ("256.8M",    "passenger-hours lost",       "#DA3633"),
        ("79.4%",     "system on-time rate",        "#D29922"),
        ("6.6 min",   "avg arrival delay",          "#2EA043"),
        ("1.3%",      "cancellation rate",          "#2EA043"),
        ("67%",       "delays airline-controllable","#DA3633"),
        ("43 pts",    "FRI gap: Delta vs Frontier", "#D29922"),
        ("−3 min",    "best slot: Tue 06:00",       "#2EA043"),
        ("13.8%",     "JetBlue severe delay rate",  "#DA3633"),
        ("19 min",    "worst slot: Sun 20:00",      "#DA3633"),
    ]
    idx = n % len(stats)
    value, label, color = stats[idx]
    return html.Div([
        html.Div(value, style={
            "fontSize": "88px", "fontWeight": "800", "color": color,
            "letterSpacing": "-3px", "lineHeight": "1", "marginBottom": "10px",
            "textAlign": "center",
        }),
        html.Div(label, style={
            "fontSize": "15px", "color": "#7D8590", "marginBottom": "28px",
            "textAlign": "center", "letterSpacing": "0.02em",
        }),
        html.Div([
            html.Div(style={
                "width": "7px", "height": "7px", "borderRadius": "50%",
                "background": color if i == idx else "rgba(255,255,255,0.1)",
                "transition": "all 0.3s", "cursor": "pointer", "flexShrink": "0",
            }) for i in range(len(stats))
        ], style={
            "display": "flex", "gap": "8px", "justifyContent": "center",
            "flexWrap": "wrap", "maxWidth": "240px", "margin": "0 auto",
        }),
        html.Div([
            html.Div("← prev", style={"fontSize": "10px", "color": "rgba(255,255,255,0.15)", "cursor": "pointer"}),
            html.Div(f"{idx+1} / {len(stats)}", style={"fontSize": "10px", "color": "#7D8590"}),
            html.Div("next →", style={"fontSize": "10px", "color": "rgba(255,255,255,0.15)", "cursor": "pointer"}),
        ], style={"display": "flex", "justifyContent": "space-between", "width": "100%",
                  "maxWidth": "220px", "margin": "12px auto 0"}),
        html.Div([
            html.Span("Next: ", style={"color": "rgba(255,255,255,0.2)", "fontSize": "11px"}),
            html.Span(stats[(idx+1) % len(stats)][0],
                      style={"color": "rgba(255,255,255,0.3)", "fontSize": "11px", "fontWeight": "600"}),
            html.Span(f" · {stats[(idx+1) % len(stats)][1]}",
                      style={"color": "rgba(255,255,255,0.15)", "fontSize": "11px"}),
        ], style={"textAlign": "center", "marginTop": "16px", "padding": "8px 0",
                  "borderTop": "1px solid rgba(255,255,255,0.04)"}),
    ], style={"textAlign": "center", "display": "flex", "flexDirection": "column",
              "alignItems": "center", "justifyContent": "center", "height": "100%"})


@app.callback(
    Output("slide-content", "children"),
    Output("slide-indicator", "children"),
    Output("slide-counter", "children"),
    Input("current-slide", "data"),
    Input("year-filter", "value"),
    Input("airline-filter", "value"),
    Input("airport-filter", "value"),
)
def render_slide(slide, selected_year, selected_airline, selected_airport):
    if slide == 0:
        return _cover_slide(), "", ""

    # ── filter df ──
    df = DATA["clean"].copy()
    if selected_year and selected_year != "All":
        df = df[df["YEAR"] == int(selected_year)]
    if selected_airline and selected_airline != "All":
        df = df[df["CARRIER_NAME"] == selected_airline]
    if selected_airport and selected_airport != "All" and "ORIGIN" in df.columns:
        df = df[(df["ORIGIN"] == selected_airport) | (df["DEST"] == selected_airport)]

    df_all = DATA["clean"].copy()
    if selected_year and selected_year != "All":
        df_all = df_all[df_all["YEAR"] == int(selected_year)]
    if selected_airline and selected_airline != "All":
        df_all = df_all[df_all["CARRIER_NAME"] == selected_airline]
    if selected_airport and selected_airport != "All" and "ORIGIN" in df_all.columns:
        df_all = df_all[(df_all["ORIGIN"] == selected_airport) | (df_all["DEST"] == selected_airport)]

    metrics = compute_metrics(df, df_all)

    try:
        kpi = DATA["carrier_kpi"].copy()
        kpi["FRI"] = (
            0.50 * kpi["on_time_pct"] +
            0.30 * (1 - normalize(kpi["avg_arr_delay"])) * 100 +
            0.20 * (1 - kpi["cancel_rate"].fillna(0)) * 100
        ).round(1)
        best = kpi.sort_values("FRI", ascending=False).iloc[0]
        fri_rec = html.Div(
            f"Top carrier: {best['CARRIER_NAME']} — FRI {best['FRI']:.0f}/100",
            style={"fontSize": "12px", "color": C["green"], "padding": "6px 0"}
        )
    except Exception:
        fri_rec = ""

    indicator = SLIDE_TITLES[slide]
    counter = f"{slide} / {TOTAL_SLIDES}"

    # ── Slide 1: The System ──
    if slide == 1:
        kpi_rows = [
            ("On-Time Performance",    "79.4%",  "#D29922", "System-wide punctuality"),
            ("Avg Arrival Delay",      "6.6 min","#2EA043", "When arriving late"),
            ("Cancellation Rate",      "1.3%",   "#2EA043", "Flights cancelled"),
            ("Flight Reliability Index","80.5",  "#2EA043", "Composite 0–100 score"),
        ]
        kpi_cards = html.Div([
            html.Div([
                html.Div(style={"height": "5px", "borderRadius": "5px 5px 0 0", "background": color}),
                html.Div([
                    html.Div(label, style={"fontSize": "13px", "color": "#9BA3AF", "fontWeight": "600",
                                           "letterSpacing": "0.08em", "textTransform": "uppercase",
                                           "marginBottom": "12px"}),
                    html.Div(value, style={"fontSize": "52px", "fontWeight": "700", "color": color,
                                           "letterSpacing": "-2px", "lineHeight": "1", "marginBottom": "8px"}),
                    html.Div(sub, style={"fontSize": "13px", "color": "#9BA3AF"}),
                ], style={"padding": "24px 28px"}),
            ], style={"flex": "1", "background": "#161B22", "borderRadius": "0 0 12px 12px",
                      "border": "1px solid rgba(255,255,255,0.06)", "borderTop": "none",
                      "minHeight": "140px"})
            for label, value, color, sub in kpi_rows
        ], style={"display": "flex", "gap": "16px", "padding": "8px 0 16px"})

        content = html.Div([
            slide_question(
                [("How broken is the ", False), ("US airline system", True), ("?", False)],
                "I had no idea. 256 million hours. That's not a delay, that's a crisis.",
                "📊",
            ),
            html.Div([
                html.Div(style={
                    "position": "absolute", "top": "50%", "left": "50%",
                    "transform": "translate(-50%,-50%)",
                    "width": "400px", "height": "200px",
                    "background": "radial-gradient(ellipse, rgba(218,54,51,0.12) 0%, transparent 70%)",
                    "pointerEvents": "none", "zIndex": "0",
                }),
                html.Div("256.8M", style={
                    "fontSize": "120px", "fontWeight": "800", "color": "#DA3633",
                    "letterSpacing": "-6px", "lineHeight": "1", "textAlign": "center",
                    "position": "relative", "zIndex": "1",
                }),
                html.Div("passenger-hours lost to delays in 2023",
                         style={"fontSize": "18px", "color": "#7D8590", "textAlign": "center",
                                "marginTop": "4px", "position": "relative", "zIndex": "1"}),
            ], style={"padding": "32px 0 28px", "position": "relative",
                      "borderBottom": "1px solid rgba(255,255,255,0.06)"}),
            kpi_cards,
            html.Div([
                # TWO BOXES ROW
                html.Div([
                    # LEFT BOX — Marcus's verdict
                    html.Div([
                        html.Div("✅  Marcus's verdict", style={
                            "fontSize": "11px", "fontWeight": "700", "color": "#2EA043",
                            "textTransform": "uppercase", "letterSpacing": "0.1em", "marginBottom": "20px",
                        }),
                        html.Div([
                            html.Span("✈  ", style={"fontSize": "32px", "color": "#2EA043"}),
                            html.Span("Delta Air Lines", style={
                                "fontSize": "36px", "fontWeight": "800", "color": "#2EA043",
                                "letterSpacing": "-1px",
                            }),
                        ], style={"marginBottom": "10px", "display": "flex", "alignItems": "center"}),
                        html.Div("Tuesday morning", style={
                            "fontSize": "24px", "fontWeight": "700", "color": "#E6EDF3",
                            "letterSpacing": "-0.5px", "marginBottom": "6px",
                        }),
                        html.Div("Never Friday or Sunday evening", style={
                            "fontSize": "16px", "color": "#7D8590", "marginBottom": "24px",
                        }),
                        html.Div([
                            html.Div([
                                html.Div("FRI 84", style={"fontSize": "20px", "fontWeight": "800",
                                         "color": "#2EA043", "letterSpacing": "-0.5px"}),
                                html.Div("reliability score", style={"fontSize": "11px", "color": "#7D8590", "marginTop": "2px"}),
                            ], style={"flex": "1", "textAlign": "center", "padding": "12px 0",
                                      "borderRight": "1px solid rgba(255,255,255,0.06)"}),
                            html.Div([
                                html.Div("5.5%", style={"fontSize": "20px", "fontWeight": "800",
                                         "color": "#2EA043", "letterSpacing": "-0.5px"}),
                                html.Div("severe delay rate", style={"fontSize": "11px", "color": "#7D8590", "marginTop": "2px"}),
                            ], style={"flex": "1", "textAlign": "center", "padding": "12px 0",
                                      "borderRight": "1px solid rgba(255,255,255,0.06)"}),
                            html.Div([
                                html.Div("83.6%", style={"fontSize": "20px", "fontWeight": "800",
                                         "color": "#2EA043", "letterSpacing": "-0.5px"}),
                                html.Div("on-time rate", style={"fontSize": "11px", "color": "#7D8590", "marginTop": "2px"}),
                            ], style={"flex": "1", "textAlign": "center", "padding": "12px 0"}),
                        ], style={
                            "display": "flex", "background": "rgba(46,160,67,0.06)",
                            "border": "1px solid rgba(46,160,67,0.15)", "borderRadius": "8px",
                            "marginBottom": "20px",
                        }),
                        html.Div([
                            html.Span("vs Frontier FRI 41 · ", style={"color": "#DA3633", "fontWeight": "600"}),
                            html.Span("43-point gap invisible from average delay alone",
                                     style={"color": "#7D8590"}),
                        ], style={"fontSize": "12px", "borderTop": "1px solid rgba(255,255,255,0.06)",
                                  "paddingTop": "14px"}),
                    ], style={
                        "flex": "1", "padding": "28px 32px",
                        "background": "linear-gradient(135deg, rgba(46,160,67,0.08) 0%, rgba(22,27,34,1) 50%)",
                        "borderRadius": "14px", "border": "1px solid rgba(46,160,67,0.2)",
                        "borderTop": "4px solid #2EA043",
                    }),
                    # RIGHT BOX — Animated data ticker
                    html.Div([
                        html.Div("Live data pulse", style={
                            "fontSize": "11px", "fontWeight": "600", "color": "#7D8590",
                            "textTransform": "uppercase", "letterSpacing": "0.08em", "marginBottom": "20px",
                            "textAlign": "center",
                        }),
                        html.Div(id="data-ticker-s1", style={"flex": "1", "display": "flex",
                                                               "alignItems": "center", "justifyContent": "center"}),
                        dcc.Interval(id="ticker-interval-s1", interval=2500, n_intervals=0),
                    ], style={
                        "flex": "1", "padding": "28px 32px",
                        "background": "linear-gradient(135deg, rgba(47,129,247,0.08) 0%, rgba(22,27,34,1) 60%)",
                        "borderRadius": "14px", "border": "1px solid rgba(255,255,255,0.06)",
                        "borderTop": "4px solid #2F81F7", "marginLeft": "16px",
                        "display": "flex", "flexDirection": "column", "justifyContent": "center",
                    }),
                ], style={"display": "flex", "gap": "0", "marginTop": "20px"}),

                # Footer
                html.Div(
                    "6,743,400 real flights · BTS 2023 · OpenFlights airports.dat · FRI original KPI by Datafonos",
                    style={
                        "textAlign": "center", "fontSize": "10px",
                        "color": "rgba(125,133,144,0.3)", "letterSpacing": "0.06em",
                        "paddingBottom": "100px", "marginTop": "6px", "userSelect": "none",
                    }
                ),
            ], style={"marginTop": "8px", "overflow": "hidden"}),
        ])

    # ── Slide 2: Where it breaks ──
    elif slide == 2:
        _map_fig = build_airport_map(df, selected_airport)
        _map_fig.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),
            height=None,
            autosize=True,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            geo=dict(bgcolor="rgba(0,0,0,0)", fitbounds="locations"),
        )
        _sidebar_items = [
            ("ATL", "Atlanta", "16.3 min", "#DA3633"),
            ("ORD", "Chicago O'Hare", "14.1 min", "#DA3633"),
            ("EWR", "Newark", "19.8 min", "#DA3633"),
            ("MCO", "Orlando", "18.2 min", "#DA3633"),
            ("LAX", "Los Angeles", "11.2 min", "#D29922"),
            ("DEN", "Denver", "9.4 min", "#D29922"),
            ("SEA", "Seattle", "7.1 min", "#2EA043"),
            ("SLC", "Salt Lake City", "6.8 min", "#2EA043"),
        ]
        content = html.Div([
            # Padded header: question + Marcus quote
            html.Div([
                slide_question([("Where do delays ", False), ("hit hardest", True), ("?", False)],
                               "Florida is a nightmare. My hub ORD is surprisingly safe.", "🗺️"),
                html.Div("Bubble size = flight volume  •  Color = avg arrival delay",
                         style={"fontSize": "12px", "color": "#7D8590", "marginTop": "-12px",
                                "marginBottom": "14px", "letterSpacing": "0.04em"}),
            ], style={"padding": "0 28px"}),
            # Map + sidebar row — zero padding so map goes edge to edge
            html.Div([
                # FIX 3: Map container — flush left, flex:3, no extra padding
                html.Div([
                    dcc.Graph(
                        id="slide-map",
                        figure=_map_fig,
                        config={"displayModeBar": False},
                        responsive=True,
                        style={
                            "height": "calc(100vh - 290px)",
                            "minHeight": "500px",
                            "width": "100%",
                        },
                    ),
                ], style={
                    "flex": "3",
                    "marginLeft": "-28px",
                    "paddingLeft": "0",
                    "overflow": "hidden",
                }),
                # FIX 5: Sidebar — stretch to full map height
                html.Div([
                    html.Div("Top delayed airports", style={
                        "fontSize": "11px", "fontWeight": "600", "color": "#7D8590",
                        "textTransform": "uppercase", "letterSpacing": "0.08em", "marginBottom": "14px",
                    }),
                    *[html.Div([
                        html.Div([
                            html.Div(iata, style={"fontSize": "13px", "fontWeight": "700",
                                                   "color": color, "minWidth": "36px"}),
                            html.Div(city, style={"fontSize": "12px", "color": "#9BA3AF", "flex": "1"}),
                            html.Div(avg, style={"fontSize": "12px", "fontWeight": "600",
                                                  "color": color, "textAlign": "right"}),
                        ], style={"display": "flex", "alignItems": "center", "gap": "8px",
                                  "padding": "8px 0",
                                  "borderBottom": "1px solid rgba(255,255,255,0.04)"}),
                    ]) for iata, city, avg, color in _sidebar_items],
                    html.Div([
                        html.Div("Avg delay", style={"fontSize": "10px", "color": "#7D8590",
                                                      "marginBottom": "6px"}),
                        html.Div([
                            html.Span("■ ", style={"color": "#DA3633"}),
                            html.Span("> 15 min", style={"color": "#9BA3AF", "fontSize": "11px"}),
                        ]),
                        html.Div([
                            html.Span("■ ", style={"color": "#D29922"}),
                            html.Span("8–15 min", style={"color": "#9BA3AF", "fontSize": "11px"}),
                        ]),
                        html.Div([
                            html.Span("■ ", style={"color": "#2EA043"}),
                            html.Span("< 8 min", style={"color": "#9BA3AF", "fontSize": "11px"}),
                        ]),
                    ], style={"marginTop": "auto", "paddingTop": "16px",
                              "borderTop": "1px solid rgba(255,255,255,0.06)"}),
                ], style={
                    "flex": "1",
                    "minWidth": "220px",
                    "maxWidth": "260px",
                    "height": "calc(100vh - 290px)",
                    "minHeight": "500px",
                    "overflowY": "auto",
                    "display": "flex",
                    "flexDirection": "column",
                    "padding": "20px 16px 20px 20px",
                    "background": "rgba(22,27,34,0.8)",
                    "borderLeft": "1px solid rgba(255,255,255,0.06)",
                    "marginRight": "-28px",
                }),
            ], style={"display": "flex", "alignItems": "stretch"}),
        ], style={"padding": "0"})

    # ── Slide 3: Why it breaks ──
    elif slide == 3:
        donut_fig = upsize_chart(build_donut(df))
        content = html.Div([
            slide_question([("Whose fault ", False), ("are the delays", True), ("?", False)],
                           "67% is the airline's fault. I can actually fix this.", "🔍"),
            html.Div([
                html.Div([
                    dcc.Graph(figure=donut_fig, config={"displayModeBar": False},
                              style={"height": "560px"}),
                ], style={"flex": "6"}),
                html.Div([
                    html.Div("Delay cause breakdown", style={
                        "fontSize": "13px", "fontWeight": "600", "color": "#7D8590",
                        "textTransform": "uppercase", "letterSpacing": "0.08em", "marginBottom": "20px",
                    }),
                    *[html.Div([
                        html.Div(style={
                            "width": "12px", "height": "12px", "borderRadius": "2px",
                            "background": col, "flexShrink": "0", "marginTop": "3px",
                        }),
                        html.Div([
                            html.Div(label, style={"fontSize": "14px", "color": "#E6EDF3",
                                                   "fontWeight": "500", "marginBottom": "2px"}),
                            html.Div(pct, style={"fontSize": "28px", "fontWeight": "800",
                                                 "color": col, "letterSpacing": "-1px", "lineHeight": "1"}),
                            html.Div(desc, style={"fontSize": "11px", "color": "#7D8590", "marginTop": "2px"}),
                        ]),
                    ], style={"display": "flex", "gap": "12px", "padding": "14px 0",
                              "borderBottom": "1px solid rgba(255,255,255,0.06)"})
                    for col, label, pct, desc in [
                        ("#DA3633", "Late Aircraft",      "40%", "Controllable by airline"),
                        ("#D29922", "Airline Operations", "27%", "Controllable by airline"),
                        ("#2F81F7", "NAS / Air Traffic",  "22%", "External factor"),
                        ("#79c0ff", "Weather",            "11%", "External factor"),
                        ("#7D8590", "Security",            "2%", "External factor"),
                    ]],
                    html.Div([
                        html.Div("67%", style={"fontSize": "48px", "fontWeight": "800",
                                               "color": "#DA3633", "letterSpacing": "-2px", "lineHeight": "1"}),
                        html.Div("of delays are the airline's fault",
                                 style={"fontSize": "13px", "color": "#7D8590", "marginTop": "4px"}),
                    ], style={
                        "background": "rgba(218,54,51,0.08)", "border": "1px solid rgba(218,54,51,0.2)",
                        "borderRadius": "10px", "padding": "20px", "marginTop": "20px", "textAlign": "center",
                    }),
                ], style={"flex": "4", "padding": "0 0 0 32px", "display": "flex",
                          "flexDirection": "column", "justifyContent": "flex-start",
                          "paddingTop": "20px"}),
            ], style={"display": "flex", "alignItems": "stretch", "height": "calc(100vh - 240px)"}),
        ])

    # ── Slide 4: When it breaks ──
    elif slide == 4:
        content = html.Div([
            slide_question([("When is the ", False), ("worst time to fly", True), ("?", False)],
                           "I always book Friday afternoon. That's literally the worst slot.", "⏰"),
            get_panel(
                "When delays peak — day × hour heatmap",
                "Avg. arrival delay by day/hour",
                [dcc.Graph(figure=upsize_chart(build_heatmap(df)),
                           config={"displayModeBar": False}, style={"height": "520px"})]
            ),
        ])

    # ── Slide 5: Who to trust ──
    elif slide == 5:
        content = html.Div([
            slide_question([("Which airline should ", False), ("Marcus trust", True), ("?", False)],
                           "43 points separate Delta from Frontier. I had no idea.", "✈️"),
            get_panel(
                "Carrier reliability — Flight Reliability Index (FRI)",
                "50% on-time + 30% low delay + 20% low cancel",
                [dcc.Graph(figure=upsize_chart(build_fri_chart(df, selected_airline)),
                           config={"displayModeBar": False}, style={"height": "580px"})]
            ),
            html.Div(fri_rec, style={"marginTop": "10px", "paddingLeft": "4px"}),
        ])

    # ── Slide 6: The domino effect ──
    elif slide == 6:
        content = html.Div([
            slide_question([("Does a late departure ", False), ("always mean", True), (" a late arrival?", False)],
                           "Board late. Arrive late. Every single time.", "⛓️"),
            dbc.Row([
                dbc.Col(
                    get_panel(
                        "Delay propagation — dep vs arr delay",
                        "Sample of 50 k flights • Color = airline • Dashed = regression",
                        [dcc.Graph(figure=upsize_chart(build_scatter_propagation(df)),
                                   config={"displayModeBar": False}, style={"height": "480px"})]
                    ),
                    md=7, style={"marginBottom": "16px"}
                ),
                dbc.Col(
                    get_panel(
                        "Severe delay rate by carrier (ARR_DELAY > 60 min)",
                        "Green <3 % · Amber 3–6 % · Red >6 %",
                        [dcc.Graph(figure=upsize_chart(build_severe_delay_chart(df)),
                                   config={"displayModeBar": False}, style={"height": "480px"})]
                    ),
                    md=5, style={"marginBottom": "16px"}
                ),
            ], className="g-4"),
        ])

    # ── Slide 7: The full picture ──
    elif slide == 7:
        content = html.Div([
            slide_question([("What does the ", False), ("future look like", True), (" for each airport?", False)],
                           "Now I know when, where and why. Time to decide.", "🔮"),
            get_panel(
                "Airports with highest predicted delay risk",
                "Based on avg delay. Red >12, Amber 8–12, Green <8.",
                [dcc.Graph(figure=upsize_chart(build_predictive_chart(df)),
                           config={"displayModeBar": False}, style={"height": "420px"})]
            ),
            html.Div(style={"height": "20px"}),
            get_panel(
                "Seasonal on-time performance — top 5 carriers",
                "Monthly on-time % (DL · AA · UA · WN · AS) — reveals summer collapse patterns",
                [dcc.Graph(figure=upsize_chart(build_seasonality_chart(df)),
                           config={"displayModeBar": False}, style={"height": "400px"})]
            ),
        ])

    # ── Slide 8: Marcus decides ──
    else:
        content = html.Div([
            slide_question([("What should ", False), ("Marcus do", True), (" now?", False)],
                           "Delta. Tuesday morning. Data confirmed.", "✅"),
            _slide8_content(),
        ])

    return content, indicator, counter



# ── RUN ────────────────────────────────────────────────────

if __name__ == "__main__":
    port = 8050
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        port = int(sys.argv[1])
    print("\n" + "=" * 56)
    print("  MARCUS REID — DELAY RISK DASHBOARD v2")
    print("=" * 56)
    print(f"  Open: http://127.0.0.1:{port}")
    print("  Stop: Ctrl+C")
    print("=" * 56 + "\n")
    app.run(debug=True, host="127.0.0.1", port=port)
