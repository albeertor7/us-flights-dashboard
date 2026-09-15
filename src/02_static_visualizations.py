"""
============================================================
 MARCUS REID'S FLIGHT RELIABILITY PROJECT
 02 — Static Visualizations (Matplotlib + Seaborn)
============================================================
Viz Lead notebook — run after 01_data_pipeline.py.

Techniques applied (per course modules 3–5):
  ✓ Grammar of Graphics: data → aesthetics → geoms → scales
  ✓ Pre-attentive attributes: color for intensity, size for magnitude
  ✓ Gestalt principles: proximity, similarity, enclosure
  ✓ Cognitive load reduction: direct labels, minimal chartjunk
  ✓ Accessible color palettes (ColorBrewer-inspired, colorblind-safe)
  ✓ Honest axes (start at 0, no truncation)
  ✓ Seaborn statistical plots: boxplot, heatmap, violin
  ✓ Matplotlib customization for publication quality
============================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path

DATA_DIR  = Path("data")
OUT_DIR   = Path("figures")
OUT_DIR.mkdir(exist_ok=True)

# ── Global style ─────────────────────────────────────────
# Minimalist, aviation-inspired palette
# ColorBrewer sequential: accessible for color blindness
PALETTE_CARRIER = {
    "American Airlines":  "#185FA5",
    "Delta Air Lines":    "#1D9E75",
    "United Airlines":    "#D85A30",
    "Southwest Airlines": "#BA7517",
    "JetBlue Airways":    "#7F77DD",
    "Alaska Airlines":    "#0F6E56",
    "Spirit Airlines":    "#639922",
    "Frontier Airlines":  "#993556",
}
ACCENT   = "#185FA5"   # strong blue
MUTED    = "#B4B2A9"   # gray for non-focal elements
DANGER   = "#E24B4A"   # red for bad performance
SUCCESS  = "#1D9E75"   # green for good performance
BG       = "#FAFAF8"   # near-white background

plt.rcParams.update({
    "figure.facecolor":  BG,
    "axes.facecolor":    BG,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.spines.left":  False,
    "axes.spines.bottom":True,
    "axes.grid":         True,
    "grid.color":        "#E8E6DF",
    "grid.linewidth":    0.6,
    "font.family":       "sans-serif",
    "font.size":         11,
    "axes.titlesize":    14,
    "axes.titleweight":  "medium",
    "axes.labelsize":    11,
    "xtick.labelsize":   10,
    "ytick.labelsize":   10,
})

def marcus_quote(ax, text, fontsize=9):
    """Add Marcus's voice as a contextual annotation."""
    ax.annotate(
        f'  "{text}"',
        xy=(0, -0.14), xycoords="axes fraction",
        fontsize=fontsize, fontstyle="italic",
        color="#5F5E5A",
        bbox=dict(boxstyle="round,pad=0.3", fc=BG, ec="#D3D1C7", lw=0.5)
    )

def save(fig, name):
    path = OUT_DIR / f"{name}.png"
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor=BG)
    print(f"  ✓ {path}")
    plt.close(fig)


# ══════════════════════════════════════════════════════════
#  VIZ 1 — HEATMAP: Average delay by Hour × Day of Week
#  Module 5 concept: heatmap geom, sequential color scale,
#  pre-attentive attribute (color intensity)
# ══════════════════════════════════════════════════════════
def plot_heatmap_hour_day(df_heat: pd.DataFrame):
    print("\n[VIZ 1] Heatmap hour × day…")

    pivot = df_heat.pivot_table(
        index="DAY_NAME", columns="HOUR_DEP",
        values="avg_delay", aggfunc="mean"
    )
    day_order = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
    pivot = pivot.reindex(day_order)

    fig, ax = plt.subplots(figsize=(14, 5))
    fig.patch.set_facecolor(BG)

    # Sequential palette: white→deep blue (colorblind-safe, course module 5)
    cmap = sns.color_palette("YlOrRd", as_cmap=True)

    sns.heatmap(
        pivot, ax=ax, cmap=cmap,
        annot=True, fmt=".0f", annot_kws={"size": 8, "color": "#2C2C2A"},
        linewidths=0.3, linecolor="#E8E6DF",
        cbar_kws={"label": "Avg. arrival delay (min)", "shrink": 0.6},
        vmin=-5, vmax=35
    )

    ax.set_title(
        "When does the US air system suffer most?  "
        "Avg. arrival delay by hour and weekday (2023)",
        pad=14, loc="left"
    )
    ax.set_xlabel("Scheduled departure hour")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=0)
    ax.tick_params(axis="y", rotation=0)

    # Highlight Marcus's worst slot: Monday 6–8am
    ax.add_patch(mpatches.FancyBboxPatch(
        (6, -0.05), 3, 1.1, transform=ax.get_xaxis_transform(),
        boxstyle="round,pad=0.02", linewidth=1.5,
        edgecolor=DANGER, facecolor="none", zorder=5,
        clip_on=False
    ))
    ax.text(7.5, -0.18, "Marcus's\nslot", transform=ax.get_xaxis_transform(),
            ha="center", fontsize=8, color=DANGER, fontweight="medium")

    marcus_quote(ax, "Monday mornings. I knew it. Now I have proof.")
    fig.tight_layout()
    save(fig, "viz1_heatmap_hour_day")


# ══════════════════════════════════════════════════════════
#  VIZ 2 — BOX PLOTS: Delay distribution per carrier
#  Module 5 concept: boxplot geom, shows distribution not
#  just mean (avoids Simpson's paradox), ordered by median
# ══════════════════════════════════════════════════════════
def plot_boxplot_carriers(df_clean: pd.DataFrame, carrier_kpi: pd.DataFrame):
    print("[VIZ 2] Box plots by carrier…")

    top_carriers = carrier_kpi.head(8)["OP_CARRIER"].tolist()
    df_box = df_clean[
        df_clean["OP_CARRIER"].isin(top_carriers) &
        (df_clean["ARR_DELAY_VIZ"] >= -30)
    ].copy()

    # Order by median delay (ascending = best first)
    order = (
        df_box.groupby("CARRIER_NAME")["ARR_DELAY_VIZ"]
        .median().sort_values().index.tolist()
    )

    fig, ax = plt.subplots(figsize=(12, 6))

    colors = [
        SUCCESS if i < len(order)//2 else DANGER
        for i in range(len(order))
    ]
    bp = sns.boxplot(
        data=df_box, x="ARR_DELAY_VIZ", y="CARRIER_NAME",
        order=order, ax=ax, width=0.55,
        palette=colors,
        flierprops={"marker":".", "ms":2, "alpha":0.15},
        medianprops={"color":"white","lw":2},
        boxprops={"alpha":0.85},
        whiskerprops={"lw":1},
        capprops={"lw":1}
    )

    ax.axvline(0, color="#444441", lw=0.8, ls="--", alpha=0.5, label="On-time threshold")
    ax.axvline(15, color=DANGER, lw=0.8, ls=":", alpha=0.6, label="BTS 'delayed' threshold (15 min)")

    ax.set_title("How are delays actually distributed?  Arrival delay by carrier (2023)",
                 pad=14, loc="left")
    ax.set_xlabel("Arrival delay (minutes)  — negative = early")
    ax.set_ylabel("")
    ax.legend(fontsize=9, framealpha=0.5)

    # Annotation: box plots reveal what averages hide
    ax.text(0.98, 0.02,
            "Each box shows median, IQR and whiskers.\nAverages alone hide the full story.",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8.5, color="#5F5E5A",
            bbox=dict(fc=BG, ec="#D3D1C7", lw=0.5, boxstyle="round,pad=0.3"))

    marcus_quote(ax, "Same average delay, very different tails. That matters when I have a connection.")
    fig.tight_layout()
    save(fig, "viz2_boxplot_carriers")


# ══════════════════════════════════════════════════════════
#  VIZ 3 — STACKED BAR: Delay cause breakdown
#  Module 5: bar geom, composition chart, sequential colors
#  Answers "whose fault is it?" — key for Marcus
# ══════════════════════════════════════════════════════════
def plot_delay_causes(delay_causes: pd.DataFrame):
    print("[VIZ 3] Delay causes stacked bar…")

    cause_cols = {
        "carrier_delay":     "Airline's fault",
        "late_ac_delay":     "Late incoming aircraft",
        "nas_delay":         "Air traffic control",
        "weather_delay":     "Weather",
        "security_delay":    "Security"
    }
    # Rename columns
    plot_df = delay_causes.copy()
    for old, new in {"CARRIER_DELAY":"carrier_delay","LATE_AIRCRAFT_DELAY":"late_ac_delay",
                     "NAS_DELAY":"nas_delay","WEATHER_DELAY":"weather_delay",
                     "SECURITY_DELAY":"security_delay"}.items():
        if old in plot_df.columns:
            plot_df.rename(columns={old: new}, inplace=True)

    # Sort by total delay
    plot_df["total"] = plot_df[list(cause_cols.keys())].sum(axis=1)
    plot_df = plot_df.sort_values("total", ascending=True)

    # Colorblind-safe palette for causes (diverging: controllable vs uncontrollable)
    cause_colors = {
        "Airline's fault":          "#E24B4A",
        "Late incoming aircraft":   "#D85A30",
        "Air traffic control":      "#BA7517",
        "Weather":                  "#378ADD",
        "Security":                 "#888780",
    }
    labels = list(cause_cols.values())
    col_keys = list(cause_cols.keys())

    fig, ax = plt.subplots(figsize=(12, 6))

    lefts = np.zeros(len(plot_df))
    for col, label in zip(col_keys, labels):
        if col not in plot_df.columns:
            continue
        vals = plot_df[col].values
        bars = ax.barh(
            plot_df["CARRIER_NAME"], vals,
            left=lefts, color=cause_colors[label], label=label, height=0.6
        )
        # Direct labels for large segments
        for bar, left, val in zip(bars, lefts, vals):
            if val > 1.5:
                ax.text(left + val/2, bar.get_y() + bar.get_height()/2,
                        f"{val:.1f}", ha="center", va="center",
                        fontsize=8, color="white", fontweight="medium")
        lefts += vals

    ax.set_title("What causes the delays?  Average delay minutes per cause (2023)",
                 pad=14, loc="left")
    ax.set_xlabel("Average delay contribution (minutes per flight)")
    ax.set_ylabel("")
    ax.legend(loc="lower right", fontsize=9, framealpha=0.5,
              title="Cause category", title_fontsize=9)

    # Controllable vs uncontrollable annotation
    ax.axvspan(0, 0, alpha=0)  # spacer
    ax.text(0.01, 1.02, "Red tones = controllable by airline   Blue = external factors",
            transform=ax.transAxes, fontsize=8.5, color="#5F5E5A")

    marcus_quote(ax, "If it's the airline's fault, I should switch. If it's weather, nowhere to hide.")
    fig.tight_layout()
    save(fig, "viz3_delay_causes")


# ══════════════════════════════════════════════════════════
#  VIZ 4 — SCATTER: Distance vs. Average Delay
#  Module 5: scatter geom, bubble size = flight volume
#  Shows whether long routes = more delay (often yes)
# ══════════════════════════════════════════════════════════
def plot_scatter_distance_delay(df_clean: pd.DataFrame, carrier_kpi: pd.DataFrame):
    print("[VIZ 4] Scatter distance vs delay…")

    top_carriers = carrier_kpi.head(6)["OP_CARRIER"].tolist()
    df_s = (
        df_clean[df_clean["OP_CARRIER"].isin(top_carriers)]
        .groupby(["OP_CARRIER","CARRIER_NAME","DISTANCE"])
        .agg(avg_delay=("ARR_DELAY","mean"),
             flights=("ARR_DELAY","count"))
        .reset_index()
    )
    # Bin distance into ranges for cleaner scatter
    df_s["dist_bin"] = pd.cut(df_s["DISTANCE"], bins=20)
    df_s = (
        df_s.groupby(["OP_CARRIER","CARRIER_NAME","dist_bin"])
        .agg(avg_delay=("avg_delay","mean"),
             flights=("flights","sum"),
             dist_mid=("DISTANCE","mean"))
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(11, 6))

    for carrier, grp in df_s.groupby("CARRIER_NAME"):
        color = PALETTE_CARRIER.get(carrier, MUTED)
        ax.scatter(
            grp["dist_mid"], grp["avg_delay"],
            s=grp["flights"]/200, alpha=0.65,
            color=color, label=carrier, zorder=3
        )
        # Trend line per carrier
        if len(grp) > 3:
            z = np.polyfit(grp["dist_mid"], grp["avg_delay"], 1)
            p = np.poly1d(z)
            x_range = np.linspace(grp["dist_mid"].min(), grp["dist_mid"].max(), 50)
            ax.plot(x_range, p(x_range), color=color, lw=1.2, alpha=0.4, ls="--")

    ax.axhline(0, color="#444441", lw=0.7, ls="--", alpha=0.4)
    ax.axhline(15, color=DANGER, lw=0.7, ls=":", alpha=0.5)
    ax.set_xlabel("Flight distance (miles)")
    ax.set_ylabel("Average arrival delay (minutes)")
    ax.set_title("Do longer flights delay more?  Distance vs. delay by carrier",
                 pad=14, loc="left")
    ax.legend(fontsize=9, title="Carrier", ncol=2)

    # Bubble size legend
    for size, label in [(50, "~10k flights"), (200, "~40k"), (500, "~100k")]:
        ax.scatter([], [], s=size, color=MUTED, alpha=0.6, label=label)

    marcus_quote(ax, "Short hops punish me more. Good to know before booking that ORD–JFK hop.")
    fig.tight_layout()
    save(fig, "viz4_scatter_distance_delay")


# ══════════════════════════════════════════════════════════
#  VIZ 5 — MONTHLY TREND: Line chart on-time % + delay
#  Module 5: line geom, temporal data, dual-axis
#  Applies primacy/recency effect from module 3 (memorable start/end)
# ══════════════════════════════════════════════════════════
def plot_monthly_trend(monthly: pd.DataFrame):
    print("[VIZ 5] Monthly trend…")

    months = ["Jan","Feb","Mar","Apr","May","Jun",
              "Jul","Aug","Sep","Oct","Nov","Dec"]

    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax2 = ax1.twinx()

    color_delay = DANGER
    color_ot    = SUCCESS

    ax1.fill_between(
        monthly["MONTH"], monthly["avg_delay"],
        alpha=0.15, color=color_delay
    )
    l1, = ax1.plot(
        monthly["MONTH"], monthly["avg_delay"],
        color=color_delay, lw=2, marker="o", ms=5, label="Avg arrival delay (min)"
    )
    l2, = ax2.plot(
        monthly["MONTH"], monthly["on_time_pct"],
        color=color_ot, lw=2, marker="s", ms=5, ls="--", label="On-time % (right axis)"
    )

    # Direct labels on line
    for _, row in monthly.iterrows():
        ax1.text(row["MONTH"], row["avg_delay"]+0.4, f"{row['avg_delay']:.1f}",
                 ha="center", va="bottom", fontsize=7.5, color=color_delay)

    ax1.set_xticks(monthly["MONTH"])
    ax1.set_xticklabels(months)
    ax1.set_ylabel("Avg. arrival delay (min)", color=color_delay)
    ax2.set_ylabel("On-time arrival %", color=color_ot)
    ax1.set_xlabel("")
    ax1.set_title("The system's rhythm — monthly on-time performance across 2023",
                  pad=14, loc="left")
    ax1.tick_params(axis="y", labelcolor=color_delay)
    ax2.tick_params(axis="y", labelcolor=color_ot)

    # Highlight summer (Jun–Aug) congestion
    ax1.axvspan(6, 8, alpha=0.07, color=DANGER, label="Summer peak")
    ax1.text(7, monthly["avg_delay"].max()*0.95, "Summer\npeak",
             ha="center", fontsize=8, color=DANGER, alpha=0.7)

    ax1.legend(handles=[l1, l2], loc="upper left", fontsize=9)
    marcus_quote(ax1, "December is actually not that bad. Note to self: avoid July.")
    fig.tight_layout()
    save(fig, "viz5_monthly_trend")


# ══════════════════════════════════════════════════════════
#  VIZ 6 — FRI BAR CHART: Flight Reliability Index
#  Our original KPI visualized as horizontal bar
#  Applies: ordered bar chart best practice (module 4)
# ══════════════════════════════════════════════════════════
def plot_fri_ranking(carrier_kpi: pd.DataFrame):
    print("[VIZ 6] FRI ranking bar chart…")

    df = carrier_kpi.sort_values("FRI").copy()

    fig, ax = plt.subplots(figsize=(9, 6))

    colors = [SUCCESS if v >= 65 else DANGER if v < 55 else "#BA7517"
              for v in df["FRI"]]

    bars = ax.barh(df["CARRIER_NAME"], df["FRI"], color=colors, height=0.6, zorder=3)

    # Direct value labels (module 5: best practice for labels)
    for bar, val in zip(bars, df["FRI"]):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f"{val:.1f}", va="center", fontsize=10, fontweight="medium")

    ax.axvline(65, color=SUCCESS, lw=1, ls="--", alpha=0.6, label="Good (≥65)")
    ax.axvline(55, color=DANGER,  lw=1, ls=":", alpha=0.6, label="Poor (<55)")
    ax.set_xlim(0, 100)
    ax.set_xlabel("Flight Reliability Index (FRI)  —  0 to 100")
    ax.set_title("Who should Marcus trust?  Flight Reliability Index by carrier",
                 pad=14, loc="left")
    ax.legend(fontsize=9)

    # FRI formula note
    ax.text(0.99, -0.1,
            "FRI = 0.5 × on-time% + 0.3 × (1−norm.delay) + 0.2 × (1−cancel.rate)",
            transform=ax.transAxes, ha="right", fontsize=7.5, color="#888780")

    marcus_quote(ax, "Finally. A single number I can act on before booking.")
    fig.tight_layout()
    save(fig, "viz6_fri_ranking")


# ══════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════
def run_static_viz():
    print("=" * 58)
    print("  STATIC VISUALIZATIONS — loading data")
    print("=" * 58)

    df_clean     = pd.read_parquet(DATA_DIR / "bts_2023_clean.parquet")
    carrier_kpi  = pd.read_parquet(DATA_DIR / "agg_carrier_kpi.parquet")
    heatmap      = pd.read_parquet(DATA_DIR / "agg_heatmap.parquet")
    monthly      = pd.read_parquet(DATA_DIR / "agg_monthly.parquet")
    delay_causes = pd.read_parquet(DATA_DIR / "agg_delay_causes.parquet")

    plot_heatmap_hour_day(heatmap)
    plot_boxplot_carriers(df_clean, carrier_kpi)
    plot_delay_causes(delay_causes)
    plot_scatter_distance_delay(df_clean, carrier_kpi)
    plot_monthly_trend(monthly)
    plot_fri_ranking(carrier_kpi)

    print(f"\n✅ All figures saved to {OUT_DIR}/")


if __name__ == "__main__":
    run_static_viz()
