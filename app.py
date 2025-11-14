import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# --------------------------------------------------
# 1) PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="State Methane Temperature Impact – TMACC",
    layout="wide"
)

plt.style.use("seaborn-v0_8-whitegrid")

# --------------------------------------------------
# 2) CUSTOM CSS
# --------------------------------------------------
st.markdown(
    """
<style>
html, body, [class*="css"]  {
    font-size: 16px !important;
    line-height: 1.4 !important;
    color: #111 !important;
}

/* Headings */
.header-style { 
    font-size: 32px !important; 
    font-weight: bold; 
    color: #2F4F4F; 
    margin-bottom: 0.5rem !important;
}
.subheader-style { 
    font-size: 24px !important; 
    color: #2E8B57; 
    border-bottom: 3px solid #2E8B57; 
    margin-top: 1.5rem !important;
    margin-bottom: 1.0rem !important;
}

/* Small metric box */
.metric-box { 
    padding: 12px; 
    background: #F0FFF0; 
    border-radius: 12px; 
    margin: 8px 0; 
    font-size: 15px !important;
    border: 1px solid #d3eecd;
}

/* Sidebar header */
.sidebar-header {
    font-weight: 600;
    font-size: 18px;
    margin-top: 0.5rem;
}

/* Footer */
.footer-text {
    text-align: center; 
    font-size: 12px; 
    color: #888888; 
    margin-top: 40px;
}
</style>
""",
    unsafe_allow_html=True,
)

# --------------------------------------------------
# 3) STATE CONFIG (you can extend this later)
# --------------------------------------------------

DATA_DIR = Path("data")

STATE_CONFIG = {
    "Haryana": {
        "csv": DATA_DIR / "haryana_tmacc_ch4_results.csv",
        "sector_max_alts": {
            "Transport":   ["ALT 4"],          # EV policy for all vehicles
            "Waste":       ["ALT 4"],          # 60% diversion
            "Industry":    ["ALT 2", "ALT 3"], # community boilers + green H2
            "Livestock":   ["ALT 3"],          # Purna Gau Charan Bhumi
            "Agriculture": ["ALT 3"],          # SRI + natural farming + diversification
            "Residential": ["ALT 3"],          # solar cooking beyond 2040
        },
    },
    "Punjab (Coming soon)": {
        "csv": None,
        "sector_max_alts": None,
    },
}

SECTOR_COLORS = {
    "Transport":   "#1f77b4",
    "Waste":       "#ff7f0e",
    "Industry":    "#2ca02c",
    "Livestock":   "#d62728",
    "Agriculture": "#9467bd",
    "Residential": "#8c564b",
}

BAU_SCENARIO_NAME = "BAU"
DEFAULT_TARGET_YEAR = 2047  # main focus


def get_sector_color(sector: str) -> str:
    return SECTOR_COLORS.get(sector, "#333333")


def add_light_grid(ax, axis="both"):
    ax.grid(True, axis=axis, linestyle=":", alpha=0.4)


# --------------------------------------------------
# 4) SIDEBAR – IGSD LOGO + STATE SELECTOR
# --------------------------------------------------

with st.sidebar:
    # Put your IGSD logo file in the app folder and change the name if needed
    st.image("igsd_logo.png", width=140)
    st.markdown('<div class="sidebar-header">State selection</div>', unsafe_allow_html=True)

    state = st.selectbox(
        "Choose state",
        list(STATE_CONFIG.keys()),
        index=0,
    )

    year_choice = st.select_slider(
        "Target year for bar charts",
        options=[2030, 2040, 2047],
        value=DEFAULT_TARGET_YEAR,
    )

# --------------------------------------------------
# 5) PAGE HEADER
# --------------------------------------------------
st.markdown(
    '<p class="header-style">State Methane Mitigation – Temperature Impact Dashboard</p>',
    unsafe_allow_html=True,
)
st.markdown(
    "This dashboard shows estimated **global temperature reduction** from state-level "
    "methane mitigation measures, using a simplified TMACC-style approach. "
    "Values are shown in **milli-degrees Celsius (milli-°C)** where 1 milli-°C = 0.001 °C."
)

# --------------------------------------------------
# 6) DATA LOADING AND PREP
# --------------------------------------------------

config = STATE_CONFIG[state]

if config["csv"] is None:
    st.warning("Data for this state is not yet available. Stay tuned – coming soon!")
else:
    csv_path = config["csv"]
    sector_max_alts = config["sector_max_alts"]

    if not csv_path.exists():
        st.error(
            f"Could not find data file for {state}: `{csv_path}`.\n\n"
            "Please make sure the file exists in the `data/` folder."
        )
        st.stop()

    # Load precomputed TMACC ΔT results
    res = pd.read_csv(csv_path)

    # Convert to milli-°C
    res["DeltaT_mK"] = res["DeltaT_degC"] * 1000.0

    res_total = res[res["Sector"] == "ALL"].copy()
    res_sectors = res[res["Sector"] != "ALL"].copy()

    all_years = sorted(res["TargetYear"].unique())
    all_scen = sorted(res["Scenario"].unique())
    all_sectors = sorted(res_sectors["Sector"].unique())

    # Small sanity info box
    st.markdown(
        f"""
        <div class="metric-box">
        <strong>Loaded state:</strong> {state}  &nbsp; | &nbsp;
        <strong>Years:</strong> {', '.join(str(y) for y in all_years)}  &nbsp; | &nbsp;
        <strong>Scenarios:</strong> {', '.join(all_scen)}
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --------------------------------------------------
    # Helper: MAS contributions by sector
    # --------------------------------------------------
    def compute_mas_sector_contrib(
        df_sectors: pd.DataFrame, year: int, sector_max_alts: dict
    ):
        """
        For each sector, sum DeltaT_mK across selected ALT(s) at given year.
        Returns:
            mas_by_sector: dict {sector: value}
            mas_total: sum of all sectors
        """
        mas_by_sector = {}
        for sector_name, alt_list in sector_max_alts.items():
            sec_df = df_sectors[
                (df_sectors["Sector"] == sector_name)
                & (df_sectors["TargetYear"] == year)
                & (df_sectors["Scenario"].isin(alt_list))
            ]
            mas_val = sec_df["DeltaT_mK"].sum()
            mas_by_sector[sector_name] = mas_val

        mas_total = sum(mas_by_sector.values())
        return mas_by_sector, mas_total

    # --------------------------------------------------
    # 7) G1 – Sector contributions to ΔT (MAS) in chosen year
    # --------------------------------------------------
    st.markdown(
        '<p class="subheader-style">G1 – Sector contributions under Maximum Ambition Scenario</p>',
        unsafe_allow_html=True,
    )
    mas_by_sector_year, mas_total_year = compute_mas_sector_contrib(
        res_sectors, year_choice, sector_max_alts
    )

    g1_sectors = list(sector_max_alts.keys())
    g1_vals = [mas_by_sector_year.get(s, 0.0) for s in g1_sectors]
    g1_colors = [get_sector_color(s) for s in g1_sectors]

    fig_g1, ax1 = plt.subplots(figsize=(7, 4))
    ax1.bar(g1_sectors, g1_vals, color=g1_colors)
    ax1.set_xlabel("Sector")
    ax1.set_ylabel(f"ΔT in {year_choice} (milli-°C)")
    ax1.set_title(f"{state}: Sector contributions to ΔT in {year_choice} (MAS)")
    add_light_grid(ax1, axis="y")
    plt.setp(ax1.get_xticklabels(), rotation=30, ha="right")
    st.pyplot(fig_g1, use_container_width=True)

    st.caption(
        f"In {year_choice}, the Maximum Ambition Scenario for {state} avoids "
        f"about **{mas_total_year:.3f} milli-°C** of global warming, "
        "summed across all sectors."
    )

    # --------------------------------------------------
    # 8) G2 – Within-sector ALT comparison (bar charts)
    # --------------------------------------------------
    st.markdown(
        '<p class="subheader-style">G2 – ALT comparison within each sector (target year)</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f"Each panel shows all ALT scenarios for a given sector in **{year_choice}**. "
        "Bars in color indicate the ALT(s) chosen for the Maximum Ambition Scenario."
    )

    n_sectors = len(sector_max_alts)
    n_cols = 3
    n_rows = int(np.ceil(n_sectors / n_cols))

    fig_g2, axes_g2 = plt.subplots(
        n_rows, n_cols, figsize=(14, 4 * n_rows), sharey=True
    )
    axes_g2 = np.array(axes_g2).reshape(-1)

    for ax, sector_name in zip(axes_g2, sector_max_alts.keys()):
        df_sec_year = res_sectors[
            (res_sectors["Sector"] == sector_name)
            & (res_sectors["TargetYear"] == year_choice)
            & (res_sectors["Scenario"] != BAU_SCENARIO_NAME)
        ].copy()

        df_sec_year = df_sec_year.sort_values("Scenario")
        scen_list = df_sec_year["Scenario"].tolist()
        vals = df_sec_year["DeltaT_mK"].values

        colors = []
        for scen in scen_list:
            if scen in sector_max_alts[sector_name]:
                colors.append(get_sector_color(sector_name))
            else:
                colors.append("#BBBBBB")

        ax.bar(scen_list, vals, color=colors)
        ax.set_title(sector_name, fontsize=11)
        ax.set_xlabel("ALT scenario", fontsize=10)
        add_light_grid(ax, axis="y")
        if ax is axes_g2[0]:
            ax.set_ylabel(f"ΔT in {year_choice} (milli-°C)")
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    # hide any leftover axes
    for i in range(len(sector_max_alts), len(axes_g2)):
        fig_g2.delaxes(axes_g2[i])

    plt.tight_layout()
    st.pyplot(fig_g2, use_container_width=True)

    # --------------------------------------------------
    # 9) G4 – Within-sector ΔT evolution over time (lines)
    # --------------------------------------------------
    st.markdown(
        '<p class="subheader-style">G4 – Evolution of ΔT over time within each sector</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        "Each panel shows how temperature impact evolves from **2019 → 2030 → 2040 → 2047** "
        "for each ALT in that sector. The line(s) forming part of the MAS are drawn slightly bolder."
    )

    fig_g4, axes_g4 = plt.subplots(
        n_rows, n_cols, figsize=(14, 4 * n_rows), sharey=True
    )
    axes_g4 = np.array(axes_g4).reshape(-1)

    for ax, sector_name in zip(axes_g4, sector_max_alts.keys()):
        df_sec = res_sectors[res_sectors["Sector"] == sector_name].copy()

        for scen, df_s in df_sec.groupby("Scenario"):
            if scen == BAU_SCENARIO_NAME:
                continue
            df_s = df_s.sort_values("TargetYear")
            ax.plot(
                df_s["TargetYear"],
                df_s["DeltaT_mK"],
                marker="o",
                linewidth=2.2 if scen in sector_max_alts[sector_name] else 1.2,
                label=scen,
                alpha=0.95 if scen in sector_max_alts[sector_name] else 0.75,
            )

        ax.set_title(sector_name, fontsize=11)
        ax.set_xlabel("Year", fontsize=10)
        add_light_grid(ax, axis="both")
        if ax is axes_g4[0]:
            ax.set_ylabel("ΔT (milli-°C)", fontsize=11)
        ax.legend(fontsize=8)

    for i in range(len(sector_max_alts), len(axes_g4)):
        fig_g4.delaxes(axes_g4[i])

    plt.tight_layout()
    st.pyplot(fig_g4, use_container_width=True)

    # --------------------------------------------------
    # 10) G6 – MAS sector contributions over time
    # --------------------------------------------------
    st.markdown(
        '<p class="subheader-style">G6 – MAS sector contributions over time</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        "This plot shows how each sector contributes to temperature reduction over time "
        "under the Maximum Ambition Scenario, together with the total MAS impact."
    )

    mas_by_sector_time = {sector_name: [] for sector_name in sector_max_alts.keys()}
    mas_total_time = []

    for year in sorted(all_years):
        by_sector, total_val = compute_mas_sector_contrib(
            res_sectors, year, sector_max_alts
        )
        for sector_name in sector_max_alts.keys():
            mas_by_sector_time[sector_name].append(by_sector.get(sector_name, 0.0))
        mas_total_time.append(total_val)

    years_sorted = sorted(all_years)

    fig_g6, ax6 = plt.subplots(figsize=(8, 5))

    for sector_name, series in mas_by_sector_time.items():
        ax6.plot(
            years_sorted,
            series,
            marker="o",
            linewidth=1.8,
            label=sector_name,
            color=get_sector_color(sector_name),
        )

    # Bold total line
    ax6.plot(
        years_sorted,
        mas_total_time,
        marker="o",
        linewidth=2.8,
        color="black",
        label="Total MAS (all sectors)",
    )

    ax6.set_xlabel("Year", fontsize=11)
    ax6.set_ylabel("ΔT (milli-°C)", fontsize=11)
    ax6.set_title(f"{state}: MAS sector contributions to ΔT over time")
    add_light_grid(ax6, axis="both")
    ax6.legend(fontsize=9, ncol=2)
    plt.tight_layout()
    st.pyplot(fig_g6, use_container_width=True)

# --------------------------------------------------
# 11) FOOTER
# --------------------------------------------------
st.markdown(
    """
<hr style="border: none; height: 1px; background-color: #ccc; margin-top: 40px; margin-bottom: 10px;" />
<div class="footer-text">
    © 2025 Institute for Governance & Sustainable Development (IGSD) – TMACC-inspired analysis
</div>
""",
    unsafe_allow_html=True,
)