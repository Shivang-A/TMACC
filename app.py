# Requirements: streamlit, pandas, numpy, plotly
# Add to requirements.txt: plotly>=5.0.0

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path

# --------------------------------------------------
# 1) PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="State Methane Temperature Impact – TMACC",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# 2) CUSTOM CSS
# --------------------------------------------------
st.markdown(
    """
<style>
    /* Main container */
    .main {
        padding: 2rem 3rem;
    }
    
    /* Headers */
    h1 {
        color: #2c3e50;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    
    h2 {
        color: #34495e;
        font-weight: 600;
        margin-top: 2rem;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #7eb899;
    }
    
    /* Metric boxes */
    .metric-container {
        background: linear-gradient(135deg, #d4edda 0%, #e8f5e9 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 5px solid #4caf50;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1b5e20;
        margin: 0;
    }
    
    .metric-label {
        font-size: 1rem;
        color: #424242;
        margin-top: 0.5rem;
    }
    
    /* Info box */
    .info-box {
        background: #fff3cd;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
        font-size: 0.95rem;
        color: #5d4037;
    }
    
    .unit-explanation {
        background: #e3f2fd;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #2196f3;
        margin: 1rem 0;
        font-size: 0.9rem;
        color: #01579b;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: #f8f9fa;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        font-size: 0.85rem;
        color: #7f8c8d;
        margin-top: 4rem;
        padding-top: 2rem;
        border-top: 1px solid #ecf0f1;
    }
    
    /* Remove extra padding */
    .block-container {
        padding-top: 2rem;
    }
    
    /* Button styling */
    .stButton > button {
        width: 100%;
    }
</style>
""",
    unsafe_allow_html=True,
)

# --------------------------------------------------
# 3) PHYSICS CONFIG – TMACC-style CH4 → ΔT
# --------------------------------------------------

TARGET_YEARS = [2040, 2047]  # Only two meaningful years
BAU_SCENARIO_NAME = "BAU"

# AGTP kernel for CH4 (degC per kg emitted) at different lags
AGTP_CH4_POINTS = {
    1:   1.75e-15,
    5:   1.51e-15,
    10:  0.98e-15,
    20:  0.45e-15,
    50:  0.15e-15,
    100: 0.04e-15,
}

# Indirect effects: ozone, stratospheric H2O, CO2 from oxidation
INDIRECT_FACTOR_CH4 = 1.75

# Climate inertia parameters
INCLUDE_CLIMATE_INERTIA = True
TAU_CLIMATE = 10.0  # years

# Darker pastel color palette
PASTEL_COLORS = {
    "Transport":   "#7fb3d5",  # Darker light blue
    "Waste":       "#f4b183",  # Darker light orange
    "Industry":    "#77c4a1",  # Darker light green
    "Livestock":   "#e57373",  # Darker light red
    "Agriculture": "#ba68c8",  # Darker light purple
    "Residential": "#a1887f",  # Darker light brown
}

def interp_agtp_ch4(lag_years: float) -> float:
    """
    Piecewise-linear interpolation of AGTP_CH4_POINTS for any lag >= 0.
    Returns degC per kg CH4 emitted.
    """
    if lag_years <= 0:
        return AGTP_CH4_POINTS[1]
    xs = sorted(AGTP_CH4_POINTS.keys())
    if lag_years >= xs[-1]:
        return AGTP_CH4_POINTS[xs[-1]]
    for i in range(len(xs) - 1):
        x0, x1 = xs[i], xs[i + 1]
        if x0 <= lag_years <= x1:
            y0, y1 = AGTP_CH4_POINTS[x0], AGTP_CH4_POINTS[x1]
            w = (lag_years - x0) / (x1 - x0)
            return y0 * (1 - w) + y1 * w
    return AGTP_CH4_POINTS[1]

def climate_inertia_factor(lag_years: float, tau: float) -> float:
    """
    Climate response factor: 1 - exp(-lag / tau).
    Represents the gradual response of the climate system.
    """
    if lag_years <= 0:
        return 0.0
    return 1.0 - np.exp(-lag_years / tau)

# --------------------------------------------------
# 4) STATE CONFIG
# --------------------------------------------------

DATA_DIR = Path("data")

STATE_CONFIG = {
    "Haryana": {
        "csv": DATA_DIR / "Haryana_Emissions.csv",
        "sector_max_alts": {
            "Transport":   ["ALT 4"],
            "Waste":       ["ALT 4"],
            "Industry":    ["ALT 2", "ALT 3"],
            "Livestock":   ["ALT 3"],
            "Agriculture": ["ALT 3"],
            "Residential": ["ALT 3"],
        },
    },
    "Punjab (Coming soon)": {
        "csv": None,
        "sector_max_alts": None,
    },
}

def get_sector_color(sector: str) -> str:
    return PASTEL_COLORS.get(sector, "#b0b0b0")

# --------------------------------------------------
# 5) DATA LOADING & PROCESSING
# --------------------------------------------------

@st.cache_data
def load_state_emissions(path: Path) -> pd.DataFrame:
    """Load and normalize raw emissions data."""
    df = pd.read_csv(path)

    # Normalize column names
    col_rename = {}
    if "Scenario Type" in df.columns:
        col_rename["Scenario Type"] = "Scenario"
    if "CH4 (kT/Year)" in df.columns:
        col_rename["CH4 (kT/Year)"] = "CH4_kT"
    if "CH4_kT" not in df.columns and "CH4" in df.columns:
        col_rename["CH4"] = "CH4_kT"

    df = df.rename(columns=col_rename)

    required = ["Sector", "Year", "Scenario", "Mitigation Strategy", "CH4_kT"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"Missing required columns: {', '.join(missing)}")
        st.stop()

    df["Year"] = df["Year"].astype(int)
    return df[required].copy()

@st.cache_data
def build_deltaE(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate emission reductions (ΔE) for all ALT scenarios vs BAU.
    ΔE = BAU emissions - ALT emissions (positive = reduction)
    """
    bau = df[df["Scenario"] == BAU_SCENARIO_NAME]
    alts = df[df["Scenario"] != BAU_SCENARIO_NAME]

    merged = alts.merge(
        bau[["Sector", "Year", "CH4_kT"]],
        on=["Sector", "Year"],
        how="left",
        suffixes=("_ALT", "_BAU"),
    )

    merged["delta_CH4_kT"] = merged["CH4_kT_BAU"] - merged["CH4_kT_ALT"]
    
    return merged[["Sector", "Year", "Scenario", "Mitigation Strategy", "delta_CH4_kT"]].copy()

@st.cache_data
def compute_deltaT_for_year(deltaE_df: pd.DataFrame, target_year: int) -> pd.DataFrame:
    """
    Compute temperature impact (ΔT) for a target year using AGTP approach.
    
    For each emission reduction ΔE(t) at year t:
    ΔT(target_year) = Σ_t [ΔE(t) × AGTP(lag) × indirect_factor × inertia_factor]
    
    where lag = target_year - t
    """
    rows = []

    for scenario, df_scen in deltaE_df.groupby("Scenario"):
        for sector, df_sec in df_scen.groupby("Sector"):
            dT = 0.0

            for _, r in df_sec.iterrows():
                year = int(r["Year"])
                if year > target_year:
                    continue

                lag = target_year - year
                
                # AGTP kernel for this lag
                kernel = interp_agtp_ch4(lag)
                
                # Apply indirect effects and climate inertia
                factor = INDIRECT_FACTOR_CH4
                if INCLUDE_CLIMATE_INERTIA:
                    factor *= climate_inertia_factor(lag, TAU_CLIMATE)

                # Convert kT to kg and calculate temperature impact
                delta_kg = float(r["delta_CH4_kT"]) * 1e6
                dT += delta_kg * kernel * factor

            rows.append({
                "Scenario": scenario,
                "Sector": sector,
                "TargetYear": target_year,
                "DeltaT_degC": dT,
            })

    result = pd.DataFrame(rows)

    # Add total across all sectors
    totals = (
        result.groupby(["Scenario", "TargetYear"])
        .agg(DeltaT_degC=("DeltaT_degC", "sum"))
        .reset_index()
    )
    totals["Sector"] = "ALL"

    return pd.concat([result, totals], ignore_index=True)

def compute_mas_sector_contrib(df_sectors: pd.DataFrame, year: int, sector_max_alts: dict):
    """Calculate Maximum Ambition Scenario contributions by sector."""
    mas_by_sector = {}
    for sector_name, alt_list in sector_max_alts.items():
        sec_df = df_sectors[
            (df_sectors["Sector"] == sector_name)
            & (df_sectors["TargetYear"] == year)
            & (df_sectors["Scenario"].isin(alt_list))
        ]
        mas_val = sec_df["DeltaT_mC"].sum()
        mas_by_sector[sector_name] = mas_val
    
    mas_total = sum(mas_by_sector.values())
    return mas_by_sector, mas_total

# --------------------------------------------------
# 6) SIDEBAR
# --------------------------------------------------

with st.sidebar:
    st.image("igsd_logo.png", width=160)
    
    st.markdown("### 🌍 Configuration")
    
    state = st.selectbox(
        "Select State",
        list(STATE_CONFIG.keys()),
        index=0,
        help="Choose the state to analyze"
    )

    st.markdown("**Target Year for Analysis:**")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📅 2040", use_container_width=True, type="primary" if 'year_choice' not in st.session_state or st.session_state.year_choice == 2040 else "secondary"):
            st.session_state.year_choice = 2040
    with col2:
        if st.button("📅 2047", use_container_width=True, type="primary" if 'year_choice' in st.session_state and st.session_state.year_choice == 2047 else "secondary"):
            st.session_state.year_choice = 2047
    
    # Initialize default year if not set
    if 'year_choice' not in st.session_state:
        st.session_state.year_choice = 2047
    
    year_choice = st.session_state.year_choice
    
    st.markdown("---")
    st.markdown("### 📊 About")
    st.markdown(
        """
        This dashboard calculates **global temperature reduction** 
        from state-level methane mitigation using the AGTP (Absolute 
        Global Temperature change Potential) approach.
        
        **Key Concepts:**
        - **BAU**: Business As Usual scenario
        - **ALT**: Alternative mitigation scenarios
        - **MAS**: Maximum Ambition Scenario
        - **ΔT**: Temperature change avoided
        """
    )

# --------------------------------------------------
# 7) MAIN CONTENT
# --------------------------------------------------

st.title("🌡️ State Methane Mitigation – Temperature Impact Dashboard")

st.markdown(
    """
    This dashboard quantifies the **global temperature reduction** achieved through state-level 
    methane mitigation policies. Results show avoided warming in **milli-degrees Celsius (m°C)**.
    """
)

# Unit explanation box
st.markdown(
    """
    <div class="unit-explanation">
        <strong>📏 Understanding Temperature Units</strong><br>
        <strong>m°C (milli-degrees Celsius)</strong> = 10<sup>-3</sup> °C = 0.001 °C<br>
        For example: <strong>120.50 m°C = 0.12050 °C</strong><br>
        While these values may seem small, they represent significant global climate impacts when considering 
        the cumulative effect of methane from multiple sources worldwide.
    </div>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------
# 8) LOAD & PROCESS DATA
# --------------------------------------------------

config = STATE_CONFIG[state]

if config["csv"] is None:
    st.info("📦 Data for this state is coming soon. Stay tuned!")
    st.stop()

csv_path = config["csv"]
sector_max_alts = config["sector_max_alts"]

if not csv_path.exists():
    st.error(f"❌ Could not find emissions file: `{csv_path}`")
    st.stop()

# Load and process data
with st.spinner("Loading and processing emissions data..."):
    df_emis = load_state_emissions(csv_path)
    deltaE = build_deltaE(df_emis)
    
    # Compute ΔT for all target years
    all_results = []
    for y in TARGET_YEARS:
        res_y = compute_deltaT_for_year(deltaE, y)
        all_results.append(res_y)
    
    results = pd.concat(all_results, ignore_index=True)
    results["DeltaT_mC"] = results["DeltaT_degC"] * 1000.0

# Separate total and sectoral results
res_total = results[results["Sector"] == "ALL"].copy()
res_sectors = results[results["Sector"] != "ALL"].copy()

# --------------------------------------------------
# 9) KEY METRICS
# --------------------------------------------------

st.markdown("## 📈 Key Metrics")

# Calculate MAS totals for all years
mas_metrics = {}
for year in TARGET_YEARS:
    _, total = compute_mas_sector_contrib(res_sectors, year, sector_max_alts)
    mas_metrics[year] = total

# Display metrics in columns
cols = st.columns(len(TARGET_YEARS))
for col, year in zip(cols, TARGET_YEARS):
    with col:
        value_mC = mas_metrics[year]
        value_C = value_mC / 1000
        
        # Format the display value intelligently
        if value_mC >= 10.0:
            display_value = f"{value_mC:.2f}"
            unit_text = "m°C"
        elif value_mC >= 1.0:
            display_value = f"{value_mC:.3f}"
            unit_text = "m°C"
        elif value_mC >= 0.001:
            # Show in microC for very small values
            display_value = f"{value_mC * 1000:.2f}"
            unit_text = "μ°C"
        else:
            # Use scientific notation for extremely small values
            display_value = f"{value_mC:.2e}"
            unit_text = "m°C"
        
        st.markdown(
            f"""
            <div class="metric-container">
                <div class="metric-value">{display_value} <span style="font-size: 1.5rem;">{unit_text}</span></div>
                <div class="metric-label">Temperature avoided by {year}</div>
                <div style="font-size: 0.85rem; color: #616161; margin-top: 0.5rem;">
                    ({value_C:.6f} °C or {value_C*1e6:.2f} μ°C)
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

# Info box
st.markdown(
    f"""
    <div class="info-box">
        <strong>ℹ️ Maximum Ambition Scenario (MAS)</strong><br>
        The MAS represents the highest ambition mitigation pathway, combining the most effective 
        alternative scenarios across all sectors in {state}.
    </div>
    """,
    unsafe_allow_html=True
)

# --------------------------------------------------
# 10) MAIN VISUALIZATIONS
# --------------------------------------------------

st.markdown("## 📊 Temperature Impact Analysis")

# Create two columns for G1 and G6
col1, col2 = st.columns(2)

with col1:
    st.markdown("### Sector Contributions (MAS)")
    
    # G1: Sector contributions in selected year
    mas_by_sector_year, mas_total_year = compute_mas_sector_contrib(
        res_sectors, year_choice, sector_max_alts
    )
    
    g1_sectors = list(sector_max_alts.keys())
    g1_vals = [mas_by_sector_year.get(s, 0.0) for s in g1_sectors]
    g1_colors = [get_sector_color(s) for s in g1_sectors]
    
    # Format labels: only show for values > 1.0 m°C
    text_labels = []
    for v in g1_vals:
        if v >= 1.0:
            text_labels.append(f"{v:.2f}")
        elif v >= 0.01:
            text_labels.append(f"{v:.3f}")
        else:
            text_labels.append("")  # Don't show label for very small values
    
    fig_g1 = go.Figure(data=[
        go.Bar(
            x=g1_sectors,
            y=g1_vals,
            marker_color=g1_colors,
            text=text_labels,
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>ΔT: %{y:.3f} m°C<br>(%{customdata:.6f} °C)<extra></extra>',
            customdata=[v/1000 for v in g1_vals]
        )
    ])
    
    fig_g1.update_layout(
        title=f"Temperature Reduction by Sector ({year_choice})",
        xaxis_title="Sector",
        yaxis_title="ΔT (m°C)",
        height=450,
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(size=11),
        margin=dict(t=60, b=100, l=60, r=30)
    )
    
    fig_g1.update_xaxes(tickangle=-45, gridcolor='lightgray', gridwidth=0.5)
    fig_g1.update_yaxes(gridcolor='lightgray', gridwidth=0.5)
    
    st.plotly_chart(fig_g1, use_container_width=True)

with col2:
    st.markdown("### Evolution Over Time (MAS)")
    
    # G6: MAS evolution over time
    mas_by_sector_time = {s: [] for s in sector_max_alts.keys()}
    mas_total_time = []
    
    for year in TARGET_YEARS:
        by_sector, total_val = compute_mas_sector_contrib(res_sectors, year, sector_max_alts)
        for sector_name in sector_max_alts.keys():
            mas_by_sector_time[sector_name].append(by_sector.get(sector_name, 0.0))
        mas_total_time.append(total_val)
    
    fig_g6 = go.Figure()
    
    # Add sector traces
    for sector_name, series in mas_by_sector_time.items():
        fig_g6.add_trace(go.Scatter(
            x=TARGET_YEARS,
            y=series,
            mode='lines+markers',
            name=sector_name,
            line=dict(width=2.5, color=get_sector_color(sector_name)),
            marker=dict(size=8),
            hovertemplate=f'<b>{sector_name}</b><br>Year: %{{x}}<br>ΔT: %{{y:.3f}} m°C<br>(%{{customdata:.6f}} °C)<extra></extra>',
            customdata=[v/1000 for v in series]
        ))
    
    # Add total trace
    fig_g6.add_trace(go.Scatter(
        x=TARGET_YEARS,
        y=mas_total_time,
        mode='lines+markers',
        name='Total',
        line=dict(width=3.5, color='#1a1a1a', dash='dash'),
        marker=dict(size=10, symbol='diamond'),
        hovertemplate='<b>Total MAS</b><br>Year: %{x}<br>ΔT: %{y:.3f} m°C<br>(%{customdata:.6f} °C)<extra></extra>',
        customdata=[v/1000 for v in mas_total_time]
    ))
    
    fig_g6.update_layout(
        title="MAS Impact Timeline",
        xaxis_title="Year",
        yaxis_title="ΔT (m°C)",
        height=450,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(size=11),
        legend=dict(orientation="v", yanchor="top", y=0.99, xanchor="left", x=0.01, bgcolor='rgba(255,255,255,0.8)'),
        margin=dict(t=60, b=60, l=60, r=30)
    )
    
    # Fix x-axis to show years as integers
    fig_g6.update_xaxes(
        tickmode='array',
        tickvals=TARGET_YEARS,
        ticktext=[str(y) for y in TARGET_YEARS],
        gridcolor='lightgray',
        gridwidth=0.5
    )
    fig_g6.update_yaxes(gridcolor='lightgray', gridwidth=0.5)
    
    st.plotly_chart(fig_g6, use_container_width=True)

# Add more vertical space between rows
st.markdown("<br><br><br>", unsafe_allow_html=True)

# --------------------------------------------------
# 11) DETAILED SECTOR ANALYSIS
# --------------------------------------------------

st.markdown("## 🔍 Detailed Sector Analysis")

# G2 & G4 in tabs
tab1, tab2 = st.tabs(["📊 Scenario Comparison", "📈 Time Evolution"])

with tab1:
    st.markdown(f"### Alternative Scenarios by Sector ({year_choice})")
    st.caption(
        f"Colored bars indicate scenarios included in the Maximum Ambition Scenario. "
        f"Gray bars show alternative options not selected."
    )
    
    # Create subplots for G2
    n_sectors = len(sector_max_alts)
    n_cols = 3
    n_rows = int(np.ceil(n_sectors / n_cols))
    
    fig_g2 = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=list(sector_max_alts.keys()),
        vertical_spacing=0.18,
        horizontal_spacing=0.10
    )
    
    for idx, sector_name in enumerate(sector_max_alts.keys()):
        row = idx // n_cols + 1
        col = idx % n_cols + 1
        
        df_sec_year = res_sectors[
            (res_sectors["Sector"] == sector_name)
            & (res_sectors["TargetYear"] == year_choice)
            & (res_sectors["Scenario"] != BAU_SCENARIO_NAME)
        ].copy()
        
        df_sec_year = df_sec_year.sort_values("Scenario")
        scen_list = df_sec_year["Scenario"].tolist()
        vals = df_sec_year["DeltaT_mC"].values
        
        colors = [
            get_sector_color(sector_name) if scen in sector_max_alts[sector_name] 
            else "#c0c0c0" 
            for scen in scen_list
        ]
        
        customdata = [[v/1000] for v in vals]
        
        # Smart text labels: only show for meaningful values
        text_labels = []
        for v in vals:
            if v >= 1.0:
                text_labels.append(f"{v:.2f}")
            elif v >= 0.01:
                text_labels.append(f"{v:.3f}")
            else:
                text_labels.append("")  # Hide very small values
        
        fig_g2.add_trace(
            go.Bar(
                x=scen_list,
                y=vals,
                marker_color=colors,
                showlegend=False,
                text=text_labels,
                textposition='outside',
                hovertemplate='<b>%{x}</b><br>ΔT: %{y:.3f} m°C<br>(%{customdata[0]:.6f} °C)<extra></extra>',
                customdata=customdata
            ),
            row=row,
            col=col
        )
    
    fig_g2.update_layout(
        height=350 * n_rows,
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(size=10)
    )
    
    fig_g2.update_xaxes(tickangle=-45, gridcolor='lightgray', gridwidth=0.3)
    fig_g2.update_yaxes(title_text="ΔT (m°C)", gridcolor='lightgray', gridwidth=0.3)
    
    st.plotly_chart(fig_g2, use_container_width=True)

with tab2:
    st.markdown("### Temperature Impact Evolution by Sector")
    st.caption(
        "Bolder lines indicate scenarios included in the Maximum Ambition Scenario."
    )
    
    # Create subplots for G4
    fig_g4 = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=list(sector_max_alts.keys()),
        vertical_spacing=0.18,
        horizontal_spacing=0.10
    )
    
    for idx, sector_name in enumerate(sector_max_alts.keys()):
        row = idx // n_cols + 1
        col = idx % n_cols + 1
        
        df_sec = res_sectors[res_sectors["Sector"] == sector_name].copy()
        
        for scen, df_s in df_sec.groupby("Scenario"):
            if scen == BAU_SCENARIO_NAME:
                continue
            
            df_s = df_s.sort_values("TargetYear")
            is_mas = scen in sector_max_alts[sector_name]
            
            customdata = [[v/1000] for v in df_s["DeltaT_mC"].values]
            
            fig_g4.add_trace(
                go.Scatter(
                    x=df_s["TargetYear"],
                    y=df_s["DeltaT_mC"],
                    mode='lines+markers',
                    name=scen,
                    line=dict(width=3 if is_mas else 1.5),
                    opacity=1.0 if is_mas else 0.6,
                    showlegend=(idx == 0),
                    legendgroup=scen,
                    hovertemplate=f'<b>{scen}</b><br>Year: %{{x}}<br>ΔT: %{{y:.3f}} m°C<br>(%{{customdata[0]:.6f}} °C)<extra></extra>',
                    customdata=customdata
                ),
                row=row,
                col=col
            )
    
    fig_g4.update_layout(
        height=350 * n_rows,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(size=10),
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5)
    )
    
    # Fix x-axis to show years as integers
    fig_g4.update_xaxes(
        tickmode='array',
        tickvals=TARGET_YEARS,
        ticktext=[str(y) for y in TARGET_YEARS],
        gridcolor='lightgray',
        gridwidth=0.3
    )
    fig_g4.update_yaxes(title_text="ΔT (m°C)", gridcolor='lightgray', gridwidth=0.3)
    
    st.plotly_chart(fig_g4, use_container_width=True)

# --------------------------------------------------
# 12) METHODOLOGY
# --------------------------------------------------

with st.expander("🔬 Methodology & Calculations"):
    st.markdown(
        """
        ### Temperature Calculation Methodology
        
        This dashboard uses the **AGTP (Absolute Global Temperature change Potential)** approach 
        to calculate avoided global warming from methane emission reductions.
        
        #### Key Steps:
        
        1. **Emission Reduction (ΔE)**: Calculate the difference between BAU and mitigation scenarios
           ```
           ΔE(t) = BAU_emissions(t) - ALT_emissions(t)
           ```
        
        2. **Temperature Impact (ΔT)**: For each target year, sum the contributions from all past reductions
           ```
           ΔT(target) = Σ [ΔE(t) × AGTP(lag) × indirect_factor × inertia_factor]
           where lag = target_year - emission_year
           ```
        
        3. **AGTP Kernel**: Time-dependent temperature response per unit emission
           - Based on atmospheric lifetime and radiative forcing
           - Interpolated from scientific literature values
        
        4. **Indirect Effects**: Multiplier accounting for:
           - Ozone formation
           - Stratospheric water vapor
           - CO₂ from methane oxidation
           - Factor: 1.75×
        
        5. **Climate Inertia**: Gradual climate system response
           ```
           inertia_factor = 1 - exp(-lag / τ)
           where τ = 10 years
           ```
        
        #### Unit Explanation:
        
        **Temperature values are reported in milli-degrees Celsius (m°C):**
        - 1 m°C = 10⁻³ °C = 0.001 °C
        - 1000 m°C = 1 °C
        
        **Examples:**
        - 120.50 m°C = 0.12050 °C = 1.205 × 10⁻¹ °C
        - 5.25 m°C = 0.00525 °C = 5.25 × 10⁻³ °C
        - 0.100 m°C = 0.0001 °C = 1.0 × 10⁻⁴ °C
        
        While individual state contributions may appear small, they represent significant impacts 
        when aggregated globally across all methane sources.
        
        #### Data Sources:
        - State-level methane emissions by sector and year
        - Multiple mitigation scenarios (ALT 1-4) compared to BAU
        - Maximum Ambition Scenario (MAS) selects best alternatives per sector
        
        #### References:
        - AGTP values based on climate science literature
        - Indirect effects factor includes ozone, stratospheric H₂O, and CO₂ feedback
        - Climate inertia represents ocean heat uptake and system response delays
        """
    )

# --------------------------------------------------
# 13) FOOTER
# --------------------------------------------------

st.markdown(
    """
    <div class="footer">
        © 2025 Institute for Governance & Sustainable Development (IGSD) | 
        TMACC-inspired Temperature Impact Analysis<br>
        <em>Dashboard powered by Streamlit & Plotly</em>
    </div>
    """,
    unsafe_allow_html=True
)
