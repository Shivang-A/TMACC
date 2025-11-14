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
        border-bottom: 3px solid #a8d5ba;
    }
    
    /* Metric boxes */
    .metric-container {
        background: linear-gradient(135deg, #e8f5e9 0%, #f1f8f4 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 5px solid #66bb6a;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #2e7d32;
        margin: 0;
    }
    
    .metric-label {
        font-size: 1rem;
        color: #546e7a;
        margin-top: 0.5rem;
    }
    
    /* Info box */
    .info-box {
        background: #fff8e1;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
        font-size: 0.95rem;
        color: #5d4037;
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
</style>
""",
    unsafe_allow_html=True,
)

# --------------------------------------------------
# 3) PHYSICS CONFIG – TMACC-style CH4 → ΔT
# --------------------------------------------------

TARGET_YEARS = [2030, 2040, 2047]
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

# Pastel color palette
PASTEL_COLORS = {
    "Transport":   "#a8d5e2",  # Light blue
    "Waste":       "#ffd8a8",  # Light orange
    "Industry":    "#a8e6cf",  # Light green
    "Livestock":   "#ffa8a8",  # Light red
    "Agriculture": "#d4a8e4",  # Light purple
    "Residential": "#c9b8a8",  # Light brown
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
        mas_val = sec_df["DeltaT_mK"].sum()
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

    year_choice = st.select_slider(
        "Target Year",
        options=TARGET_YEARS,
        value=TARGET_YEARS[-1],
        help="Select year for detailed analysis"
    )
    
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
    methane mitigation policies. Results show avoided warming in **milli-degrees Celsius (m°C)**, 
    where 1 m°C = 0.001°C.
    """
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
    results["DeltaT_mK"] = results["DeltaT_degC"] * 1000.0

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
        st.markdown(
            f"""
            <div class="metric-container">
                <div class="metric-value">{mas_metrics[year]:.2f}</div>
                <div class="metric-label">m°C avoided by {year}</div>
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
    
    fig_g1 = go.Figure(data=[
        go.Bar(
            x=g1_sectors,
            y=g1_vals,
            marker_color=g1_colors,
            text=[f"{v:.2f}" for v in g1_vals],
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>ΔT: %{y:.3f} m°C<extra></extra>'
        )
    ])
    
    fig_g1.update_layout(
        title=f"Temperature Reduction by Sector ({year_choice})",
        xaxis_title="Sector",
        yaxis_title="ΔT (m°C)",
        height=400,
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(size=11),
        margin=dict(t=50, b=80, l=50, r=20)
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
            hovertemplate=f'<b>{sector_name}</b><br>Year: %{{x}}<br>ΔT: %{{y:.3f}} m°C<extra></extra>'
        ))
    
    # Add total trace
    fig_g6.add_trace(go.Scatter(
        x=TARGET_YEARS,
        y=mas_total_time,
        mode='lines+markers',
        name='Total',
        line=dict(width=3.5, color='#2c3e50', dash='dash'),
        marker=dict(size=10, symbol='diamond'),
        hovertemplate='<b>Total MAS</b><br>Year: %{x}<br>ΔT: %{y:.3f} m°C<extra></extra>'
    ))
    
    fig_g6.update_layout(
        title="MAS Impact Timeline",
        xaxis_title="Year",
        yaxis_title="ΔT (m°C)",
        height=400,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(size=11),
        legend=dict(orientation="v", yanchor="top", y=0.99, xanchor="left", x=0.01),
        margin=dict(t=50, b=50, l=50, r=20)
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
        vertical_spacing=0.12,
        horizontal_spacing=0.08
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
        vals = df_sec_year["DeltaT_mK"].values
        
        colors = [
            get_sector_color(sector_name) if scen in sector_max_alts[sector_name] 
            else "#d0d0d0" 
            for scen in scen_list
        ]
        
        fig_g2.add_trace(
            go.Bar(
                x=scen_list,
                y=vals,
                marker_color=colors,
                showlegend=False,
                hovertemplate='<b>%{x}</b><br>ΔT: %{y:.3f} m°C<extra></extra>'
            ),
            row=row,
            col=col
        )
    
    fig_g2.update_layout(
        height=300 * n_rows,
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
        vertical_spacing=0.12,
        horizontal_spacing=0.08
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
            
            fig_g4.add_trace(
                go.Scatter(
                    x=df_s["TargetYear"],
                    y=df_s["DeltaT_mK"],
                    mode='lines+markers',
                    name=scen,
                    line=dict(width=3 if is_mas else 1.5),
                    opacity=1.0 if is_mas else 0.6,
                    showlegend=(idx == 0),
                    legendgroup=scen,
                    hovertemplate=f'<b>{scen}</b><br>Year: %{{x}}<br>ΔT: %{{y:.3f}} m°C<extra></extra>'
                ),
                row=row,
                col=col
            )
    
    fig_g4.update_layout(
        height=300 * n_rows,
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
        
        #### Data Sources:
        - State-level methane emissions by sector and year
        - Multiple mitigation scenarios (ALT 1-4) compared to BAU
        - Maximum Ambition Scenario (MAS) selects best alternatives per sector
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
