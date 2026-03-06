"""
Phase 8 — Streamlit frontend for AgroOpt.

Usage
-----
With the FastAPI backend already running on port 8000:

    streamlit run src/frontend/app.py

Then open http://localhost:8501 in your browser.

Three tabs
----------
Predict    — predict yield for a single (conditions, crop) pair
Recommend  — rank all four crops by predicted yield with a bar chart
Optimize   — grid-search management inputs for a target crop
"""

from __future__ import annotations

import requests
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="AgroOpt — Crop Yield Intelligence",
    page_icon="assets/favicon.png" if False else "🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Constants — must match Pydantic schema exactly
# ---------------------------------------------------------------------------

REGIONS = ["East", "North", "South", "West"]
SOIL_TYPES = ["Chalky", "Clay", "Loam", "Peaty", "Sandy", "Silt"]
WEATHER_CONDITIONS = ["Cloudy", "Rainy", "Sunny"]
CROPS = ["Maize", "Rice", "Soybean", "Wheat"]

# FAO 2013 USA benchmarks (hg/ha) — for contextual display
FAO_BENCHMARKS: dict[str, float] = {
    "Maize": 99_689.0,
    "Rice": 83_159.0,
    "Soybean": 28_489.0,
    "Wheat": 31_060.0,
}

CROP_COLORS: dict[str, str] = {
    "Maize": "#f6c90e",
    "Rice": "#4caf50",
    "Soybean": "#9c7b4f",
    "Wheat": "#e8a838",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _api_url(path: str) -> str:
    base = st.session_state.get("api_base_url", "http://localhost:8000").rstrip("/")
    return f"{base}{path}"


def _post(path: str, payload: dict) -> dict | None:
    """POST *payload* to *path*; return JSON dict or None on error."""
    try:
        resp = requests.post(_api_url(path), json=payload, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        st.error(
            "Cannot reach the API. Make sure the FastAPI backend is running:\n\n"
            "```\npython -m src.api.main\n```"
        )
    except requests.exceptions.HTTPError as exc:
        detail = exc.response.json().get("detail", str(exc))
        st.error(f"API error {exc.response.status_code}: {detail}")
    except Exception as exc:  # noqa: BLE001
        st.error(f"Unexpected error: {exc}")
    return None


def _build_conditions_payload(
    rainfall_mm: float,
    temperature_celsius: float,
    days_to_harvest: int,
    region: str,
    soil_type: str,
    weather_condition: str,
    fertilizer_used: bool,
    irrigation_used: bool,
) -> dict:
    return {
        "rainfall_mm": rainfall_mm,
        "temperature_celsius": temperature_celsius,
        "days_to_harvest": days_to_harvest,
        "region": region,
        "soil_type": soil_type,
        "weather_condition": weather_condition,
        "fertilizer_used": fertilizer_used,
        "irrigation_used": irrigation_used,
    }


def _check_health() -> bool:
    try:
        resp = requests.get(_api_url("/health"), timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            st.sidebar.success(
                f"API online — model: **{data['model']}**, "
                f"{data['n_features']} features"
            )
            return True
        st.sidebar.error(f"API returned status {resp.status_code}")
    except requests.exceptions.ConnectionError:
        st.sidebar.error("API offline — start the backend first.")
    except Exception as exc:  # noqa: BLE001
        st.sidebar.error(f"Health check failed: {exc}")
    return False


# ---------------------------------------------------------------------------
# Sidebar — configuration & farm conditions
# ---------------------------------------------------------------------------

st.sidebar.title("AgroOpt")
st.sidebar.caption("Crop Yield Intelligence")

# API URL
if "api_base_url" not in st.session_state:
    st.session_state["api_base_url"] = "http://localhost:8000"

st.sidebar.text_input(
    "API base URL",
    value=st.session_state["api_base_url"],
    key="api_base_url",
    help="Change if running the backend on a different host or port.",
)

if st.sidebar.button("Check API health", use_container_width=True):
    _check_health()

st.sidebar.divider()
st.sidebar.subheader("Farm Conditions")
st.sidebar.caption(
    "Enter observed conditions for a single growing season. "
    "Pre-filled values are representative USA farm examples."
)

# --- Continuous inputs ---
rainfall_mm = st.sidebar.slider(
    "Rainfall (mm)",
    min_value=0,
    max_value=5000,
    value=650,
    step=10,
    help="Total seasonal rainfall in millimetres. Schema range: 0–5 000. "
         "Example: 650 mm (typical US Midwest maize season).",
)

temperature_celsius = st.sidebar.slider(
    "Mean Temperature (°C)",
    min_value=-10,
    max_value=50,
    value=22,
    step=1,
    help="Mean growing-season temperature. Schema range: −10 to 50 °C. "
         "Example: 22 °C (warm temperate climate).",
)

days_to_harvest = st.sidebar.slider(
    "Days to Harvest",
    min_value=1,
    max_value=365,
    value=120,
    step=1,
    help="Length of the growing season in days. Schema range: 1–365. "
         "Examples: Maize ~120 d, Rice ~130 d, Wheat ~240 d, Soybean ~100 d.",
)

# --- Categorical inputs ---
region = st.sidebar.selectbox(
    "Region",
    options=REGIONS,
    index=REGIONS.index("East"),
    help="Geographic region. Allowed values: East, North, South, West.",
)

soil_type = st.sidebar.selectbox(
    "Soil Type",
    options=SOIL_TYPES,
    index=SOIL_TYPES.index("Loam"),
    help="Soil classification. Options: Chalky, Clay, Loam, Peaty, Sandy, Silt. "
         "Loam is generally the most productive.",
)

weather_condition = st.sidebar.selectbox(
    "Weather Condition",
    options=WEATHER_CONDITIONS,
    index=WEATHER_CONDITIONS.index("Sunny"),
    help="Dominant weather pattern. Options: Cloudy, Rainy, Sunny.",
)

# --- Boolean inputs ---
st.sidebar.markdown("**Management**")
col_fert, col_irr = st.sidebar.columns(2)
fertilizer_used = col_fert.checkbox("Fertilizer", value=True)
irrigation_used = col_irr.checkbox("Irrigation", value=True)

# Convenience dict shared across tabs
conditions_payload = _build_conditions_payload(
    rainfall_mm=float(rainfall_mm),
    temperature_celsius=float(temperature_celsius),
    days_to_harvest=int(days_to_harvest),
    region=region,
    soil_type=soil_type,
    weather_condition=weather_condition,
    fertilizer_used=fertilizer_used,
    irrigation_used=irrigation_used,
)

# ---------------------------------------------------------------------------
# Example presets (sidebar quick-fill buttons)
# ---------------------------------------------------------------------------

st.sidebar.divider()
st.sidebar.subheader("Example Presets")
st.sidebar.caption("Click a preset to populate representative values below the sliders.")

PRESETS = {
    "Maize — Midwest": dict(
        rainfall_mm=650, temperature_celsius=22, days_to_harvest=120,
        region="East", soil_type="Loam", weather_condition="Sunny",
        fertilizer_used=True, irrigation_used=True,
    ),
    "Rice — South": dict(
        rainfall_mm=1200, temperature_celsius=28, days_to_harvest=130,
        region="South", soil_type="Clay", weather_condition="Rainy",
        fertilizer_used=True, irrigation_used=True,
    ),
    "Wheat — Plains": dict(
        rainfall_mm=400, temperature_celsius=12, days_to_harvest=240,
        region="North", soil_type="Silt", weather_condition="Cloudy",
        fertilizer_used=True, irrigation_used=False,
    ),
    "Soybean — Sandy": dict(
        rainfall_mm=500, temperature_celsius=24, days_to_harvest=100,
        region="West", soil_type="Sandy", weather_condition="Sunny",
        fertilizer_used=False, irrigation_used=True,
    ),
}

for preset_name, preset_vals in PRESETS.items():
    if st.sidebar.button(preset_name, use_container_width=True, key=f"preset_{preset_name}"):
        # Overwrite session state so sliders update on next render
        for k, v in preset_vals.items():
            st.session_state[k] = v
        st.rerun()

# Apply session state preset values to conditions (if preset was clicked)
for key in ["rainfall_mm", "temperature_celsius", "days_to_harvest"]:
    if key in st.session_state:
        conditions_payload[key] = st.session_state[key]
for key in ["region", "soil_type", "weather_condition", "fertilizer_used", "irrigation_used"]:
    if key in st.session_state:
        conditions_payload[key] = st.session_state[key]

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------

st.title("AgroOpt — Crop Yield Intelligence")
st.caption(
    "Data-driven crop yield prediction for US farms. "
    "Powered by a Ridge regression pipeline (R² = 0.913, 666 K training records)."
)

tab_predict, tab_recommend, tab_optimize = st.tabs(
    ["Predict", "Recommend", "Optimize"]
)

# ===========================================================================
# Tab 1 — PREDICT
# ===========================================================================

with tab_predict:
    st.subheader("Single-Crop Yield Prediction")
    st.markdown(
        "Select a crop and click **Predict** to estimate yield for the farm "
        "conditions set in the sidebar."
    )

    crop_predict = st.selectbox(
        "Crop",
        options=CROPS,
        index=0,
        key="crop_predict",
        help="Allowed values: Maize, Rice, Soybean, Wheat.",
    )

    # Reference table for users
    with st.expander("FAO 2013 USA benchmark yields (reference)"):
        bench_rows = [
            {"Crop": c, "FAO Benchmark (hg/ha)": f"{v:,.0f}", "FAO Benchmark (t/ha)": f"{v/10000:.2f}"}
            for c, v in FAO_BENCHMARKS.items()
        ]
        st.table(bench_rows)

    if st.button("Predict Yield", type="primary", key="btn_predict"):
        with st.spinner("Calling /predict …"):
            result = _post(
                "/predict",
                {"conditions": conditions_payload, "crop": crop_predict},
            )
        if result:
            hg_ha = result["predicted_yield_hg_ha"]
            t_ha = result["predicted_yield_t_ha"]
            bench = FAO_BENCHMARKS[crop_predict]
            pct_vs_bench = (hg_ha - bench) / bench * 100

            col1, col2, col3 = st.columns(3)
            col1.metric(
                label=f"{crop_predict} — Predicted Yield",
                value=f"{t_ha:.3f} t/ha",
                delta=f"{pct_vs_bench:+.1f}% vs FAO benchmark",
            )
            col2.metric("hg/ha", f"{hg_ha:,.1f}")
            col3.metric("FAO Benchmark (t/ha)", f"{bench/10000:.2f}")

            # Gauge chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=t_ha,
                delta={"reference": bench / 10_000, "valueformat": ".3f"},
                title={"text": f"{crop_predict} Yield (t/ha)"},
                gauge={
                    "axis": {"range": [0, max(bench / 10_000 * 1.5, t_ha * 1.2)]},
                    "bar": {"color": CROP_COLORS.get(crop_predict, "#1f77b4")},
                    "threshold": {
                        "line": {"color": "red", "width": 3},
                        "thickness": 0.75,
                        "value": bench / 10_000,
                    },
                },
            ))
            fig.update_layout(height=320, margin=dict(t=40, b=10))
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Red line = FAO 2013 USA benchmark.")

# ===========================================================================
# Tab 2 — RECOMMEND
# ===========================================================================

with tab_recommend:
    st.subheader("Crop Recommendation — All Four Crops Ranked")
    st.markdown(
        "Click **Rank Crops** to evaluate Maize, Rice, Soybean, and Wheat "
        "for the conditions in the sidebar."
    )

    if st.button("Rank Crops", type="primary", key="btn_recommend"):
        with st.spinner("Calling /recommend …"):
            result = _post("/recommend", {"conditions": conditions_payload})
        if result:
            rankings = result["rankings"]

            # Bar chart
            crops_ordered = [r["crop"] for r in rankings]
            yields_t = [r["predicted_yield_t_ha"] for r in rankings]
            bench_t = [FAO_BENCHMARKS[c] / 10_000 for c in crops_ordered]
            colors = [CROP_COLORS.get(c, "#888") for c in crops_ordered]

            fig = go.Figure()
            fig.add_bar(
                name="Predicted yield",
                x=crops_ordered,
                y=yields_t,
                marker_color=colors,
                text=[f"{v:.3f}" for v in yields_t],
                textposition="outside",
            )
            fig.add_bar(
                name="FAO benchmark",
                x=crops_ordered,
                y=bench_t,
                marker_color="rgba(150,150,150,0.4)",
                marker_line_color="grey",
                marker_line_width=1,
                text=[f"{v:.2f}" for v in bench_t],
                textposition="outside",
            )
            fig.update_layout(
                barmode="group",
                title="Predicted vs FAO Benchmark Yield (t/ha)",
                yaxis_title="Yield (t/ha)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
                height=420,
                margin=dict(t=60, b=20),
            )
            st.plotly_chart(fig, use_container_width=True)

            # Ranking table
            st.markdown("#### Detailed Rankings")
            table_rows = []
            for r in rankings:
                table_rows.append({
                    "Rank": r["rank"],
                    "Crop": r["crop"],
                    "Yield (t/ha)": f"{r['predicted_yield_t_ha']:.3f}",
                    "Yield (hg/ha)": f"{r['predicted_yield_hg_ha']:,.0f}",
                    "Water Stress": f"{r['water_stress']:.1f}",
                    "Heat Stress (°C)": f"{r['heat_stress']:.1f}",
                    "FAO Benchmark (t/ha)": f"{r['fao_benchmark_hg_ha']/10000:.2f}",
                })
            st.table(table_rows)

            st.caption(
                "Water stress: deviation from crop optimal rainfall (0 = none). "
                "Heat stress: °C above crop max temperature threshold (0 = none)."
            )

# ===========================================================================
# Tab 3 — OPTIMIZE
# ===========================================================================

with tab_optimize:
    st.subheader("Management Input Optimisation")
    st.markdown(
        "Grid-searches over **fertilizer**, **irrigation**, and "
        "**days to harvest** (60–200, step 10) to maximise predicted yield "
        "for the chosen crop. Climate, soil, and region are held fixed."
    )

    crop_optimize = st.selectbox(
        "Crop to Optimise",
        options=CROPS,
        index=0,
        key="crop_optimize",
        help="Allowed values: Maize, Rice, Soybean, Wheat.",
    )

    if st.button("Optimise Management", type="primary", key="btn_optimize"):
        with st.spinner("Calling /optimize — grid-searching 2×2×15 = 60 combinations …"):
            result = _post(
                "/optimize",
                {"conditions": conditions_payload, "crop": crop_optimize},
            )
        if result:
            bc = result["best_conditions"]
            gain_pct = result["yield_gain_pct"]
            best_t = result["best_yield_t_ha"]
            base_t = result["baseline_yield_t_ha"]

            # KPI row
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Yield Gain", f"{gain_pct:+.1f}%")
            col2.metric("Best Yield (t/ha)", f"{best_t:.3f}")
            col3.metric("Baseline Yield (t/ha)", f"{base_t:.3f}", delta=f"{best_t - base_t:+.3f}")
            col4.metric(
                "Yield Gain (hg/ha)",
                f"{result['yield_gain_hg_ha']:,.0f}",
            )

            # Recommended conditions
            st.markdown("#### Recommended Management Inputs")
            rec_col1, rec_col2, rec_col3 = st.columns(3)
            rec_col1.info(f"**Fertilizer:** {'Yes' if bc['fertilizer_used'] else 'No'}")
            rec_col2.info(f"**Irrigation:** {'Yes' if bc['irrigation_used'] else 'No'}")
            rec_col3.info(f"**Days to Harvest:** {bc['days_to_harvest']} d")

            # Waterfall chart: baseline -> gain -> best
            fig = go.Figure(go.Waterfall(
                orientation="v",
                measure=["absolute", "relative", "total"],
                x=["Baseline", "Gain from optimisation", "Optimised"],
                y=[base_t, best_t - base_t, best_t],
                text=[f"{base_t:.3f}", f"+{best_t - base_t:.3f}", f"{best_t:.3f}"],
                textposition="outside",
                connector={"line": {"color": "rgb(63, 63, 63)"}},
                increasing={"marker": {"color": CROP_COLORS.get(crop_optimize, "#2ecc71")}},
                totals={"marker": {"color": "#2980b9"}},
            ))
            fig.update_layout(
                title=f"{crop_optimize} — Yield Waterfall (t/ha)",
                yaxis_title="Yield (t/ha)",
                height=380,
                margin=dict(t=50, b=20),
            )
            st.plotly_chart(fig, use_container_width=True)

            st.caption(
                "Baseline uses the exact conditions from the sidebar. "
                "Optimised conditions keep climate/soil/region fixed and "
                "vary only fertilizer, irrigation, and harvest timing."
            )

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.divider()
st.caption(
    "AgroOpt Phase 8 — Streamlit frontend | "
    "Backend: FastAPI + Ridge pipeline (sklearn) | "
    "Data: 666 K synthetic USA farm records"
)
