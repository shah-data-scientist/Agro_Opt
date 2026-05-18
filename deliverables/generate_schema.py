"""
CropWise — Data Schema Diagram Generator
Produces a PNG showing the full data lineage:
  5 raw sources → merged_dataset → features_dataset → API schemas
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
C_RAW       = "#1565C0"   # raw sources — blue
C_PROC      = "#2E7D32"   # processed — green
C_FEAT      = "#6A1B9A"   # features — purple
C_API       = "#BF360C"   # API — red-orange
C_BG        = "#F8F9FA"
C_HDR       = "#ECEFF1"
C_BORDER    = "#455A64"
C_TXT       = "#212121"
C_TXT_LIGHT = "#FFFFFF"
C_ARROW     = "#546E7A"

FIG_W, FIG_H = 22, 14

# ---------------------------------------------------------------------------
# Schema definitions
# ---------------------------------------------------------------------------
RAW_TABLES = [
    ("crop_yield.csv\n666 494 rows", C_RAW, [
        ("Region",                "str"),
        ("Soil_Type",             "str"),
        ("Crop",                  "str"),
        ("Rainfall_mm",           "float"),
        ("Temperature_Celsius",   "float"),
        ("Fertilizer_Used",       "bool"),
        ("Irrigation_Used",       "bool"),
        ("Weather_Condition",     "str"),
        ("Days_to_Harvest",       "int"),
        ("Yield_tons_per_hectare","float ← target"),
    ]),
    ("yield.csv\nFAO hg/ha", C_RAW, [
        ("Area",    "str (USA)"),
        ("Item",    "str (crop)"),
        ("Year",    "int"),
        ("Element", "str"),
        ("Unit",    "str"),
        ("Value",   "int"),
    ]),
    ("rainfall.csv\nFAO mm/yr", C_RAW, [
        ("Area",                          "str (USA)"),
        ("Year",                          "int"),
        ("average_rain_fall_mm_per_year", "int"),
    ]),
    ("temp.csv\nFAO °C", C_RAW, [
        ("country",  "str (USA)"),
        ("year",     "int"),
        ("avg_temp", "float"),
    ]),
    ("pesticides.csv\nFAO tonnes", C_RAW, [
        ("Area",    "str (USA)"),
        ("Item",    "str"),
        ("Year",    "int"),
        ("Unit",    "str"),
        ("Value",   "int"),
    ]),
]

MERGED = ("merged_dataset.csv\n666 494 rows · 15 cols", C_PROC, [
    ("crop",                 "str"),
    ("year",                 "int  [2013]"),
    ("region",               "str"),
    ("soil_type",            "str"),
    ("weather_condition",    "str"),
    ("rainfall_mm",          "float"),
    ("temperature_celsius",  "float"),
    ("fertilizer_used",      "bool"),
    ("irrigation_used",      "bool"),
    ("days_to_harvest",      "int"),
    ("yield_hg_ha",          "float ← target"),
    ("fao_yield_hg_ha",      "int"),
    ("fao_rainfall_mm",      "int"),
    ("fao_pesticides_tonnes","float"),
    ("fao_avg_temp",         "float"),
])

FEATURES = ("features_dataset.csv\n666 494 rows · 37 cols", C_FEAT, [
    # original
    ("rainfall_mm / temperature_celsius", "float"),
    ("fertilizer_used / irrigation_used", "int [0/1]"),
    ("days_to_harvest",                   "int"),
    ("fao_yield_hg_ha",                   "int"),
    # anomalies
    ("rainfall_anomaly / temp_anomaly",   "float"),
    # interactions
    ("rainfall_x_fertilizer/irrigation",  "float"),
    ("temp_x_irrigation",                 "float"),
    ("agro_intensity",                    "int"),
    # domain
    ("heat_moisture_ratio / aridity_idx", "float"),
    ("harvest_rainfall_rate",             "float"),
    ("water_stress / heat_stress",        "float"),
    ("gdd_proxy / soil_quality_score",    "float/int"),
    # one-hot (17 cols)
    ("crop_*  (4 cols)",                  "int [0/1]"),
    ("region_*  (4 cols)",                "int [0/1]"),
    ("soil_type_*  (6 cols)",             "int [0/1]"),
    ("weather_condition_*  (3 cols)",     "int [0/1]"),
    # target
    ("yield_hg_ha",                       "float ← target"),
])

API_SCHEMAS = ("API Schemas\n(Pydantic)", C_API, [
    # request
    ("FarmConditionsRequest", "─── rainfall_mm, temp, days,"),
    ("",                      "    region, soil, weather,"),
    ("",                      "    fertilizer, irrigation"),
    ("PredictRequest",        "─── conditions + crop"),
    ("RecommendRequest",      "─── conditions"),
    ("OptimizeRequest",       "─── conditions + crop"),
    # response
    ("PredictResponse",       "─── crop, yield_hg_ha, t_ha"),
    ("CropRanking",           "─── rank, crop, yield,"),
    ("",                      "    water/heat_stress, fao_ref"),
    ("OptimizeResponse",      "─── best_conditions, gain_%"),
    ("HealthResponse",        "─── status, model, n_features"),
])

# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def draw_table(ax, x, y, w, h_row, title, colour, rows, title_h=0.45):
    n = len(rows)
    total_h = title_h + n * h_row

    # Box shadow
    ax.add_patch(FancyBboxPatch(
        (x + 0.04, y - total_h - 0.04), w, total_h,
        boxstyle="round,pad=0.02", linewidth=0,
        facecolor="#B0BEC5", zorder=1
    ))
    # Main box
    ax.add_patch(FancyBboxPatch(
        (x, y - total_h), w, total_h,
        boxstyle="round,pad=0.02", linewidth=1.2,
        edgecolor=C_BORDER, facecolor="white", zorder=2
    ))
    # Header
    ax.add_patch(FancyBboxPatch(
        (x, y - title_h), w, title_h,
        boxstyle="round,pad=0.02", linewidth=0,
        facecolor=colour, zorder=3
    ))
    ax.text(x + w / 2, y - title_h / 2, title,
            ha="center", va="center", fontsize=7.5, fontweight="bold",
            color=C_TXT_LIGHT, zorder=4, linespacing=1.4)

    # Rows
    for i, (col, dtype) in enumerate(rows):
        row_y = y - title_h - (i + 0.5) * h_row
        # Alternating row bg
        bg = "#EDE7F6" if colour == C_FEAT else ("#E8F5E9" if colour == C_PROC else
             "#FBE9E7" if colour == C_API else "#E3F2FD")
        if i % 2 == 0:
            ax.add_patch(plt.Rectangle(
                (x + 0.02, y - title_h - (i + 1) * h_row + 0.01),
                w - 0.04, h_row - 0.01,
                facecolor=bg, edgecolor="none", zorder=2
            ))
        ax.text(x + 0.12, row_y, col,
                ha="left", va="center", fontsize=6.5, color=C_TXT, zorder=5)
        ax.text(x + w - 0.08, row_y, dtype,
                ha="right", va="center", fontsize=6.0, color="#546E7A",
                style="italic", zorder=5)

    return total_h


def draw_arrow(ax, x0, y0, x1, y1, label=""):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=C_ARROW,
                                lw=1.5, mutation_scale=12),
                zorder=6)
    if label:
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        ax.text(mx, my + 0.1, label, ha="center", va="bottom",
                fontsize=6, color=C_ARROW, style="italic")


# ---------------------------------------------------------------------------
# Build figure
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
fig.patch.set_facecolor(C_BG)
ax.set_facecolor(C_BG)
ax.set_xlim(0, FIG_W)
ax.set_ylim(0, FIG_H)
ax.axis("off")

# Title
ax.text(FIG_W / 2, FIG_H - 0.35, "CropWise — Data Schema & Pipeline",
        ha="center", va="center", fontsize=14, fontweight="bold", color=C_TXT)
ax.axhline(FIG_H - 0.6, color=C_BORDER, lw=0.8, alpha=0.4)

# ---------------------------------------------------------------------------
# RAW TABLES  (x: 0.3 … 8.5)
# ---------------------------------------------------------------------------
H_ROW = 0.29
raw_xs = [0.3, 2.15, 4.0, 5.6, 7.2]
raw_tops = []

for i, (title, colour, rows) in enumerate(RAW_TABLES):
    x = raw_xs[i]
    y = FIG_H - 0.8
    th = draw_table(ax, x, y, 1.7, H_ROW, title, colour, rows)
    raw_tops.append((x + 0.85, y - th))   # bottom-centre

# ---------------------------------------------------------------------------
# MERGED (x: 9.0)
# ---------------------------------------------------------------------------
MG_X, MG_Y = 9.0, FIG_H - 0.8
mg_w = 3.2
mg_h = draw_table(ax, MG_X, MG_Y, mg_w, H_ROW, MERGED[0], MERGED[1], MERGED[2])
mg_in  = (MG_X, MG_Y - mg_h / 2)        # left-centre
mg_out = (MG_X + mg_w, MG_Y - mg_h / 2) # right-centre

# Arrows: raw → merged (fan in)
for (rx, ry) in raw_tops:
    # use a curved path via annotation
    ax.annotate("", xy=(MG_X, MG_Y - 0.3), xytext=(rx + 0.85 - 0.85, ry),
                arrowprops=dict(arrowstyle="-|>", color=C_ARROW,
                                lw=1.2, mutation_scale=10,
                                connectionstyle="arc3,rad=0.0"),
                zorder=6)

# merge label
ax.text((raw_xs[-1] + 1.7 + MG_X) / 2, FIG_H - 4.5,
        "join on\ncrop + year\n(USA only)",
        ha="center", va="center", fontsize=6.5,
        color=C_ARROW, style="italic",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#ECEFF1",
                  edgecolor=C_ARROW, lw=0.8))

# ---------------------------------------------------------------------------
# FEATURES (x: 12.7)
# ---------------------------------------------------------------------------
FT_X, FT_Y = 12.7, FIG_H - 0.8
ft_w = 3.8
ft_h = draw_table(ax, FT_X, FT_Y, ft_w, H_ROW, FEATURES[0], FEATURES[1], FEATURES[2])
ft_in  = (FT_X, FT_Y - ft_h / 2)
ft_out = (FT_X + ft_w, FT_Y - ft_h / 2)

draw_arrow(ax, MG_X + mg_w, MG_Y - mg_h / 2,
               FT_X, FT_Y - ft_h / 2,
               "feature_engineering.py")

# ---------------------------------------------------------------------------
# MODEL BOX  (x: 17.0, centre)
# ---------------------------------------------------------------------------
MODEL_X, MODEL_Y = 17.1, FIG_H - 3.5
model_w, model_h = 2.3, 1.8
ax.add_patch(FancyBboxPatch(
    (MODEL_X, MODEL_Y - model_h), model_w, model_h,
    boxstyle="round,pad=0.05", linewidth=1.5,
    edgecolor=C_BORDER, facecolor="#37474F", zorder=2
))
ax.text(MODEL_X + model_w / 2, MODEL_Y - 0.3,
        "Ridge Regression\nPipeline",
        ha="center", va="center", fontsize=8, fontweight="bold",
        color="white", zorder=4)
ax.text(MODEL_X + model_w / 2, MODEL_Y - 0.9,
        "StandardScaler\n+ Ridge(α=1.0)",
        ha="center", va="center", fontsize=7,
        color="#CFD8DC", zorder=4)
ax.text(MODEL_X + model_w / 2, MODEL_Y - 1.45,
        "R²=0.913 | RMSE=4 989 hg/ha",
        ha="center", va="center", fontsize=6.5,
        color="#80CBC4", zorder=4)

draw_arrow(ax, FT_X + ft_w, FT_Y - ft_h / 2,
               MODEL_X, MODEL_Y - model_h / 2,
               "train.py / MLflow")

# ---------------------------------------------------------------------------
# API SCHEMAS (x: 17.0, lower)
# ---------------------------------------------------------------------------
AP_X, AP_Y = 17.1, FIG_H - 6.2
ap_w = 4.5
ap_h = draw_table(ax, AP_X, AP_Y, ap_w, 0.32, API_SCHEMAS[0], API_SCHEMAS[1], API_SCHEMAS[2])

draw_arrow(ax, MODEL_X + model_w / 2, MODEL_Y - model_h,
               AP_X + ap_w / 2, AP_Y,
               "best_model.pkl")

# ---------------------------------------------------------------------------
# Legend
# ---------------------------------------------------------------------------
legend_items = [
    (C_RAW,  "Raw data (CSV)"),
    (C_PROC, "Processed / merged"),
    (C_FEAT, "Feature-engineered"),
    (C_API,  "API schemas"),
]
lx, ly = 0.3, 1.0
for colour, label in legend_items:
    ax.add_patch(plt.Rectangle((lx, ly), 0.25, 0.18,
                               facecolor=colour, edgecolor="none"))
    ax.text(lx + 0.32, ly + 0.09, label,
            va="center", fontsize=7, color=C_TXT)
    lx += 1.8

ax.text(0.3, 0.65, "Key:  bold italic = target variable  |  all row counts after deduplication",
        fontsize=6.5, color="#546E7A", style="italic")

plt.tight_layout(pad=0.2)
out = "deliverables/data_schema_032025.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=C_BG)
print(f"Saved → {out}")
