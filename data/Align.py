#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

plt.rcParams["figure.dpi"] = 120
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False

# ── File paths — update these to match your local directory ──────────────
DATA_DIR = "/Users/lunga_2.0/freight-analysis/data/"   # <-- change this to your data folder path

BDI_PATH    = DATA_DIR + "BDI.csv"
OIL_PATH    = DATA_DIR + "BrentOilPrices.csv"
TRADE_PATH  = DATA_DIR + "API_NE.EXP.GNFS.CD_DS2_en_csv_v2_175529.csv"
FLEET_PATH  = DATA_DIR + "US_Merchant.csv"  # UNCTAD bulk carrier fleet DWT
PLOTS_DIR   = "/Users/lunga_2.0/freight-analysis/plots/"
#%%
# ── 1. Load Baltic Dry Index (BDI) ───────────────────────────────────────
bdi = pd.read_csv(BDI_PATH)
bdi["Date"] = pd.to_datetime(bdi["Date"], format="%b %d, %Y")
bdi = bdi[["Date", "CI"]].rename(columns={"CI": "Freight_Rate"})
bdi = bdi.set_index("Date").sort_index()

# ── 2. Load Brent Oil Prices ─────────────────────────────────────────────
oil = pd.read_csv(OIL_PATH)
oil["Date"] = pd.to_datetime(oil["Date"], format="mixed")
oil = oil.set_index("Date").rename(columns={"Price": "Oil_Price"}).sort_index()

# ── 3. Load Global Trade Volume (World Bank) ─────────────────────────────
trade_raw = pd.read_csv(TRADE_PATH, skiprows=4)
world_row = trade_raw[trade_raw["Country Name"] == "World"]
year_cols = [c for c in trade_raw.columns if c.isdigit()]
world_long = world_row[year_cols].T.reset_index()
world_long.columns = ["Year", "Trade_Volume"]
world_long["Date"] = pd.to_datetime(world_long["Year"].astype(str))
world_long = world_long.set_index("Date")[["Trade_Volume"]]

# ── 4. Load Bulk Carrier Fleet Supply (UNCTAD) ───────────────────────────
# Source: UNCTAD merchant fleet statistics
# Filtered to: Ship type = Bulk carriers, Indicator = Dead weight tonnage, Economy = World
# Values are in thousands of DWT — multiplied by 1,000 below
fleet_raw = pd.read_csv(FLEET_PATH)
fleet_world = fleet_raw[fleet_raw["Economy_Label"] == "World"]

# Extract yearly Value columns only (exclude Footnote and MissingValue columns)
fleet_val_cols = [c for c in fleet_raw.columns if c.endswith("_Value") and c[:4].isdigit()]
fleet_rows = []
for col in fleet_val_cols:
    year = int(col.split("_")[0])
    val  = fleet_world[col].values[0]
    fleet_rows.append({"Date": pd.Timestamp(str(year)), "Fleet_DWT": val * 1000})

fleet_df = pd.DataFrame(fleet_rows).set_index("Date").dropna()

print("Datasets loaded successfully.")
print(f"BDI rows:   {len(bdi):,}")
print(f"Oil rows:   {len(oil):,}")
print(f"Trade rows: {len(world_long):,}")
print(f"Fleet rows: {len(fleet_df):,}  (years: {fleet_df.index.year.min()}–{fleet_df.index.year.max()})")
# ── 5. Align all datasets to monthly frequency ───────────────────────────
bdi_monthly   = bdi.resample("ME").mean()            # daily → monthly average
oil_monthly   = oil.resample("ME").mean()            # daily → monthly average
trade_monthly = world_long.resample("ME").ffill()    # yearly → monthly (forward-fill)
fleet_monthly = fleet_df.resample("ME").ffill()      # yearly → monthly (forward-fill)

# ── 6. Merge on date index ───────────────────────────────────────────────
df = (bdi_monthly
      .join(oil_monthly)
      .join(trade_monthly)
      .join(fleet_monthly)
      .dropna()
      .reset_index()
      .rename(columns={"index": "Date"}))

print(f"Merged dataset: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"Date range:     {df['Date'].min().strftime('%b %Y')} – {df['Date'].max().strftime('%b %Y')}")
print(f"\nColumn names: {list(df.columns)}")
df.head()
fig, axes = plt.subplots(4, 1, figsize=(13, 14))
fig.suptitle("Freight Rate, Oil Price, Trade Volume & Fleet Supply (2012–2019)", fontsize=14, y=1.01)

plot_cfg = [
    ("Freight_Rate", "Freight Rate (Capesize BDI)", "Index",           "steelblue"),
    ("Oil_Price",    "Brent Crude Oil Price",        "USD per Barrel",  "darkorange"),
    ("Trade_Volume", "Global Trade Volume",          "USD (World Bank)","seagreen"),
    ("Fleet_DWT",    "Bulk Carrier Fleet Supply",    "DWT (tonnes)",    "mediumpurple"),
]

for ax, (col, title, ylabel, color) in zip(axes, plot_cfg):
    ax.plot(df["Date"], df[col], color=color, linewidth=1.8)
    ax.set_title(title, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS_DIR + "trends_v2.png", dpi=150, bbox_inches="tight")
plt.show()


# ── 2. Correlation Matrix ─────────────────────────────────
corr_cols = ["Freight_Rate", "Oil_Price", "Trade_Volume", "Fleet_DWT"]
print("=== Correlation Matrix ===")
print(df[corr_cols].corr().round(3))

fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(df[corr_cols].corr(), annot=True, fmt=".3f",
            cmap="coolwarm", center=0, ax=ax,
            annot_kws={"size": 11})
ax.set_title("Correlation Matrix — Freight Market Variables", pad=12)
plt.tight_layout()
plt.savefig(PLOTS_DIR + "correlation_v2.png", dpi=150, bbox_inches="tight")
plt.show()

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Freight Rate vs Explanatory Variables", fontsize=13)

scatter_cfg = [
    ("Oil_Price",    "Oil Price (USD/barrel)",    "darkorange"),
    ("Trade_Volume", "Trade Volume (USD)",        "seagreen"),
    ("Fleet_DWT",    "Fleet Supply (DWT)",        "mediumpurple"),
]

for ax, (col, xlabel, color) in zip(axes, scatter_cfg):
    ax.scatter(df[col], df["Freight_Rate"], color=color, alpha=0.55, edgecolors="white", linewidths=0.3)
    # Trend line
    z = np.polyfit(df[col].astype(float), df["Freight_Rate"].astype(float), 1)
    p = np.poly1d(z)
    x_line = np.linspace(df[col].min(), df[col].max(), 100)
    ax.plot(x_line, p(x_line), color="black", linewidth=1.2, linestyle="--", alpha=0.6)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel("Freight Rate", fontsize=9)
    ax.set_title(f"Freight Rate vs {col.replace('_', ' ')}", fontsize=10)
    ax.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig(PLOTS_DIR + "scatter_v2.png", dpi=150, bbox_inches="tight")
plt.show()

# ── 4. Regression model ──────────────────────────────────

X1 = df[["Oil_Price", "Trade_Volume"]]
y  = df["Freight_Rate"]

model1 = LinearRegression().fit(X1, y)
df["Pred_M1"] = model1.predict(X1)

r2_m1  = r2_score(y, df["Pred_M1"])
mae_m1 = mean_absolute_error(y, df["Pred_M1"])

print("=== Model 1: Baseline (Oil + Trade) ===")
print(f"Intercept:        {model1.intercept_:.2f}")
print(f"β₁ Oil Price:     {model1.coef_[0]:.4f}")
print(f"β₂ Trade Volume:  {model1.coef_[1]:.6e}")
print(f"\nR² Score:         {r2_m1:.3f}")
print(f"MAE:              {mae_m1:.2f}")

#%%
X2 = df[["Oil_Price", "Trade_Volume", "Fleet_DWT"]]

model2 = LinearRegression().fit(X2, y)
df["Pred_M2"] = model2.predict(X2)

r2_m2  = r2_score(y, df["Pred_M2"])
mae_m2 = mean_absolute_error(y, df["Pred_M2"])

print("=== Model 2: Full Market Model (Oil + Trade + Fleet) ===")
print(f"Intercept:        {model2.intercept_:.2f}")
print(f"β₁ Oil Price:     {model2.coef_[0]:.4f}")
print(f"β₂ Trade Volume:  {model2.coef_[1]:.6e}")
print(f"β₃ Fleet DWT:     {model2.coef_[2]:.6e}")
print(f"\nR² Score:         {r2_m2:.3f}")
print(f"MAE:              {mae_m2:.2f}")
print(f"\nR² improvement over Baseline: +{r2_m2 - r2_m1:.3f}")



df["Oil_Lag1"] = df["Oil_Price"].shift(1)
df["Oil_Lag2"] = df["Oil_Price"].shift(2)
df_lag = df.dropna().copy()

X3 = df_lag[["Oil_Price", "Trade_Volume", "Fleet_DWT", "Oil_Lag1", "Oil_Lag2"]]
y3 = df_lag["Freight_Rate"]

model3 = LinearRegression().fit(X3, y3)
df_lag["Pred_M3"] = model3.predict(X3)

r2_m3  = r2_score(y3, df_lag["Pred_M3"])
mae_m3 = mean_absolute_error(y3, df_lag["Pred_M3"])

print("=== Model 3: Extended (Oil + Trade + Fleet + Lags) ===")
print(f"Intercept:        {model3.intercept_:.2f}")
print(f"β₁ Oil Price:     {model3.coef_[0]:.4f}")
print(f"β₂ Trade Volume:  {model3.coef_[1]:.6e}")
print(f"β₃ Fleet DWT:     {model3.coef_[2]:.6e}")
print(f"β₄ Oil Lag 1M:    {model3.coef_[3]:.4f}")
print(f"β₅ Oil Lag 2M:    {model3.coef_[4]:.4f}")
print(f"\nR² Score:         {r2_m3:.3f}")
print(f"MAE:              {mae_m3:.2f}")
print(f"\nR² improvement over Baseline: +{r2_m3 - r2_m1:.3f}")

#%%
fig, axes = plt.subplots(3, 1, figsize=(13, 13), sharex=True)
fig.suptitle("Actual vs Predicted Freight Rate — Model Comparison", fontsize=13)

models = [
    ("Pred_M1", df,     f"Model 1: Baseline (R²={r2_m1:.3f})",     "tomato"),
    ("Pred_M2", df,     f"Model 2: Full Market (R²={r2_m2:.3f})",   "seagreen"),
    ("Pred_M3", df_lag, f"Model 3: Extended (R²={r2_m3:.3f})",      "darkorchid"),
]

for ax, (pred_col, data, label, color) in zip(axes, models):
    ax.plot(data["Date"], data["Freight_Rate"], label="Actual",    color="steelblue", linewidth=1.8)
    ax.plot(data["Date"], data[pred_col],        label=label,       color=color,       linewidth=1.5, linestyle="--")
    ax.set_ylabel("Freight Rate", fontsize=9)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

axes[-1].set_xlabel("Date")
plt.tight_layout()
plt.savefig(PLOTS_DIR + "regression_comparison.png", dpi=150, bbox_inches="tight")
plt.show()

summary = pd.DataFrame({
    "Model":     ["Baseline (Oil + Trade)",
                  "Full Market (Oil + Trade + Fleet)",
                  "Extended (+ Lags)"],
    "Variables": [2, 3, 5],
    "R² Score":  [round(r2_m1, 3), round(r2_m2, 3), round(r2_m3, 3)],
    "MAE":       [round(mae_m1, 2), round(mae_m2, 2), round(mae_m3, 2)],
    "R² vs Baseline": ["—",
                        f"+{r2_m2 - r2_m1:.3f}",
                        f"+{r2_m3 - r2_m1:.3f}"]
})
print(summary.to_string(index=False))