import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ── 1. Load BDI ──────────────────────────────────────────
bdi       = pd.read_csv("/Users/lunga_2.0/freight-analysis/data/BDI.csv")

bdi["Date"] = pd.to_datetime(bdi["Date"], format="%b %d, %Y")
bdi = bdi[["Date", "CI"]].rename(columns={"CI": "Freight_Rate"})
bdi = bdi.set_index("Date").sort_index()

# ── 2. Load Oil ──────────────────────────────────────────
oil       = pd.read_csv("/Users/lunga_2.0/freight-analysis/data/BrentOilPrices.csv")

oil["Date"] = pd.to_datetime(oil["Date"], format="mixed")
oil = oil.set_index("Date").rename(columns={"Price": "Oil_Price"}).sort_index()

# ── 3. Load Trade (World Bank) ───────────────────────────
trade_raw = pd.read_csv("/Users/lunga_2.0/freight-analysis/data/API_NE.EXP.GNFS.CD_DS2_en_csv_v2_175529.csv", skiprows=4)
world = trade_raw[trade_raw["Country Name"] == "World"]

# Melt year columns into rows
year_cols = [c for c in trade_raw.columns if c.isdigit()]
world_long = world[year_cols].T.reset_index()
world_long.columns = ["Year", "Trade_Volume"]
world_long["Date"] = pd.to_datetime(world_long["Year"].astype(str))
world_long = world_long.set_index("Date")[["Trade_Volume"]]

# Resample yearly → monthly (forward fill)
trade_monthly = world_long.resample("ME").ffill()

# ── 4. Merge all on monthly frequency ───────────────────
bdi_monthly = bdi.resample("ME").mean()        # daily → monthly average
oil_monthly  = oil.resample("ME").mean()        # daily → monthly average

df = bdi_monthly.join(oil_monthly).join(trade_monthly).dropna()
df = df.reset_index()

#print(df.shape)
#print(df.head())

# ── 1. Individual Trend Plots ─────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(12, 10))
fig.suptitle("Freight Rate, Oil Price & Trade Volume Trends (2012–2019)", fontsize=14)

axes[0].plot(df["Date"], df["Freight_Rate"], color="steelblue")
axes[0].set_title("Freight Rate (Capesize Index)")
axes[0].set_ylabel("Index")

axes[1].plot(df["Date"], df["Oil_Price"], color="darkorange")
axes[1].set_title("Brent Crude Oil Price")
axes[1].set_ylabel("USD per Barrel")

axes[2].plot(df["Date"], df["Trade_Volume"], color="green")
axes[2].set_title("Global Trade Volume (World Exports)")
axes[2].set_ylabel("USD")

for ax in axes:
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("/Users/lunga_2.0/freight-analysis/plots/trends.png", dpi=150)
plt.show()

# ── 2. Correlation Matrix ─────────────────────────────────
print("\n=== Correlation Matrix ===")
print(df[["Freight_Rate", "Oil_Price", "Trade_Volume"]].corr().round(3))

# ── 3. Scatter Plots ──────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Freight Rate vs Drivers", fontsize=14)

axes[0].scatter(df["Oil_Price"], df["Freight_Rate"], color="darkorange", alpha=0.6)
axes[0].set_xlabel("Oil Price (USD/barrel)")
axes[0].set_ylabel("Freight Rate")
axes[0].set_title("Freight Rate vs Oil Price")

axes[1].scatter(df["Trade_Volume"], df["Freight_Rate"], color="green", alpha=0.6)
axes[1].set_xlabel("Trade Volume (USD)")
axes[1].set_ylabel("Freight Rate")
axes[1].set_title("Freight Rate vs Trade Volume")

plt.tight_layout()
plt.savefig("/Users/lunga_2.0/freight-analysis/plots/scatter.png", dpi=150)
plt.show()

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
import numpy as np

# ── Features and target ──────────────────────────────────
X = df[["Oil_Price", "Trade_Volume"]]
y = df["Freight_Rate"]

# ── Train model ──────────────────────────────────────────
model = LinearRegression()
model.fit(X, y)

# ── Coefficients ─────────────────────────────────────────
print("=== Regression Results ===")
print(f"Intercept:        {model.intercept_:.2f}")
print(f"β1 Oil Price:     {model.coef_[0]:.4f}")
print(f"β2 Trade Volume:  {model.coef_[1]:.6e}")

# ── Model performance ────────────────────────────────────
df["Predicted"] = model.predict(X)
r2  = r2_score(y, df["Predicted"])
mae = mean_absolute_error(y, df["Predicted"])
print(f"\nR² Score:         {r2:.3f}")
print(f"MAE:              {mae:.2f}")

# ── Actual vs Predicted plot ──────────────────────────────
plt.figure(figsize=(12, 5))
plt.plot(df["Date"], df["Freight_Rate"], label="Actual", color="steelblue")
plt.plot(df["Date"], df["Predicted"],   label="Predicted", color="red", linestyle="--")
plt.title("Actual vs Predicted Freight Rate")
plt.ylabel("Freight Rate")
plt.xlabel("Date")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("/Users/lunga_2.0/freight-analysis/plots/regression.png", dpi=150)
plt.show()