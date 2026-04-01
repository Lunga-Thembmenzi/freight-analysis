import pandas as pd

# ── 1. Load BDI ──────────────────────────────────────────
bdi = pd.read_csv("MASTER EXCEL SHEET BDI -Table 1.csv")
bdi["Date"] = pd.to_datetime(bdi["Date"], format="%b %d, %Y")
bdi = bdi[["Date", "CI"]].rename(columns={"CI": "Freight_Rate"})
bdi = bdi.set_index("Date").sort_index()

# ── 2. Load Oil ──────────────────────────────────────────
oil = pd.read_csv("BrentOilPrices.csv")
oil["Date"] = pd.to_datetime(oil["Date"], format="mixed")
oil = oil.set_index("Date").rename(columns={"Price": "Oil_Price"}).sort_index()

# ── 3. Load Trade (World Bank) ───────────────────────────
trade_raw = pd.read_csv("API_NE.EXP.GNFS.CD_DS2_en_csv_v2_175529.csv", skiprows=4)
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

print(df.shape)
print(df.head())