import pandas as pd
import numpy as np
import pandas_ta as ta

# raw data for SPY
df = pd.read_csv("dataset/vanilla.csv")
df["Date"] = pd.DatetimeIndex(df["Date"])

# raw data for VIX
vix_df = pd.read_csv("dataset/vix_vanilla.csv")
vix_df["Date"] = pd.DatetimeIndex(vix_df["Date"])

# generate market indicators
indicators = [
    "ao", "apo", "bbands", "bias", "cmo", "coppock", "decay", "dm", 
    "dpo",  "midprice", "natr", "ohlc4", "qqe", "rsi", "rvi", "wcp",
    "thermo", "trix", "zscore"
]

for indicator in indicators:
    if hasattr(df.ta, indicator):
        try:
            getattr(df.ta, indicator)(append=True)
        except Exception as e:
            print(f"Error with sp when calling {indicator}: {e}")
        try:
            getattr(vix_df.ta, indicator)(append=True)
        except Exception as e:
            print(f"Error with vix when calling {indicator}: {e}")

            
base = ["Open", "Close", "High", "Low", "Volume"]
columns_to_select = [
    "DMN_14",
    "AO_5_34",
    "APO_12_26",
    "BIAS_SMA_26",
    "CMO_14",
    "COPC_11_14_10",
    "DPO_20",
    "NATR_14",
    "QQE_14_5_4.236_RSIMA",
    "RSI_14",
    "RVI_14",
    "THERMOma_20_2_0.5",
    "ZS_30",
    "TRIX_30_9"
]
VIX_columns_to_select = [
    "VIX_Close", 
    "VIX_Open", 
    "VIX_High", 
    "VIX_Low", 
    "VIX_WCP", 
    "VIX_OHLC4",
    "VIX_MIDPRICE_2", 
    "VIX_LDECAY_5"]

vix_df = vix_df.rename(columns=lambda x: f"VIX_{x}" if x != "Date" else x)
data = pd.concat([df[base+columns_to_select], vix_df[VIX_columns_to_select]], axis = 1)

data["High_Open_ration"] = (data["High"] -data["Open"]) / data["Open"] * 100
data["Close_Open_Ratio"] = (data["Close"] -data["Open"]) / data["Open"] * 100

# delete open and high price columns, since we only focus on their difference (w.r.t. to open price"s ratio)
# same for close.
raw_price = data.iloc[:, 0].to_numpy().astype(float)
# data = data.drop(["Open", "Close", "High"], axis=1)

# get the missing data positions and the fill-mask
mask_filled = data.isna().to_numpy().astype(int)[:, :-2]

# fill the missing data by the mean value of each column
data_filled = data.fillna(data.mean())

# Normalize the feature
feature = data_filled.iloc[:, :-2]
feature_normalized = (feature - feature.min()) / (feature.max() - feature.min())
feature_normalized = feature_normalized.to_numpy().astype(float)

# Add labels
label = data_filled.iloc[:, -2:].to_numpy().astype(float)
data_normalized = np.hstack([feature_normalized, label, raw_price.reshape(-1, 1)])

print(mask_filled.shape, data_normalized.shape)

np.save("dataset/try2try.npy", data_normalized)