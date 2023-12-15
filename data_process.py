import pandas as pd
import numpy as np
import pandas_ta as ta
from sys import float_info as sflt

def ma(series, length):
    alpha = (1.0 / length) if length > 0 else 0.5
    ma = series.ewm(alpha=alpha, min_periods=length).mean()
    return ma

def zero(x):
    return 0 if abs(x) < sflt.epsilon else x

def dm(high, low, length, drift=1):
    up = high - high.shift(drift)
    dn = low.shift(drift) - low

    neg_ = ((dn > up) & (dn > 0)) * dn

    neg_ = neg_.apply(zero)

    neg = ma(neg_, length=length)


    _params = f"_{length}"
    data = {
        f"DMN{_params}": neg,
    }

    dmdf = pd.DataFrame(data)
    return dmdf

df = pd.read_csv("dataset/vanilla.csv")
df["Date"] = pd.DatetimeIndex(df["Date"])
vix_df = pd.read_csv("dataset/vix_vanilla.csv")
vix_df["Date"] = pd.DatetimeIndex(vix_df["Date"])

# for length in range(10, 40, 2):
# for length in [14]:
#     dmdf = dm(df["High"], df["Low"], length)
#     df = pd.concat([df, dmdf], axis=1)

indicators = [
    "aberration", "above", "above_value", "accbands", "ad", "adosc", "adx", "alma", "amat", "ao", "aobv", "apo", "aroon", 
    "atr", "bbands", "below", "below_value", "bias", "bop", "brar", "cci", "cdl_pattern", "cdl_z", "cfo", "cg", "chop", 
    "cksp", "cmf", "cmo", "coppock", "cross", "cross_value", "cti", "decay", "decreasing", "dema", "dm", "donchian", 
    "dpo", "ebsw", "efi", "ema", "entropy", "eom", "er", "eri", "fisher", "fwma", "ha", "hilo", "hl2", "hlc3", "hma", 
    "hwc", "hwma", "ichimoku", "increasing", "inertia", "jma", "kama", "kc", "kdj", "kst", "kurtosis", "kvo", "linreg", 
    "log_return", "long_run", "macd", "mad", "massi", "mcgd", "median", "mfi", "midpoint", "midprice", "mom", "natr", 
    "nvi", "obv", "ohlc4", "pdist", "percent_return", "pgo", "ppo", "psar", "psl", "pvi", "pvo", "pvol", "pvr", "pvt", 
    "pwma", "qqe", "qstick", "quantile", "rma", "roc", "rsi", "rsx", "rvgi", "rvi", "short_run", "sinwma", "skew", "slope", 
    "sma", "smi", "squeeze", "squeeze_pro", "ssf", "stc", "stdev", "stoch", "stochrsi", "supertrend", "swma", "t3", 
    "td_seq", "tema", "thermo", "tos_stdevall", "trima", "trix", "true_range", "tsi", "tsignals", "ttm_trend", "ui", 
    "uo", "variance", "vhf", "vidya", "vortex", "vp", "vwap", "vwma", "wcp", "willr", "wma", "xsignals", "zlma", "zscore"
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
    "ABER_ATR_5_15",
    "DMN_14",
    "AO_5_34",
    "APO_12_26",
    "ATRr_14",
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
    "VIX_MIDPOINT_2", 
    "VIX_HLC3", 
    "VIX_HL2",
    "VIX_HA_close", 
    "VIX_LDECAY_5"]

vix_df = vix_df.rename(columns=lambda x: f"VIX_{x}" if x != "Date" else x)
data = pd.concat([df[base+columns_to_select], vix_df[VIX_columns_to_select]], axis = 1)

data["High_Open_ration"] = (data["High"] -data["Open"]) / data["Open"] * 100
data["Close_Open_Ratio"] = (data["Close"] -data["Open"]) / data["Open"] * 100
data.to_csv("ttt.csv")


# import data from csv file, exclude the first column Date since we will treat it as a time-series data
# data = df.iloc[:, 1:]

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
# save the data
# np.save("dataset/spx.npy", mask_filled)
np.save("dataset/data_vix.npy", data_normalized)