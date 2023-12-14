import pandas as pd
import numpy as np
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

df = pd.read_csv('dataset/vanilla.csv')

# for length in range(10, 40, 2):
for length in [7]:
    dmdf = dm(df['High'], df['Low'], length)
    df = pd.concat([df, dmdf], axis=1)

df['High_Open_ration'] = (df['High'] -df['Open']) / df['Open'] * 100
df['Close_Open_Ratio'] = (df['Close'] -df['Open']) / df['Open'] * 100


# import data from csv file, exclude the first column Date since we will treat it as a time-series data
data = df.iloc[:, 1:]

# delete open and high price columns, since we only focus on their difference (w.r.t. to open price's ratio)
# same for close.
raw_price = data.iloc[:, 0].to_numpy().astype(float)
# data = data.drop(['Open', 'Close', 'High'], axis=1)

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
# np.save('dataset/spx.npy', mask_filled)
np.save('dataset/try1try.npy', data_normalized)