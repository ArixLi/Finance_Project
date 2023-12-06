import pandas as pd
import numpy as np


# import data from csv file, exclude the first column Date since we will treat it as a time-series data
data = pd.read_csv('good_w_vix.csv').iloc[:, 1:]

# delete open and high price columns, since we only focus on their difference (w.r.t. to open price's ratio)
# same for close.
data = data.drop(['Open', 'Close', 'High', 'High_Open'], axis=1)

# get the missing data positions and the fill-mask
mask_filled = data.isna().to_numpy().astype(int)

# fill the missing data by the mean value of each column
data_filled = data.fillna(data.mean())

# Normalize the feature
feature = data_filled.iloc[:, :-2]
feature_normalized = (feature - feature.min()) / (feature.max() - feature.min())
feature_normalized = feature_normalized.to_numpy().astype(float)

# Add labels
label = data_filled.iloc[:, -2:].to_numpy().astype(float)
data_normalized = np.hstack([feature_normalized, label])

# save the data
np.save('mask_vix.npy', mask_filled)
np.save('data_vix.npy', data_normalized)