import pandas as pd
import numpy as np


# import data from csv file, exclude the first column Date since we will treat it as a time-series data
data = pd.read_csv('good_cdf.csv').iloc[:, 1:]

# delete open and high price columns, since we only focus on their difference (w.r.t. to open price's ratio)
data = data.drop(['Open', 'High'], axis=1)

# get the missing data positions and the fill-mask
mask_filled = data.isna().to_numpy().astype(int)

# fill the missing data by the mean value of each column
data_filled = data.fillna(data.mean())

# normalize the data
data_normalized = (data_filled - data_filled.min()) / (data_filled.max() - data_filled.min())

# save the data
np.save('mask.npy', mask_filled)
np.save('data.npy', data_normalized)