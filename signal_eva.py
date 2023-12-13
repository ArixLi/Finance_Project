import pandas as pd
import pandas_ta as ta
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("dataset/vanilla.csv", sep=",")

def plot_cdf(df, x, y):
    sub_df = pd.DataFrame({'X': df[x], 'Y': df[y].shift(-1)})
    sub_df = sub_df.sort_values(by='X').replace([np.inf, -np.inf], np.nan).dropna()
    sub_df['Y_cdf'] = (sub_df['Y'] - sub_df['Y'].mean()).cumsum()
    plt.plot(sub_df['X'], sub_df['Y_cdf'], label=f'{x} CDF')
    plt.legend()
    plt.show()

df.ta.dm( append=True)

plot_cdf(df, 'DMN_14', 'High_Open_ration')