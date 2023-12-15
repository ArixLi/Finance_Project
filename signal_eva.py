import pandas as pd
import pandas_ta as ta
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("dataset/vanilla.csv", sep=",")

def plot_cdf(df, x, y):
    """
    Plot the Cumulative Distribution Function (CDF) for given 'x' and 'y' columns in a DataFrame.

    This function takes a DataFrame 'df' and two column names 'x' and 'y'. It calculates the CDF of the 'y' values 
    relative to 'x', and then plots this CDF. The CDF is calculated by first shifting the 'y' values, sorting the 
    DataFrame based on 'x' values, handling infinite values, and then computing the cumulative sum of the 
    centered 'y' values.

    Parameters:
    df (DataFrame): The input DataFrame containing the data.
    x (str): The column name in 'df' to be plotted on the x-axis.
    y (str): The column name in 'df' whose cumulative distribution is to be plotted on the y-axis.

    The function creates a plot showing how the 'y' values are cumulatively distributed across the 'x' values.
    """
    sub_df = pd.DataFrame({'X': df[x], 'Y': df[y].shift(-1)})
    sub_df = sub_df.sort_values(by='X').replace([np.inf, -np.inf], np.nan).dropna()
    sub_df['Y_cdf'] = (sub_df['Y'] - sub_df['Y'].mean()).cumsum()
    plt.plot(sub_df['X'], sub_df['Y_cdf'], label=f'{x} CDF')
    plt.legend()
    plt.show()



df.ta.dm( append=True)

plot_cdf(df, 'DMN_14', 'High_Open_ration')