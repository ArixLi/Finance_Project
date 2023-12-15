import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# the threshold for long
delta = 0.5 / 100

df = pd.read_csv('dataset/good_cdf.csv').iloc[:, 1:4].to_numpy()

# compute the High - Open ratio and Close - Open ratio
high_open_ratio = (df[:, 2] - df[:, 0]) / df[:, 0]
close_open_ratio = (df[:, 1] - df[:, 0]) / df[:, 0]

long_day = []

test_ind = int(len(df) * 0.7)

# if S(i-2) < S(i-1) < S(i), long at i+i day
for i in range(max(3, test_ind), len(df)):
    if df[i-3, 1] < df[i-2, 1] < df[i-1, 1]:
        long_day.append(i)

# Compute PL for the long days
PL = 0
for i in long_day:
    if high_open_ratio[i] >= delta:
        PL += high_open_ratio[i] * df[i, 0]
    else:
        PL += close_open_ratio[i] * df[i, 0]

print(PL)

