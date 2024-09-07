import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
import numpy as np
import pmdarima as pm

chunks = []
chunk_size = 15000

for chunk in pd.read_csv("prices.csv", chunksize=chunk_size):
    chunks.append(chunk)

df = pd.concat(chunks)
print(df.info())

print(df['symbol'].value_counts())

apple_df = df[df['symbol'] == 'AAPL']
print(apple_df)
print(apple_df.info())
apple_df = apple_df.copy()
apple_df['date'] = pd.to_datetime(apple_df['date'])
apple_df.set_index('date', inplace=True)
apple_df[['open', 'close']].plot()
plt.show()
apple_df['volume'].plot()
plt.show()

apple_df = apple_df[apple_df['open']>=150]

print(len(apple_df))
apple_df[['open', 'close']].plot()
plt.show()
apple_df['volume'].plot()
plt.show()

print(apple_df)

apple_df = apple_df.drop("symbol", axis=1)

for column in apple_df.columns:
    apple_df[f"log_{column}"] = np.log(apple_df[column])

print(apple_df)

apple_df['log_volume'].plot()
plt.show()
stationarity_dict = {}
for column in apple_df.columns:
    p_value = adfuller(apple_df[column])[1]
    stationarity_dict[column] = p_value
    print(f"p-value of augmented dickey- fuller stationarity test for {column} variable: {p_value}")

apple_df_diff = apple_df.diff().dropna()
print(apple_df_diff)

for column in apple_df_diff.columns:
    apple_df_diff[column].plot()
    plt.title(f"trend plot of {column} variable after first order differencing")
    plt.show()
diff_stationarity_dict = {}
for column in apple_df_diff.columns:
    p_value = adfuller(apple_df_diff[column])[1]
    stationarity_dict[column] = p_value
    print(f"p-value of augmented dickey- fuller stationarity test for {column} variable: {p_value}")

fig,ax = plt.subplots(1,2)
ax[0].bar(stationarity_dict.keys(), stationarity_dict.values())
ax[1].bar(diff_stationarity_dict.keys(), diff_stationarity_dict.values())
plt.show()

for column in apple_df_diff.columns:
    plot_pacf(apple_df_diff[column])
    plt.title(f"PACF plot of differenced {column} variable")
    plot_acf(apple_df_diff[column])
    plt.title(f"ACF plot of differenced {column} variable")
    plt.show()

for column in apple_df.columns:
    plot_pacf(apple_df[column])
    plt.title(f"PACF plot of {column} variable")
    plot_acf(apple_df[column])
    plt.title(f"ACF plot of {column} variable")
    plt.show()

for column in apple_df.columns:
    auto_arima = pm.auto_arima(apple_df[column], stepwise=False, seasonal=False)
    print(f"{column}: {auto_arima}")


