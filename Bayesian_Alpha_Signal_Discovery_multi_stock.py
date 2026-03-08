import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

tickers = ["AAPL", "MSFT", "AMZN", "GOOG", "META", "NVDA", "TSLA", "JPM", "XOM", "UNH"]
raw = yf.download(tickers, start="2015-01-01", auto_adjust=True)
print(raw.head())
print(raw.columns)

#only look at close price
close = raw["Close"]
returns = close.pct_change()   #daily returns
momentum = close.pct_change(20)    #past 20-day return
volatility = returns.rolling(20).std()   #rolling 20-day standard deviation of daily returns
target = close.pct_change().shift(-1)    #next day return

# wide to long format
df = pd.concat(
    [momentum.stack().rename("momentum"),
     volatility.stack().rename("volatility"),
     target.stack().rename("target")
     ],
    axis = 1       #Puts those 3 Series side by side into one DataFrame.
).reset_index()

df.columns = ["date", "ticker", "momentum", "volatility", "target"]
df = df.dropna().copy()
df["prediction"] = np.nan    #creates an empty NaN prediction column that you will fill later

# -----------------------------
#  Walk-forward prediction
# -----------------------------
dates = sorted(df["date"].unique())     #This gets all unique trading dates in order.

#train on past 3 years
# predict the next month
# then roll forward and repeat
# train on 2015–2017
# test on Jan 2018
# then train on Feb 2015–Jan 2018
# test on Feb 2018

train_window = 252 * 3      # about 3 years of trading days
test_window = 21            # about 1 month of trading days for prediction

for start in range(train_window, len(dates) - test_window, test_window):

    train_dates = dates[start - train_window:start]      # the past 3 years
    test_dates = dates[start:start + test_window]         # the next month

    train_data = df[df["date"].isin(train_dates)]
    test_data = df[df["date"].isin(test_dates)]

    X_train = train_data[["momentum", "volatility"]]
    y_train = train_data["target"]

    X_test = test_data[["momentum", "volatility"]]

    model = LinearRegression()
    model.fit(X_train, y_train)

    #This fills in predictions for the next month only.
    df.loc[df["date"].isin(test_dates), "prediction"] = model.predict(X_test)

# keep only rows with out-of-sample predictions
df = df.dropna(subset=["prediction"]).copy()

# -----------------------------
# 4. Create long/short signals by date
# -----------------------------
q_low = df.groupby("date")["prediction"].transform(lambda x: x.quantile(0.2))
q_high = df.groupby("date")["prediction"].transform(lambda x: x.quantile(0.8))

df["signal"] = 0
df.loc[df["prediction"] >= q_high, "signal"] = 1
df.loc[df["prediction"] <= q_low, "signal"] = -1

# -----------------------------
# 5. Compute portfolio returns
# -----------------------------

#For each date: take all stocks with signal = 1 average their realized next-day returns
# This gives the equal-weight return of the long basket.
long_ret = df[df["signal"] == 1].groupby("date")["target"].mean()
short_ret = df[df["signal"] == -1].groupby("date")["target"].mean()

portfolio = pd.concat([long_ret, short_ret], axis=1)
portfolio.columns = ["long_ret", "short_ret"]

#daily long-short portfolio return.
# So if: long basket makes +1%; short basket makes -0.5% ; then strategy return is: 1%−(−0.5%)=1.5%
portfolio["strategy_return"] = portfolio["long_ret"] - portfolio["short_ret"]
portfolio = portfolio.dropna()

# This converts daily returns into cumulative wealth.
portfolio["equity_curve"] = (1 + portfolio["strategy_return"]).cumprod()

# -----------------------------
# 6. Build SPY benchmark
# -----------------------------
spy = yf.download("SPY", start="2015-01-01", auto_adjust=True)["Close"]
spy_ret = spy.pct_change()
spy_curve = (1 + spy_ret).cumprod()

# align dates
plot_df = pd.concat([portfolio["equity_curve"], spy_curve], axis=1, join="inner")
plot_df.columns = ["Strategy", "SPY"]
plot_df = plot_df / plot_df.iloc[0]  #Divides both curves by their starting value so both begin at 1.

# -----------------------------
# 7. Plot
# -----------------------------
plt.figure(figsize=(10, 5))
plot_df["Strategy"].plot(label="Strategy")
plot_df["SPY"].plot(label="SPY")
plt.legend()
plt.title("Walk-Forward Long/Short Strategy vs SPY")
plt.ylabel("Growth of $1")
plt.show()

print("Long mean return:", portfolio["long_ret"].mean())
print("Short mean return:", portfolio["short_ret"].mean())
print("Strategy mean:", portfolio["strategy_return"].mean())