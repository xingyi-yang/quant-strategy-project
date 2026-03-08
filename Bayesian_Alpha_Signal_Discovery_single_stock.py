import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

data = yf.download("AAPL", start="2015-01-01")
print(data.head())

# daily return
data["return"] = data["Close"].pct_change()

# momentum return
data["momentum"] = data["Close"].pct_change(20)

# standard deviation of the last 20 returns
data["volatility"] = data["return"].rolling(20).std()

# the return over the next 20 days
data["target"] = data["Close"].pct_change(20).shift(-20)

# check the correlation
data[["momentum","volatility","target"]].corr()

# check scatter plot
plt.scatter(data["momentum"], data["target"], alpha = 0.3)
plt.xlabel("momentum")
plt.ylabel("future return")
plt.show()

plt.scatter(data["return"], data["target"], alpha = 0.3)
plt.xlabel("return")
plt.ylabel("future return")
plt.show()
plt.show()

# print(data.tail(50))
# simple linear regression
data = data[["momentum", "volatility", "target"]].dropna()
X = data[["momentum","volatility"]]
y = data["target"]

model = LinearRegression()
model.fit(X, y)
print(model.coef_)

data["prediction"] = model.predict(X)

data["signal"] = data["prediction"].apply(
    lambda x: 1 if x > 0.01 else (-1 if x < -0.01 else 0)
)

#strategy return: If you followed the model's trading signal, how much you would gain or lose.
data["strategy_return"] = data["signal"] * data["target"]

# Build cumulative performance
# portfolio grows over time.
# It converts individual strategy returns into total accumulated wealth.
# cumprod() calculates cumulative products
data["equity_curve"] = (1 + data["strategy_return"]).cumprod()
data["equity_curve"].plot(figsize=(10,5))
plt.title("Strategy Equity Curve")
plt.show()

#average over volatility: How much reward you get for each unit of risk.
sharpe = data["strategy_return"].mean() / data["strategy_return"].std()
print(sharpe)