# GBM Quant Project

This project estimates the probability that a stock price will increase by at least 20% within one year under a Geometric Brownian Motion (GBM) model.

## Features

- Downloads historical stock data using `yfinance`
- Estimates annualized drift and volatility
- Computes the closed-form GBM probability
- Validates the result with Monte Carlo simulation
- Builds a 95% confidence interval for the Monte Carlo estimate
- Visualizes simulated price paths and terminal price distribution

## Project Structure

gbm-quant-project/
- data/
- models/
- analysis/
- visualization/
- main.py
- requirements.txt
- README.md

## How to Run

```bash
pip install -r requirements.txt
python main.py
