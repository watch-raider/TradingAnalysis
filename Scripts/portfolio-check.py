from datetime import date
import os
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import math
import scipy.optimize as spo

symbols = ['SPY']

def test_run():
    # Get user input for parameters
    print("Enter start date (YYYY-MM-DD):")
    start_date = input().strip()
    print("Enter end date (YYYY-MM-DD):")
    end_date = input().strip()
    print("Enter value of portfolio:")
    portfolio_val = float(input().strip())
    print("Enter desired max allocation for any individual stock:")
    max_alloc = float(input().strip())

    # Get portfolio details
    print('\n')
    print('### Your Portfolio ###')
    stock_symbol = ''
    symbol_txt = 'SPY'
    initial_allocs = []
    alloc_total = 0
    bnds_list = list()
    while alloc_total < 1.0:
        print("Enter stock symbol:")
        stock_symbol = input().strip().upper()
        if stock_symbol != "STOP":
            print("Enter stock allocation for " + stock_symbol + " (Value between 0 - 1.0):")
            alloc = float(input().strip())

            symbols.append(stock_symbol)
            symbol_txt += " " + stock_symbol

            initial_allocs.append(alloc)
            alloc_total += alloc
            bnds_list.append((0.01, max_alloc))

    bnds = tuple(bnds_list)

    # Get data
    all_stock_data = yf.download(symbol_txt, start=start_date, end=end_date, group_by="ticker")

    # Read data
    dates = pd.date_range(start_date, end_date)  # one month only
    df_data = get_data(all_stock_data, dates)
    df_filled_data = fill_missing_values(df_data)
    df_normed = normalise_prices(df_filled_data)
    df_alloc = alloc_normalise(df_normed, initial_allocs)
    df_pos_vals = postion_alloc(df_alloc, portfolio_val)
    
    df_port_vals = portfolio_postion(df_pos_vals)
    df_port_vals = df_port_vals.rename({'port_val':'portfolio_val'}, axis='columns')
    
    df_daily_returns = compute_daily_returns(df_port_vals)
    df_daily_returns = df_daily_returns.rename({'portfolio_val':'daily_returns'}, axis='columns')
    daily_rets = df_daily_returns[1:]

    cum_ret = (df_port_vals['portfolio_val'][-1] / df_port_vals['portfolio_val'][0]) - 1
    avg_daily_ret = daily_rets['daily_returns'].mean()
    std_daily_ret = daily_rets['daily_returns'].std()
    sharpe_ratio = math.sqrt(252) * (avg_daily_ret / std_daily_ret)

    print('Cumulative returns:')
    print(cum_ret)
    print('Average daily returns:')
    print(avg_daily_ret)
    print('Standard deviation daily returns:')
    print(std_daily_ret)
    print('Sharpe ratio:')
    print(sharpe_ratio)
    print('\n')

    print('Optimised Allocations:')
    optimized_allocs = maximise_sharpe_ratio(initial_allocs, df_normed, portfolio_val, bnds)
    print(optimized_allocs)

def normalise_prices(df):
    df_temp = df.copy()
    for symbol in symbols:
        df_temp[symbol] = df[symbol] / df[symbol][0]

    return df_temp

def alloc_normalise(df, allocs):
    df_temp = df.copy()
    count = 0
    for symbol in symbols:
        df_temp[symbol] = df[symbol] * allocs[count]
        count += 1

    return df_temp

def postion_alloc(df, portfolio_val):
    df_temp = df.copy()
    for symbol in symbols:
        df_temp[symbol] = df[symbol] * portfolio_val

    return df_temp

def portfolio_postion(df):
    df_temp = pd.DataFrame(index=df.index)
    #df_temp = df.copy()
    for symbol in symbols:
        df_temp['port_val'] = df.sum(axis=1)

    return df_temp

def fill_missing_values(df_data):
    """Fill missing values in data frame, in place."""
    df_data.fillna(method='ffill', inplace=True)
    df_data.fillna(method='backfill', inplace=True)
    return df_data


def get_data(all_stock_data, dates):
    df = pd.DataFrame(index=dates)

    for symbol in symbols:
        stock_data = all_stock_data[symbol]
        stock_data = stock_data[['Close']]
        df_temp = stock_data.rename(columns={'Close': symbol})
        df = df.join(df_temp)
        if symbol == 'SPY':  # drop dates GOOGLE did not trade
            df = df.dropna(subset=["SPY"])

    del df['SPY']
    symbols.remove("SPY")

    return df

def compute_daily_returns(df):
    df_copy = df.copy()
    df_copy[1:] = (df[1:] / df[:-1].values) - 1
    df_copy[:1] = 0
    return df_copy

def maximise_sharpe_ratio(initial_allocs, df_normed, portfolio_val, bnds):
    Xguess = initial_allocs
    cons = {'type':'eq', 'fun': con}
    result = spo.minimize(f, Xguess, args=(df_normed, portfolio_val), method='SLSQP',bounds=bnds, constraints=cons, options={'disp': True})
    return result.x

def f(allocs, df_normed, portfolio_val):
    sharpe_ratio =  calculate_sharpe_ratio(allocs, df_normed, portfolio_val) * -1
    return sharpe_ratio

def con(x):
    return sum(x) - 1

def calculate_sharpe_ratio(allocs, df_normed, portfolio_val):
    df_alloc = alloc_normalise(df_normed, allocs)
    df_pos_vals = postion_alloc(df_alloc, portfolio_val)
    
    df_port_vals = portfolio_postion(df_pos_vals)
    df_port_vals = df_port_vals.rename({'port_val':'portfolio_val'}, axis='columns')
    
    df_daily_returns = compute_daily_returns(df_port_vals)
    df_daily_returns = df_daily_returns.rename({'portfolio_val':'daily_returns'}, axis='columns')
    daily_rets = df_daily_returns[1:]

    avg_daily_ret = daily_rets['daily_returns'].mean()
    std_daily_ret = daily_rets['daily_returns'].std()
    sharpe_ratio = math.sqrt(252) * (avg_daily_ret / std_daily_ret)
    return sharpe_ratio


if __name__ == "__main__":
    test_run()