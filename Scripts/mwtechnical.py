import numpy as np
import pandas as pd

def calculate_indicators_v2(df, window, stock_symbol) :
    stock_price_data, volume_data = np.array(df[stock_symbol].values), np.array(df['Volume'])
    day_ahead_prices, day_before_prices = stock_price_data[1:], stock_price_data[:-1]
    n_day_ahead_prices, n_day_before_prices = stock_price_data[window:], stock_price_data[:-window]
    day_ahead_volume, day_before_volume = volume_data[1:], volume_data[:-1]

    n = len(n_day_ahead_prices)
    sma_vals, bb_vals, upper_bbs, lower_bbs, smas = np.empty(n), np.empty(n), np.empty(n), np.empty(n), np.empty(n)

    # calculate % changes in price = p[t] - p[t-1] / p[t-1] 
    price_changes = ((day_ahead_prices - day_before_prices) / day_before_prices)
    # calcualte Momentum values = p[t] / price[t - n] - 1
    m_vals = (n_day_ahead_prices / n_day_before_prices) - 1
    # calculate % change in volume = v[t] - v[t-1] / v[t-1]
    v_vals = ((day_ahead_volume - day_before_volume) / day_before_volume)

    count = 0
    # Loop through data to calculate SMA and BB values
    for i, val in enumerate(sma_vals, window):
        window_avg = stock_price_data[i-window:i+1].mean()
        window_std = stock_price_data[i-window:i+1].std()
        two_stds = window_std * 2
        # calculate SMA = price[t] / price[t-n:t].mean() - 1 
        sma_vals[count] = (stock_price_data[i] / window_avg) - 1
        # calcualte BB = price[t] - SMA[t] / 2 * std[t] 
        bb_vals[count] = (stock_price_data[i] - window_avg) / (two_stds)

        # actual values for indicators
        smas[count] = window_avg
        upper_bbs[count] = window_avg + (two_stds)
        lower_bbs[count] = window_avg - (two_stds)

        count += 1

    df_cut = df[-n:]
    price_change_data = price_changes[-n:]

    df_price_technical_analysis = pd.DataFrame({
            stock_symbol: np.round(n_day_ahead_prices, 2),
            'SMA': np.round(smas, 2),
            'Upper_Bollinger_Band': np.round(upper_bbs, 2),
            'Lower_Bollinger_Band': np.round(lower_bbs, 2)
        }, index=df_cut.index)

    df_indicators = pd.DataFrame({
            'SMA': sma_vals,
            'Bollinger_Bands': bb_vals,
            'Momentum': m_vals,
            'Volume': v_vals[-n:]
        }, index=df_cut.index)

    return df_price_technical_analysis, price_change_data, df_indicators

def normalise_values(values):
    return (values - values.mean()) / values.std()

def reverse_price_change(predicted_price_changes, prices):
    return prices + (prices * predicted_price_changes)