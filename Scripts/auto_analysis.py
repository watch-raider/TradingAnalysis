import pandas as pd
import numpy as np
import yfinance as yf
from datetime import date
from dateutil.relativedelta import relativedelta
import smtplib
import mwclean
import mwtechnical
import mwml

symbols = ['MSFT', 'FB', 'TWTR', 'GOOGL','AMZN','NVDA', 'AAPL']#, 'ATVI', 'AMD', 'NFLX', 'DIS', 'PYPL', 'V']
tickers = "SPY MSFT FB TWTR GOOGL AMZN NVDA AAPL"# ATVI AMD NFLX DIS PYPL V"
spy_symbol = "SPY"
months = [6, 12, 36]
num_of_models = 2

def main():
    months_before = date.today() - relativedelta(months=months[len(months)-1])
    day_ahead = date.today() + relativedelta(days=1)
    start_date = months_before
    end_date = day_ahead
    n_day_window = 5

    # Machine learning inputs
    # Get user input for machine learning parameters
    n_day_future = 3
    # Get user input for KNN parameters
    k = 3

    # Get data
    all_stock_data = yf.download(tickers, start=start_date, end=end_date)

    # Read data
    dates = pd.date_range(start_date, end_date)  # one month only
    df_data = get_data(all_stock_data, dates)
    df_filled_data = mwclean.fill_missing_values(df_data)
    df_all_stock_data = df_filled_data.copy()
    df_all_stock_data = df_all_stock_data.drop(columns=[spy_symbol])
    prediction_txt = """"""

    # Technical Analysis
    for symbol in symbols:
        recent_price = round(df_all_stock_data[symbol][len(df_all_stock_data) - 1], 2)
        prediction_txt += symbol + ' - ' + str(recent_price) + '\n'
        for period in months:
            starting_date = date.today() - relativedelta(months=period)
            period_data = df_all_stock_data[starting_date:day_ahead]
            period_data.dropna(inplace=True)
            df_stock_data = pd.DataFrame({
                symbol: period_data[symbol].values,
                'Volume': period_data[symbol + '_volume'].values
            }, period_data.index)

            df_price_technical_analysis, price_change_data,  df_indicators = mwtechnical.calculate_indicators_v2(df_stock_data, n_day_window, symbol)
            df_normalised = mwtechnical.normalise_values(df_indicators)
            #df_normalised.insert(0, 'Price_change_%', price_change_data)
            df_normalised.insert(0, 'Price', df_price_technical_analysis[symbol].values)
            df_normalised.dropna(inplace=True)

            # ML methods
            price_data = np.asarray(df_stock_data[symbol].values)
            recent_price = price_data[len(price_data) - 1]
            future_prices = np.empty(n_day_future)

            # Loop for ML methods for each day in future
            for i, day in enumerate(range(1, n_day_future + 1)):
                x_train, y_train, x_test, y_test, start_prices, end_prices = mwml.train_test_split(df_normalised, day)

                ### Review from here
                lr_predicted_normed_prices = mwml.linear_reg(x_train, y_train, x_test, y_test)
                knn_predicted_normed_prices = mwml.knn(x_train, y_train, x_test, y_test, k)

                y_test_prices = price_data[-len(y_test):]

                # number of learning models e.g. Linear Regressions, KNN...
                predictions = np.empty(shape=(num_of_models, len(y_test)))
                predictions[0] = lr_predicted_normed_prices#mwtechnical.reverse_price_change(lr_predicted_normed_prices, start_prices) #reverse_normalisation(lr_predicted_normed_prices, y_test_prices)
                predictions[1] = knn_predicted_normed_prices#mwtechnical.reverse_price_change(knn_predicted_normed_prices, start_prices)#reverse_normalisation(knn_predicted_normed_prices, y_test_prices)
                combined_predictions = np.mean(predictions, axis=0)
                reversed_predictions = mwtechnical.reverse_price_change(combined_predictions, start_prices)

                future_prices[i] = reversed_predictions[len(reversed_predictions) - 1]
            
            future_diffs = future_prices - recent_price
            future_diffs = np.round((future_diffs / recent_price) * 100, 2)
            prediction_txt += str(period) + " mos -    " + str(future_diffs) + " %\n"
        
        prediction_txt = prediction_txt + '\n'

    gmail_user = '07waltonmi@gmail.com'
    gmail_password = 'giympoevgxdsabva'

    #email properties
    sent_from = gmail_user
    to = [gmail_user]
    email_text = prediction_txt.replace('[', "").replace(']', "")

    #email send request
    try:
        server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
        server.ehlo()
        server.login(gmail_user, gmail_password)
        server.sendmail(sent_from, to, email_text)
        server.close()

        print ('Email sent!')
    except Exception as e:
        print(e)
        print ('Something went wrong...')

def get_data(all_stock_data, dates):

    adj_close_data = all_stock_data['Adj Close']
    volume_data = all_stock_data['Volume']

    df = pd.DataFrame({
            "SPY": adj_close_data['SPY']
        }, index=dates)

    df = df.dropna(subset=["SPY"])

    stock_adj_close_data = []
    stock_volume_data = []

    for symbol in symbols:
        if symbol != 'SPY':
            stock_adj_close_data = adj_close_data[symbol]
            stock_volume_data = volume_data[symbol]
            df[symbol] = stock_adj_close_data
            df[symbol + '_volume'] = stock_volume_data
    
    return df

main()