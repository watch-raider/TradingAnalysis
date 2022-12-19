import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import streamlit as st
from datetime import date
from dateutil.relativedelta import relativedelta
import mwclean
import mwtechnical
import mwml
import mwerror

plt.style.use('seaborn-whitegrid')

symbols = ['SPY']#, 'TWTR', 'FB', 'MSFT', 'GOOGL','AMZN','CRM','NVDA', 'AAPL', 'DPZ']
spy_symbol = "SPY"# TWTR FB MSFT GOOGL AMZN CRM NVDA AAPL DPZ"
months = 3

def main():
    # streamlit setup
    st.title('Waltomic Trading')

    # Get user input for technical analysis parameters
    st.sidebar.header('Technical Analysis')
    months_before = date.today() - relativedelta(months=months)
    min_months_before = date.today() - relativedelta(weeks=6)
    day_ahead = date.today() + relativedelta(days=1)
    start_date = st.sidebar.date_input("Start date", months_before, max_value=min_months_before)
    end_date = st.sidebar.date_input("End date", day_ahead)
    stock_symbol = st.sidebar.text_input('Stock symbol', 'MSFT')
    stock_symbol = stock_symbol.strip().upper()
    n_day_window = st.sidebar.slider('Look back period (Days)', 2, 10, 5)

    # Machine learning inputs
    st.sidebar.header('Machince Learning')
    # Get user input for machine learning parameters
    n_day_future = st.sidebar.slider('Days into the future', 1, 10, 3)
    # Get user input for KNN parameters
    st.sidebar.subheader('Nearest neighbor')
    k = st.sidebar.slider('K', 1, 10, 3)

    # Retrieve and display fundamental factors
    # st.header('Fundamental Analysis')
    # stock = yf.Ticker(stock_symbol)
    # stock_info = stock.info
    # column_1, column_2 = st.columns(2)
    # with column_1:
    #     st.write('PEG ratio: ' + str(stock_info['pegRatio']))
    #     st.write('Trailing EPS: ' + str(stock_info['trailingEps']))
    #     #st.write('Trailing PE: ' + str(stock_info['trailingPE']))
    #     st.write('Market Cap: ' + str(stock_info['marketCap']))
    # with column_2:
    #     st.write('Book value per share: ' + str(stock_info['bookValue']))
    #     st.write('Forward EPS: ' + str(stock_info['forwardEps']))
    #     st.write('Forward PE: ' + str(stock_info['forwardPE']))
    #     st.write('Dividend Yield: ' + str(stock_info['dividendYield']))

    symbols.append(stock_symbol)
    symbol_txt = spy_symbol + " " + stock_symbol

    # Get data
    all_stock_data = yf.download(symbol_txt, start=start_date, end=end_date, group_by="ticker")

    # Read data
    dates = pd.date_range(start_date, end_date)  # one month only
    df_data = get_data(all_stock_data, dates)
    df_filled_data = mwclean.fill_missing_values(df_data)
    df_stock_data = df_filled_data.copy()
    df_stock_data = df_stock_data.drop(columns=[spy_symbol])

    # Technical Analysis
    df_price_technical_analysis, price_change_data, df_indicators = mwtechnical.calculate_indicators_v2(df_stock_data.copy(), n_day_window, stock_symbol)
    df_graph_normalised = mwtechnical.normalise_values(df_indicators) #create_technical_analysis_normalised_table(stock_symbol, df_technical_analysis)
    df_normalised = df_graph_normalised.copy()
    df_normalised_pretty = df_graph_normalised.copy()
    df_normalised_pretty.insert(0, 'Price_change_%', price_change_data * 100)

    # Display Technical Analysis
    st.header('Technical Analysis')
    
    left_column, right_column = st.columns(2)

    fig, ax = plt.subplots()
    ax.plot(df_price_technical_analysis.index, df_price_technical_analysis[stock_symbol].values, label=stock_symbol)
    ax.plot(df_price_technical_analysis.index, df_price_technical_analysis['SMA'].values, label='SMA')
    ax.plot(df_price_technical_analysis.index, df_price_technical_analysis['Upper_Bollinger_Band'].values, label='Upper BB')
    ax.plot(df_price_technical_analysis.index, df_price_technical_analysis['Lower_Bollinger_Band'].values, label='Lower BB')
    ax.set_title("Stock Price Technical Analysis")
    ax.set_xlabel('Date')
    ax.set_ylabel('Price ($)')
    ax.legend(['Price', 'SMA', 'Upper Bollinger Band', 'Lower Bollinger Band'], loc='best')
    fig.autofmt_xdate()
    
    fig2, ax2 = plt.subplots()
    ax2.plot(df_graph_normalised.index, df_graph_normalised['SMA'].values, label='SMA')
    ax2.plot(df_graph_normalised.index, df_graph_normalised['Bollinger_Bands'].values, label='BBs')
    ax2.plot(df_graph_normalised.index, df_graph_normalised['Momentum'].values, label='Momentum')
    ax2.plot(df_graph_normalised.index, df_graph_normalised['Volume'].values, label='Volume')
    ax2.set_title("Normalised Technical Analysis")
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Normalised Value')
    ax2.legend(['Price', 'SMA', 'Momentum', 'Bollinger Bands', 'Volume'],loc='best')
    fig2.autofmt_xdate()
    
    with left_column:
        st.pyplot(fig)
        df_price_technical_analysis.iloc[::-1]

    with right_column:
        st.pyplot(fig2)
        df_normalised_pretty.iloc[::-1]

    # ML methods
    price_data = np.asarray(df_stock_data[stock_symbol].values)
    recent_price = price_data[len(price_data) - 1]
    future_prices = np.empty(n_day_future)
    rmses = []
    prs = []
    df_normalised.insert(0, 'Price', df_price_technical_analysis[stock_symbol].values)

    # Loop for ML methods for each day in future
    for i, day in enumerate(range(1, n_day_future + 1)):
        x_train, y_train, x_test, y_test, start_prices, end_prices = mwml.train_test_split(df_normalised, day)

        #continue
        ### Review from here
        lr_predicted_normed_prices = mwml.linear_reg(x_train, y_train, x_test, y_test)
        knn_predicted_normed_prices = mwml.knn(x_train, y_train, x_test, y_test, k)

        #y_test_prices = price_data[-len(y_test):]

        # number of learning models e.g. Linear Regressions, KNN...
        num_of_models = 2
        predictions = np.empty(shape=(num_of_models, len(y_test)))
        predictions[0] = lr_predicted_normed_prices#mwtechnical.reverse_price_change(lr_predicted_normed_prices, start_prices) #reverse_normalisation(lr_predicted_normed_prices, y_test_prices)
        predictions[1] = knn_predicted_normed_prices#mwtechnical.reverse_price_change(knn_predicted_normed_prices, start_prices)#reverse_normalisation(knn_predicted_normed_prices, y_test_prices)
        combined_predictions = np.mean(predictions, axis=0)
        reversed_predictions = mwtechnical.reverse_price_change(combined_predictions, start_prices)

        # calculate prediction accuracy
        rmse, pr = mwerror.calculate_accuracy(reversed_predictions, end_prices, day)
        rmses.append(rmse)
        prs.append(pr)

        future_prices[i] = reversed_predictions[len(reversed_predictions) - 1]
    
    #return
    future_diffs = future_prices - recent_price

    future_data = {
        "Price Change": future_diffs
    }

    df_ml = pd.DataFrame(data=future_data)
    df_ml['Day(s) into future'] = range(1,len(future_prices)+1)
    df_ml = df_ml.set_index('Day(s) into future')

    #Display ML analysis
    st.header('Machine Learning Analysis')

    st.subheader("Machine Learning Predictions")
    st.bar_chart(df_ml)

    left_column2, middle_column2, right_column2 = st.columns(3)
    for i, rmse in enumerate(rmses):
        with left_column2:
            st.write("Day " + str(i + 1) + ":")
        with middle_column2:
            st.write("RMSE = " + str(rmse))
        with right_column2:
            st.write("Pearson's r = " + str(prs[i]))
 
def get_data(all_stock_data, dates):
    df = pd.DataFrame(index=dates)

    for symbol in symbols:
        stock_data = all_stock_data[symbol]
        if symbol == 'SPY':
            stock_data = stock_data[['Adj Close']]
        else :
            stock_data = stock_data[['Adj Close', 'Volume']]
        df_temp = stock_data.rename(columns={'Adj Close': symbol})
        df = df.join(df_temp)
        if symbol == 'SPY':  # drop dates GOOGLE did not trade
            df = df.dropna(subset=["SPY"])
        
    return df

main()