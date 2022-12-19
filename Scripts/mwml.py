import numpy as np
import pandas as pd

def train_test_split(all_raw_data, day_in_advance):
    raw_data = all_raw_data.copy()
    aligned_data = align_inputs_outputs(raw_data, day_in_advance)
    end_data = aligned_data[-day_in_advance:]
    randomised_data = aligned_data[:-day_in_advance].sample(frac = 1) 

    n = len(randomised_data)
    n_size = int(n / 4)

    train_dataset = randomised_data[0 : n - n_size]
    test_dataset = randomised_data[n - n_size : n]
    test_dataset = test_dataset.append(end_data)

    x_train, y_train, start_train_prices, end_train_prices = separate_input_output(train_dataset, day_in_advance)
    x_test, y_test, start_test_prices, end_test_prices = separate_input_output(test_dataset, day_in_advance)

    return x_train, y_train, x_test, y_test, start_test_prices, end_test_prices

def align_inputs_outputs(data, day_in_advance):
    #sma_vals, bb_vals, mom_vals, vol_vals = np.asarray(data['SMA']), np.asarray(data['Bollinger_Bands']), np.asarray(data['Momentum']), np.asarray(data['Volume'])
    #price_change_vals = np.asarray(data['Price_change_%'])
    price_vals = np.asarray(data['Price'])

    empty_arr = np.zeros(day_in_advance)
    #temp_price_changes = np.concatenate([price_change_vals, empty_arr])
    temp_prices = np.concatenate([price_vals, empty_arr])

    temp_prices_v2 = np.concatenate([empty_arr, price_vals])
    #shifted_prices_changes = temp_price_changes[day_in_advance:]
    shifted_prices = temp_prices[day_in_advance:]
    shifted_prices_v2 = temp_prices_v2[:-day_in_advance]
    shifted_price_change = (shifted_prices - price_vals) / price_vals

    #data['Price_change_%'] = shifted_prices_changes
    #data['Prediction_Price'] = shifted_prices
    data.insert(1, 'ahead_Price', shifted_prices)
    data.insert(1, 'before_Price', shifted_prices_v2)
    data.insert(1, 'Price_Y', shifted_price_change)
    # remove data that can't be used due to shift
    return data[day_in_advance:]

def separate_input_output(data, day_in_advance):
    xs, ys, start_prices, end_prices = np.asarray(data[['SMA', 'Bollinger_Bands', 'Momentum', 'Volume']]), np.asarray(data['Price_Y']), np.asarray(data['Price']), np.asarray(data['ahead_Price']) 

    return xs, ys, start_prices, end_prices

# linear regression
def linear_reg(x_train, y_train, x_test, y_test):
    c = np.linalg.lstsq(x_train, y_train)[0]

    test_predictions = (x_test @ c) 
    
    return test_predictions

def knn(x_train, y_train, x_test, y_test, k) :
    y_predictions = np.asarray(y_test)
    nearest_xs = np.empty(shape=(len(y_test), len(x_train[0])))
    ys = np.empty(shape=(len(y_test), k))

    # process each of the test data points
    for i, test_item in enumerate(x_test):
        # calculate the distances to all training points
        distances = [dist(train_item, test_item) for train_item in x_train]

        # add your code here
        sorted_distances = np.sort(distances)
        nearest = sorted_distances[:k]
        for ii, d in enumerate(nearest) :
            index = distances.index(d)
            ys[i] = y_train[index]
            nearest_xs[i] = x_train[index]

        y_predictions[i] = np.mean(ys[i])

    return y_predictions

# distance function
def dist(a, b):
    sum = 0
    for ai, bi in zip(a, b):
        sum = sum + (ai - bi)**2
    return np.sqrt(sum)