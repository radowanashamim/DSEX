# %%
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# %% [markdown]
# In this Notebook we'll just implement the Deepl learning portion of the Stock Market prediction project

# %% [markdown]
# **Loading data into data frames**

# %%
df1 = pd.read_json('C:/Users/User/Documents/DSEX/data/prices_2008.json')
df2 = pd.read_json('C:/Users/User/Documents/DSEX/data/prices_2009.json')
df3 = pd.read_json('C:/Users/User/Documents/DSEX/data/prices_2010.json')
df4 = pd.read_json('C:/Users/User/Documents/DSEX/data/prices_2011.json')
df5 = pd.read_json('C:/Users/User/Documents/DSEX/data/prices_2012.json')
df6 = pd.read_json('C:/Users/User/Documents/DSEX/data/prices_2013.json')
df7 = pd.read_json('C:/Users/User/Documents/DSEX/data/prices_2014.json')
df8 = pd.read_json('C:/Users/User/Documents/DSEX/data/prices_2015.json')
df9 = pd.read_json('C:/Users/User/Documents/DSEX/data/prices_2016.json')
df10 = pd.read_json('C:/Users/User/Documents/DSEX/data/prices_2017.json')
df11 = pd.read_json('C:/Users/User/Documents/DSEX/data/prices_2018.json')
df12 = pd.read_json('C:/Users/User/Documents/DSEX/data/prices_2019.json')
df13 = pd.read_json('C:/Users/User/Documents/DSEX/data/prices_2020.json')
df14 = pd.read_json('C:/Users/User/Documents/DSEX/data/prices_2021.json')
df15 = pd.read_json('C:/Users/User/Documents/DSEX/data/prices_2022.json')

# %%
df1 = df1.iloc[::-1]
df2 = df2.iloc[::-1]
df3 = df3.iloc[::-1]
df4 = df4.iloc[::-1]
df5 = df5.iloc[::-1]
df6 = df6.iloc[::-1]
df7 = df7.iloc[::-1]
df8 = df8.iloc[::-1]
df9 = df9.iloc[::-1]
df10 = df10.iloc[::-1]
df11 = df11.iloc[::-1]
df12 = df12.iloc[::-1]
df13 = df13.iloc[::-1]
df14 = df14.iloc[::-1]
df15 = df15.iloc[::-1]

# %% [markdown]
# **Combining into one whole list**

# %%
lst=[df1, df2, df3, df4, df5, df6, df7,df8,df9,df10,df11,df12,df13,df14, df15]
df=pd.concat(lst)

# %%
df=df.set_index('date')

# %%
df_without_trading_code = df.drop(columns=['trading_code'])
correlation_with_opening_price = df_without_trading_code.corr()['opening_price']
print(correlation_with_opening_price)

# %% [markdown]
# *Here we can see that opening price has a high amount of correlation wuth (Closing price& Yesterday's closing price)
# Opening price also has slight correlation with (trade, value_nm & colume)
# Lastlt, it has negligible correlation with the rest*.
# **To make the model be efficient, we'll use the first five columns for our predictions**

# %% [markdown]
# **Preprocessing the data**

# %%
df.dropna(inplace=True)

# %%
df=df.drop(['last_traded_price', 'high', 'low'], axis=1, inplace=False)

# %%
df = df[df['opening_price'] != 0]

# %%
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
columns_to_scale = ['yesterdays_closing_price', 'value_mn', 'opening_price', 'closing_price', 'trade', 'volume']

df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])

# %% [markdown]
# Getting the company data of companies with the largest market cap

# %%
GP=df.loc[df['trading_code'].isin(['GP'])]
BATBC=df.loc[df['trading_code'].isin(['BATBC'])]
WALTONHIL=df.loc[df['trading_code'].isin(['WALTONHIL'])]
SQURPHARMA=df.loc[df['trading_code'].isin(['SQURPHARMA'])]
ROBI=df.loc[df['trading_code'].isin(['ROBI'])]
RENATA=df.loc[df['trading_code'].isin(['RENATA'])]
BEXIMCO=df.loc[df['trading_code'].isin(['BEXIMCO'])]
UPGDCL=df.loc[df['trading_code'].isin(['UPGDCL'])]
BERGERPBL=df.loc[df['trading_code'].isin(['BERGERPBL'])]
LHBL=df.loc[df['trading_code'].isin(['LHBL'])]

# %% [markdown]
# **Data visualization for GP**

# %%
import matplotlib.pyplot as plt

plt.figure(figsize=(5, 3))
plt.plot(GP.index, GP['opening_price'], marker='o', markersize=1, linestyle='-')  # Adjust markersize as needed
plt.title('Opening Price Over Time')
plt.xlabel('Date')
plt.ylabel('Opening Price')
plt.grid(True)
plt.tight_layout()
plt.show()

# %%
# Define the features and target variable
features = ['closing_price', 'yesterdays_closing_price', 'trade', 'value_mn', 'volume']
X = GP[features].values
y = GP['opening_price'].values

# %%
# Define the sizes for training, validation, and testing sets
train_size = int(0.8 * len(X))  # 80% for training
val_size = int(0.1 * len(X))    # 10% for validation
test_size = len(X) - train_size - val_size  # Remaining 10% for testing

# %%
# Split the features and target variable accordingly
X_train = X[:train_size]
X_val = X[train_size:train_size+val_size]
X_test = X[train_size+val_size:]

# %%
y_train = y[:train_size]
y_val = y[train_size:train_size+val_size]
y_test = y[train_size+val_size:]

# %%
# Reshape the data for LSTM input (assuming a time step of 1)
X_train_reshaped = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
X_val_reshaped = X_val.reshape(X_val.shape[0], 1, X_val.shape[1])
X_test_reshaped = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

# %%
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.optimizers import Adam
import numpy as np

# Define the LSTM model
model = Sequential([
    LSTM(units=50, input_shape=(X_train_reshaped.shape[1], X_train_reshaped.shape[2])),
    Dense(units=1)
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

# %%
# Train the model
history = model.fit(X_train_reshaped, y_train, epochs=50, batch_size=32, validation_data=(X_val_reshaped, y_val))

# Evaluate the model on the test set
loss = model.evaluate(X_test_reshaped, y_test)
print(f'Test Loss: {loss}')

# Calculate and print RMSE
y_pred = model.predict(X_test_reshaped)
rmse = np.sqrt(np.mean((y_pred - y_test)**2))
print(f'RMSE: {rmse}')

# Plot training history
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# %% [markdown]
# > **3 Day Forecast**

# %%
import numpy as np
import matplotlib.pyplot as plt

# Assuming X_scaled, model, and y are defined

# Get the last available data point
last_data_point = X_scaled[-1].reshape(1, 1, X_scaled.shape[1])

# Predict the next three days
forecast = []
for _ in range(3):
    next_day_prediction = model.predict(last_data_point)
    forecast.append(next_day_prediction)
    # Update last_data_point to include the predicted value for the next day
    last_data_point = np.concatenate([last_data_point, next_day_prediction.reshape(1, 1, 1)], axis=2)

# Extract opening prices from the forecast
forecast_opening_prices = np.array(forecast).reshape(-1)

print("Forecasted opening prices for the next three days:")
print(forecast_opening_prices)

# Number of original days
n_days_original = len(y)

# Plot forecasted opening prices for the next three days without connecting lines
forecast_days = np.arange(n_days_original, n_days_original + 3)
plt.plot(forecast_days, forecast_opening_prices, marker='o', linestyle='None', label='Forecasted Data')

plt.title('Three-Day Forecast')
plt.xlabel('Day')
plt.ylabel('Price')
plt.xticks(np.arange(n_days_original, n_days_original + 3), ['Day 1', 'Day 2', 'Day 3'])
plt.legend()
plt.show()


# %% [markdown]
# > **7 Day Forecast**

# %%
import numpy as np
import matplotlib.pyplot as plt

# Assuming X_scaled, model, and y are defined

# Get the last available data point
last_data_point = X_scaled[-1].reshape(1, 1, X_scaled.shape[1])

# Predict the next seven days
forecast = []
for _ in range(7):
    next_day_prediction = model.predict(last_data_point)
    forecast.append(next_day_prediction)
    # Update last_data_point to include the predicted value for the next day
    last_data_point = np.concatenate([last_data_point, next_day_prediction.reshape(1, 1, 1)], axis=2)

# Extract opening prices from the forecast
forecast_opening_prices = np.array(forecast).reshape(-1)

print("Forecasted opening prices for the next seven days:")
print(forecast_opening_prices)

# Plot forecasted opening prices for the next seven days without connecting lines
forecast_days = np.arange(n_days_original, n_days_original + 7)
plt.plot(forecast_days, forecast_opening_prices, marker='o', linestyle='None', label='Forecasted Data')

plt.title('Seven-Day Forecast')
plt.xlabel('Day')
plt.ylabel('Price')
plt.xticks(np.arange(n_days_original, n_days_original + 7), ['Day 1', 'Day 2', 'Day 3', 'Day 4', 'Day 5', 'Day 6', 'Day 7'])
plt.legend()
plt.show()


# %% [markdown]
# > **30 Day Forecast**

# %%
import numpy as np
import matplotlib.pyplot as plt

# Assuming X_scaled, model, and y are defined

# Get the last available data point
last_data_point = X_scaled[-1].reshape(1, 1, X_scaled.shape[1])

# Predict the next 30 days
forecast = []
for _ in range(30):
    next_day_prediction = model.predict(last_data_point)
    forecast.append(next_day_prediction)
    # Update last_data_point to include the predicted value for the next day
    last_data_point = np.concatenate([last_data_point, next_day_prediction.reshape(1, 1, 1)], axis=2)

# Extract opening prices from the forecast
forecast_opening_prices = np.array(forecast).reshape(-1)

print("Forecasted opening prices for the next 30 days:")
print(forecast_opening_prices)

# Plot forecasted opening prices for the next 30 days without connecting lines
forecast_days = np.arange(n_days_original, n_days_original + 30)
plt.plot(forecast_days, forecast_opening_prices, marker='o', linestyle='None', label='Forecasted Data')

plt.title('Thirty-Day Forecast')
plt.xlabel('Day')

# Labeling x-axis ticks for days 5, 10, 15, 20, 25, and 30
x_ticks = [5, 10, 15, 20, 25, 29]  # Corrected the indexing
plt.xticks(np.arange(n_days_original, n_days_original + 30)[x_ticks], [f'Day {i}' for i in x_ticks])

plt.ylabel('Price')
plt.legend()
plt.show()


# %% [markdown]
# **XAI**

# %%
import matplotlib.pyplot as plt
import numpy as np
import keras.backend as K
import tensorflow as tf

# Define a function to generate Grad-CAM
def generate_grad_cam(model, sequence, class_index):
    # Expand dimensions to make it compatible with model input shape
    sequence = np.expand_dims(sequence, axis=0)
    
    # Get the output of the last LSTM layer
    lstm_output = model.layers[0](sequence)
    
    # Get the output of the Dense layer
    dense_output = model.layers[1](lstm_output)
    
    # Compute the gradient of the class output with respect to the Dense layer output
    with tf.GradientTape() as tape:
        tape.watch(dense_output)
        loss = dense_output[:, class_index]
    grads = tape.gradient(loss, dense_output)
    
    # Get the weights of the Dense layer
    dense_weights = model.layers[1].get_weights()[0]
    
    # Compute the importance of each feature map
    importance = tf.reduce_mean(grads * dense_weights, axis=(0, 1))
    
    # Generate the heatmap
    heatmap = np.dot(importance, lstm_output.numpy().squeeze())
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    
    return heatmap

# Choose a sample from the test set
sample_index = 0  # Choose any sample index from the test set
sample_sequence = X_test_reshaped[sample_index]
sample_label = y_test[sample_index]

# Generate the Grad-CAM heatmap
heatmap = generate_grad_cam(model, sample_sequence, class_index=0)  # Assuming binary classification

# Plot the saliency map (heatmap)
plt.plot(heatmap)
plt.title('Saliency Map (Heatmap)')
plt.xlabel('Time Steps')
plt.ylabel('Importance')
plt.show()


# %%
import numpy as np
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

def permutation_importance(model, X, y, metric=mean_squared_error, n_iterations=100):
    baseline_score = metric(y, model.predict(X))
    feature_importances = {}

    for feature_idx in tqdm(range(X.shape[1]), desc='Calculating Permutation Importances'):
        shuffled_X = X.copy()
        np.random.shuffle(shuffled_X[:, feature_idx])
        permuted_score = 0
        for _ in range(n_iterations):
            permuted_score += metric(y, model.predict(shuffled_X))
        feature_importances[feature_idx] = baseline_score - permuted_score / n_iterations

    return feature_importances

# Calculate permutation importance
perm_importances = permutation_importance(model, X_test_reshaped, y_test)

# Print feature importances
for i, importance in sorted(perm_importances.items(), key=lambda x: x[1], reverse=True):
    print(f'Feature {i}: {importance}')


# %%
import numpy as np
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

def permutation_importance(model, X, y, metric=mean_squared_error, n_iterations=100):
    baseline_score = metric(y, model.predict(X))
    feature_importances = {}

    for feature_idx in tqdm(range(X.shape[2]), desc='Calculating Permutation Importances'):
        permuted_scores = []
        for _ in range(n_iterations):
            shuffled_X = X.copy()
            np.random.shuffle(shuffled_X[:, :, feature_idx])
            permuted_score = metric(y, model.predict(shuffled_X))
            permuted_scores.append(permuted_score)
        feature_importances[feature_idx] = baseline_score - np.mean(permuted_scores)

    return feature_importances

# Calculate permutation importance
perm_importances = permutation_importance(model, X_test_reshaped, y_test)

# Print feature importances
for i, importance in sorted(perm_importances.items(), key=lambda x: x[1], reverse=True):
    print(f'Feature {i}: {importance}')


# %%
import matplotlib.pyplot as plt

# Make predictions on the test set
y_pred = model.predict(X_test_reshaped)

# Plot actual vs predicted values
plt.figure(figsize=(10, 6))
plt.plot(y_test, label='Actual Opening Price', color='blue')
plt.plot(y_pred, label='Predicted Opening Price', color='red')
plt.title('Actual vs Predicted Opening Price')
plt.xlabel('Time')
plt.ylabel('Opening Price')
plt.legend()
plt.show()

# %% [markdown]
# **Data visualization for BATBC**

# %%
import matplotlib.pyplot as plt

plt.figure(figsize=(5, 3))
plt.plot(BATBC.index, BATBC['opening_price'], marker='o', markersize=1, linestyle='-')  # Adjust markersize as needed
plt.title('Opening Price Over Time')
plt.xlabel('Date')
plt.ylabel('Opening Price')
plt.grid(True)
plt.tight_layout()
plt.show()


# %%
from sklearn.preprocessing import MinMaxScaler

# Define the features and target variable
features = ['opening_price', 'closing_price', 'yesterdays_closing_price', 'trade', 'value_mn', 'volume']
X = BATBC[features].values
y = BATBC['opening_price'].values

# Scale the features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Define the sizes for training, validation, and testing sets
train_size = int(0.8 * len(X))  # 80% for training
val_size = int(0.1 * len(X))    # 10% for validation
test_size = len(X) - train_size - val_size  # Remaining 10% for testing

# Split the features and target variable accordingly
X_train = X_scaled[:train_size]
X_val = X_scaled[train_size:train_size+val_size]
X_test = X_scaled[train_size+val_size:]

y_train = y[:train_size]
y_val = y[train_size:train_size+val_size]
y_test = y[train_size+val_size:]

# Reshape the data for LSTM input (assuming a time step of 1)
X_train_reshaped = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
X_val_reshaped = X_val.reshape(X_val.shape[0], 1, X_val.shape[1])
X_test_reshaped = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

# Now you can use X_train_reshaped, y_train, X_val_reshaped, y_val, X_test_reshaped, and y_test for training and evaluating your LSTM model

# %%
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.optimizers import Adam
import numpy as np

# Define the LSTM model
model = Sequential([
    LSTM(units=50, input_shape=(X_train_reshaped.shape[1], X_train_reshaped.shape[2])),
    Dense(units=1)
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

# Train the model
history = model.fit(X_train_reshaped, y_train, epochs=50, batch_size=32, validation_data=(X_val_reshaped, y_val))

# Evaluate the model on the test set
loss = model.evaluate(X_test_reshaped, y_test)
print(f'Test Loss: {loss}')

# Calculate and print RMSE
y_pred = model.predict(X_test_reshaped)
rmse = np.sqrt(np.mean((y_pred - y_test)**2))
print(f'RMSE: {rmse}')

# Plot training history
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# %%
import matplotlib.pyplot as plt

# Make predictions on the test set
y_pred = model.predict(X_test_reshaped)

# Plot actual vs predicted values
plt.figure(figsize=(10, 6))
plt.plot(y_test, label='Actual Opening Price', color='blue')
plt.plot(y_pred, label='Predicted Opening Price', color='red')
plt.title('Actual vs Predicted Opening Price')
plt.xlabel('Time')
plt.ylabel('Opening Price')
plt.legend()
plt.show()


# %% [markdown]
# > **3 Day Forecast**

# %%
import numpy as np
import matplotlib.pyplot as plt

# Assuming X_scaled, model, and y are defined

# Get the last available data point
last_data_point = X_scaled[-1].reshape(1, 1, X_scaled.shape[1])

# Predict the next three days
forecast = []
for _ in range(3):
    next_day_prediction = model.predict(last_data_point)
    forecast.append(next_day_prediction)
    # Update last_data_point to include the predicted value for the next day
    last_data_point = np.concatenate([last_data_point, next_day_prediction.reshape(1, 1, 1)], axis=2)

# Extract opening prices from the forecast
forecast_opening_prices = np.array(forecast).reshape(-1)

print("Forecasted opening prices for the next three days:")
print(forecast_opening_prices)

# Number of original days
n_days_original = len(y)

# Plot forecasted opening prices for the next three days without connecting lines
forecast_days = np.arange(n_days_original, n_days_original + 3)
plt.plot(forecast_days, forecast_opening_prices, marker='o', linestyle='None', label='Forecasted Data')

plt.title('Three-Day Forecast')
plt.xlabel('Day')
plt.ylabel('Price')
plt.xticks(np.arange(n_days_original, n_days_original + 3), ['Day 1', 'Day 2', 'Day 3'])
plt.legend()
plt.show()


# %% [markdown]
# > **7 Day Forecast**

# %%
import numpy as np
import matplotlib.pyplot as plt

# Assuming X_scaled, model, and y are defined

# Get the last available data point
last_data_point = X_scaled[-1].reshape(1, 1, X_scaled.shape[1])

# Predict the next seven days
forecast = []
for _ in range(7):
    next_day_prediction = model.predict(last_data_point)
    forecast.append(next_day_prediction)
    # Update last_data_point to include the predicted value for the next day
    last_data_point = np.concatenate([last_data_point, next_day_prediction.reshape(1, 1, 1)], axis=2)

# Extract opening prices from the forecast
forecast_opening_prices = np.array(forecast).reshape(-1)

print("Forecasted opening prices for the next seven days:")
print(forecast_opening_prices)

# Plot forecasted opening prices for the next seven days without connecting lines
forecast_days = np.arange(n_days_original, n_days_original + 7)
plt.plot(forecast_days, forecast_opening_prices, marker='o', linestyle='None', label='Forecasted Data')

plt.title('Seven-Day Forecast')
plt.xlabel('Day')
plt.ylabel('Price')
plt.xticks(np.arange(n_days_original, n_days_original + 7), ['Day 1', 'Day 2', 'Day 3', 'Day 4', 'Day 5', 'Day 6', 'Day 7'])
plt.legend()
plt.show()


# %% [markdown]
# > **30 Day Forecast**

# %%
import numpy as np
import matplotlib.pyplot as plt

# Assuming X_scaled, model, and y are defined

# Get the last available data point
last_data_point = X_scaled[-1].reshape(1, 1, X_scaled.shape[1])

# Predict the next 30 days
forecast = []
for _ in range(30):
    next_day_prediction = model.predict(last_data_point)
    forecast.append(next_day_prediction)
    # Update last_data_point to include the predicted value for the next day
    last_data_point = np.concatenate([last_data_point, next_day_prediction.reshape(1, 1, 1)], axis=2)

# Extract opening prices from the forecast
forecast_opening_prices = np.array(forecast).reshape(-1)

print("Forecasted opening prices for the next 30 days:")
print(forecast_opening_prices)

# Plot forecasted opening prices for the next 30 days without connecting lines
forecast_days = np.arange(n_days_original, n_days_original + 30)
plt.plot(forecast_days, forecast_opening_prices, marker='o', linestyle='None', label='Forecasted Data')

plt.title('Thirty-Day Forecast')
plt.xlabel('Day')

# Labeling x-axis ticks for days 5, 10, 15, 20, 25, and 30
x_ticks = [5, 10, 15, 20, 25, 29]  # Corrected the indexing
plt.xticks(np.arange(n_days_original, n_days_original + 30)[x_ticks], [f'Day {i}' for i in x_ticks])

plt.ylabel('Price')
plt.legend()
plt.show()


# %%
import numpy as np
import matplotlib.pyplot as plt

# Assuming X_scaled, model, and y are defined

# Get the last available data point
last_data_point = X_scaled[-1].reshape(1, 1, X_scaled.shape[1])

# Predict the next seven days
forecast = []
for _ in range(7):
    next_day_prediction = model.predict(last_data_point)
    forecast.append(next_day_prediction)
    # Update last_data_point to include the predicted value for the next day
    last_data_point = np.concatenate([last_data_point, next_day_prediction.reshape(1, 1, 1)], axis=2)

# Extract opening prices from the forecast
forecast_opening_prices = np.array(forecast).reshape(-1)

print("Forecasted opening prices for the next seven days:")
print(forecast_opening_prices)

# Plot forecasted opening prices for the next seven days without connecting lines
forecast_days = np.arange(n_days_original, n_days_original + 7)
plt.plot(forecast_days, forecast_opening_prices, marker='o', linestyle='None', label='Forecasted Data')

plt.title('Seven-Day Forecast')
plt.xlabel('Day')
plt.ylabel('Price')
plt.xticks(np.arange(n_days_original, n_days_original + 7), ['Day 1', 'Day 2', 'Day 3', 'Day 4', 'Day 5', 'Day 6', 'Day 7'])
plt.legend()
plt.show()


# %%
import numpy as np
import matplotlib.pyplot as plt

# Assuming X_scaled, model, and y are defined

# Get the last available data point
last_data_point = X_scaled[-1].reshape(1, 1, X_scaled.shape[1])

# Predict the next 30 days
forecast = []
for _ in range(30):
    next_day_prediction = model.predict(last_data_point)
    forecast.append(next_day_prediction)
    # Update last_data_point to include the predicted value for the next day
    last_data_point = np.concatenate([last_data_point, next_day_prediction.reshape(1, 1, 1)], axis=2)

# Extract opening prices from the forecast
forecast_opening_prices = np.array(forecast).reshape(-1)

print("Forecasted opening prices for the next 30 days:")
print(forecast_opening_prices)

# Plot forecasted opening prices for the next 30 days without connecting lines
forecast_days = np.arange(n_days_original, n_days_original + 30)
plt.plot(forecast_days, forecast_opening_prices, marker='o', linestyle='None', label='Forecasted Data')

plt.title('Thirty-Day Forecast')
plt.xlabel('Day')

# Labeling x-axis ticks for days 5, 10, 15, 20, 25, and 30
x_ticks = [5, 10, 15, 20, 25, 29]  # Corrected the indexing
plt.xticks(np.arange(n_days_original, n_days_original + 30)[x_ticks], [f'Day {i}' for i in x_ticks])

plt.ylabel('Price')
plt.legend()
plt.show()


# %% [markdown]
# **XAI**

# %%
import matplotlib.pyplot as plt
import numpy as np
import keras.backend as K
import tensorflow as tf

# Define a function to generate Grad-CAM
def generate_grad_cam(model, sequence, class_index):
    # Expand dimensions to make it compatible with model input shape
    sequence = np.expand_dims(sequence, axis=0)
    
    # Get the output of the last LSTM layer
    lstm_output = model.layers[0](sequence)
    
    # Get the output of the Dense layer
    dense_output = model.layers[1](lstm_output)
    
    # Compute the gradient of the class output with respect to the Dense layer output
    with tf.GradientTape() as tape:
        tape.watch(dense_output)
        loss = dense_output[:, class_index]
    grads = tape.gradient(loss, dense_output)
    
    # Get the weights of the Dense layer
    dense_weights = model.layers[1].get_weights()[0]
    
    # Compute the importance of each feature map
    importance = tf.reduce_mean(grads * dense_weights, axis=(0, 1))
    
    # Generate the heatmap
    heatmap = np.dot(importance, lstm_output.numpy().squeeze())
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    
    return heatmap

# Choose a sample from the test set
sample_index = 0  # Choose any sample index from the test set
sample_sequence = X_test_reshaped[sample_index]
sample_label = y_test[sample_index]

# Generate the Grad-CAM heatmap
heatmap = generate_grad_cam(model, sample_sequence, class_index=0)  # Assuming binary classification

# Plot the saliency map (heatmap)
plt.plot(heatmap)
plt.title('Saliency Map (Heatmap)')
plt.xlabel('Time Steps')
plt.ylabel('Importance')
plt.show()


# %%
# import shap

# # Assuming 'model', 'X_train_reshaped', 'X_test_reshaped' are defined

# # Initialize the SHAP explainer with appropriate feature_perturbation
# explainer = shap.Explainer(model, X_train_reshaped, feature_perturbation="interventional")

# # Compute SHAP values for the test set
# shap_values = explainer.shap_values(X_test_reshaped)

# # Plot SHAP summary plot
# shap.summary_plot(shap_values, X_test_reshaped)

# # Choose a specific prediction
# sample_index = 0

# # Visualize the SHAP values for this prediction
# shap.force_plot(explainer.expected_value[0], shap_values[0][sample_index], X_test_reshaped[sample_index])


# %%
# type(X_train_reshaped)
# type(X_test_reshaped)
# X_train_reshaped.shape
# X_test_reshaped.shape

# %%
import numpy as np
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

def permutation_importance(model, X, y, metric=mean_squared_error, n_iterations=100):
    baseline_score = metric(y, model.predict(X))
    feature_importances = {}

    for feature_idx in tqdm(range(X.shape[1]), desc='Calculating Permutation Importances'):
        shuffled_X = X.copy()
        np.random.shuffle(shuffled_X[:, feature_idx])
        permuted_score = 0
        for _ in range(n_iterations):
            permuted_score += metric(y, model.predict(shuffled_X))
        feature_importances[feature_idx] = baseline_score - permuted_score / n_iterations

    return feature_importances

# Calculate permutation importance
perm_importances = permutation_importance(model, X_test_reshaped, y_test)

# Print feature importances
for i, importance in sorted(perm_importances.items(), key=lambda x: x[1], reverse=True):
    print(f'Feature {i}: {importance}')


# %%
import numpy as np
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

def permutation_importance(model, X, y, metric=mean_squared_error, n_iterations=100):
    baseline_score = metric(y, model.predict(X))
    feature_importances = {}

    for feature_idx in tqdm(range(X.shape[2]), desc='Calculating Permutation Importances'):
        permuted_scores = []
        for _ in range(n_iterations):
            shuffled_X = X.copy()
            np.random.shuffle(shuffled_X[:, :, feature_idx])
            permuted_score = metric(y, model.predict(shuffled_X))
            permuted_scores.append(permuted_score)
        feature_importances[feature_idx] = baseline_score - np.mean(permuted_scores)

    return feature_importances

# Calculate permutation importance
perm_importances = permutation_importance(model, X_test_reshaped, y_test)

# Print feature importances
for i, importance in sorted(perm_importances.items(), key=lambda x: x[1], reverse=True):
    print(f'Feature {i}: {importance}')


# %%
import numpy as np
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

def permutation_importance(model, X, y, metric=mean_squared_error, n_iterations=100):
    baseline_score = metric(y, model.predict(X))
    feature_importances = {}

    for feature_idx in tqdm(range(X.shape[1]), desc='Calculating Permutation Importances'):
        shuffled_X = X.copy()
        np.random.shuffle(shuffled_X[:, feature_idx])
        permuted_score = 0
        for _ in range(n_iterations):
            permuted_score += metric(y, model.predict(shuffled_X))
        feature_importances[feature_idx] = baseline_score - permuted_score / n_iterations

    return feature_importances

# Calculate permutation importance for all features
perm_importances = permutation_importance(model, X_test_reshaped, y_test)

# Print feature importances
for i, importance in sorted(perm_importances.items(), key=lambda x: x[1], reverse=True):
    print(f'Feature {i}: {importance}')


# %%
import matplotlib.pyplot as plt

# Make predictions on the test set
y_pred = model.predict(X_test_reshaped)

# Calculate differences between consecutive predicted opening prices
predicted_growth = [y_pred[i+1] - y_pred[i] for i in range(len(y_pred)-1)]

# Get dates for plotting
dates = BATBC.index[train_size + val_size + 1:]  # Assuming dates are in the index

# Plot predicted growth over time (date)
plt.figure(figsize=(10, 6))
plt.plot(dates, predicted_growth, label='Predicted Growth', color='green')
plt.title('Predicted Growth Over Time')
plt.xlabel('Date')
plt.ylabel('Predicted Growth')
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.legend()
plt.show()


# %% [markdown]
# **Data visualization for WALTONHIL**

# %%
import matplotlib.pyplot as plt

plt.figure(figsize=(5, 3))
plt.plot(WALTONHIL.index, WALTONHIL['opening_price'], marker='o', markersize=1, linestyle='-')
plt.title('Opening Price Over Time')
plt.xlabel('Date')
plt.ylabel('Opening Price')
plt.grid(True)
plt.tight_layout()
plt.show()


# %%
from sklearn.preprocessing import MinMaxScaler

# Define the features and target variable
features = ['opening_price', 'closing_price', 'yesterdays_closing_price', 'trade', 'value_mn', 'volume']
X = WALTONHIL[features].values
y = WALTONHIL['opening_price'].values

# Scale the features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Define the sizes for training, validation, and testing sets
train_size = int(0.8 * len(X))  # 80% for training
val_size = int(0.1 * len(X))    # 10% for validation
test_size = len(X) - train_size - val_size  # Remaining 10% for testing

# Split the features and target variable accordingly
X_train = X_scaled[:train_size]
X_val = X_scaled[train_size:train_size+val_size]
X_test = X_scaled[train_size+val_size:]

y_train = y[:train_size]
y_val = y[train_size:train_size+val_size]
y_test = y[train_size+val_size:]

# Reshape the data for LSTM input (assuming a time step of 1)
X_train_reshaped = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
X_val_reshaped = X_val.reshape(X_val.shape[0], 1, X_val.shape[1])
X_test_reshaped = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

# Now you can use X_train_reshaped, y_train, X_val_reshaped, y_val, X_test_reshaped, and y_test for training and evaluating your LSTM model

# %%
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.optimizers import Adam
import numpy as np

# Define the LSTM model
model = Sequential([
    LSTM(units=50, input_shape=(X_train_reshaped.shape[1], X_train_reshaped.shape[2])),
    Dense(units=1)
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

# Train the model
history = model.fit(X_train_reshaped, y_train, epochs=50, batch_size=32, validation_data=(X_val_reshaped, y_val))

# Evaluate the model on the test set
loss = model.evaluate(X_test_reshaped, y_test)
print(f'Test Loss: {loss}')

# Calculate and print RMSE
y_pred = model.predict(X_test_reshaped)
rmse = np.sqrt(np.mean((y_pred - y_test)**2))
print(f'RMSE: {rmse}')

# Plot training history
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# %% [markdown]
# > **3 Day Forecast**

# %%
import numpy as np
import matplotlib.pyplot as plt

# Assuming X_scaled, model, and y are defined

# Get the last available data point
last_data_point = X_scaled[-1].reshape(1, 1, X_scaled.shape[1])

# Predict the next three days
forecast = []
for _ in range(3):
    next_day_prediction = model.predict(last_data_point)
    forecast.append(next_day_prediction)
    # Update last_data_point to include the predicted value for the next day
    last_data_point = np.concatenate([last_data_point, next_day_prediction.reshape(1, 1, 1)], axis=2)

# Extract opening prices from the forecast
forecast_opening_prices = np.array(forecast).reshape(-1)

print("Forecasted opening prices for the next three days:")
print(forecast_opening_prices)

# Number of original days
n_days_original = len(y)

# Plot forecasted opening prices for the next three days without connecting lines
forecast_days = np.arange(n_days_original, n_days_original + 3)
plt.plot(forecast_days, forecast_opening_prices, marker='o', linestyle='None', label='Forecasted Data')

plt.title('Three-Day Forecast')
plt.xlabel('Day')
plt.ylabel('Price')
plt.xticks(np.arange(n_days_original, n_days_original + 3), ['Day 1', 'Day 2', 'Day 3'])
plt.legend()
plt.show()


# %% [markdown]
# > **7 Day Forecast**

# %%
import numpy as np
import matplotlib.pyplot as plt

# Assuming X_scaled, model, and y are defined

# Get the last available data point
last_data_point = X_scaled[-1].reshape(1, 1, X_scaled.shape[1])

# Predict the next seven days
forecast = []
for _ in range(7):
    next_day_prediction = model.predict(last_data_point)
    forecast.append(next_day_prediction)
    # Update last_data_point to include the predicted value for the next day
    last_data_point = np.concatenate([last_data_point, next_day_prediction.reshape(1, 1, 1)], axis=2)

# Extract opening prices from the forecast
forecast_opening_prices = np.array(forecast).reshape(-1)

print("Forecasted opening prices for the next seven days:")
print(forecast_opening_prices)

# Plot forecasted opening prices for the next seven days without connecting lines
forecast_days = np.arange(n_days_original, n_days_original + 7)
plt.plot(forecast_days, forecast_opening_prices, marker='o', linestyle='None', label='Forecasted Data')

plt.title('Seven-Day Forecast')
plt.xlabel('Day')
plt.ylabel('Price')
plt.xticks(np.arange(n_days_original, n_days_original + 7), ['Day 1', 'Day 2', 'Day 3', 'Day 4', 'Day 5', 'Day 6', 'Day 7'])
plt.legend()
plt.show()


# %% [markdown]
# > **30 Day Forecast**

# %%
import numpy as np
import matplotlib.pyplot as plt

# Assuming X_scaled, model, and y are defined

# Get the last available data point
last_data_point = X_scaled[-1].reshape(1, 1, X_scaled.shape[1])

# Predict the next 30 days
forecast = []
for _ in range(30):
    next_day_prediction = model.predict(last_data_point)
    forecast.append(next_day_prediction)
    # Update last_data_point to include the predicted value for the next day
    last_data_point = np.concatenate([last_data_point, next_day_prediction.reshape(1, 1, 1)], axis=2)

# Extract opening prices from the forecast
forecast_opening_prices = np.array(forecast).reshape(-1)

print("Forecasted opening prices for the next 30 days:")
print(forecast_opening_prices)

# Plot forecasted opening prices for the next 30 days without connecting lines
forecast_days = np.arange(n_days_original, n_days_original + 30)
plt.plot(forecast_days, forecast_opening_prices, marker='o', linestyle='None', label='Forecasted Data')

plt.title('Thirty-Day Forecast')
plt.xlabel('Day')

# Labeling x-axis ticks for days 5, 10, 15, 20, 25, and 30
x_ticks = [5, 10, 15, 20, 25, 29]  # Corrected the indexing
plt.xticks(np.arange(n_days_original, n_days_original + 30)[x_ticks], [f'Day {i}' for i in x_ticks])

plt.ylabel('Price')
plt.legend()
plt.show()


# %% [markdown]
# **XAI**

# %%
import matplotlib.pyplot as plt
import numpy as np
import keras.backend as K
import tensorflow as tf

# Define a function to generate Grad-CAM
def generate_grad_cam(model, sequence, class_index):
    # Expand dimensions to make it compatible with model input shape
    sequence = np.expand_dims(sequence, axis=0)
    
    # Get the output of the last LSTM layer
    lstm_output = model.layers[0](sequence)
    
    # Get the output of the Dense layer
    dense_output = model.layers[1](lstm_output)
    
    # Compute the gradient of the class output with respect to the Dense layer output
    with tf.GradientTape() as tape:
        tape.watch(dense_output)
        loss = dense_output[:, class_index]
    grads = tape.gradient(loss, dense_output)
    
    # Get the weights of the Dense layer
    dense_weights = model.layers[1].get_weights()[0]
    
    # Compute the importance of each feature map
    importance = tf.reduce_mean(grads * dense_weights, axis=(0, 1))
    
    # Generate the heatmap
    heatmap = np.dot(importance, lstm_output.numpy().squeeze())
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    
    return heatmap

# Choose a sample from the test set
sample_index = 0  # Choose any sample index from the test set
sample_sequence = X_test_reshaped[sample_index]
sample_label = y_test[sample_index]

# Generate the Grad-CAM heatmap
heatmap = generate_grad_cam(model, sample_sequence, class_index=0)  # Assuming binary classification

# Plot the saliency map (heatmap)
plt.plot(heatmap)
plt.title('Saliency Map (Heatmap)')
plt.xlabel('Time Steps')
plt.ylabel('Importance')
plt.show()


# %%
import numpy as np
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

def permutation_importance(model, X, y, metric=mean_squared_error, n_iterations=100):
    baseline_score = metric(y, model.predict(X))
    feature_importances = {}

    for feature_idx in tqdm(range(X.shape[1]), desc='Calculating Permutation Importances'):
        shuffled_X = X.copy()
        np.random.shuffle(shuffled_X[:, feature_idx])
        permuted_score = 0
        for _ in range(n_iterations):
            permuted_score += metric(y, model.predict(shuffled_X))
        feature_importances[feature_idx] = baseline_score - permuted_score / n_iterations

    return feature_importances

# Calculate permutation importance
perm_importances = permutation_importance(model, X_test_reshaped, y_test)

# Print feature importances
for i, importance in sorted(perm_importances.items(), key=lambda x: x[1], reverse=True):
    print(f'Feature {i}: {importance}')


# %%
import numpy as np
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

def permutation_importance(model, X, y, metric=mean_squared_error, n_iterations=100):
    baseline_score = metric(y, model.predict(X))
    feature_importances = {}

    for feature_idx in tqdm(range(X.shape[2]), desc='Calculating Permutation Importances'):
        permuted_scores = []
        for _ in range(n_iterations):
            shuffled_X = X.copy()
            np.random.shuffle(shuffled_X[:, :, feature_idx])
            permuted_score = metric(y, model.predict(shuffled_X))
            permuted_scores.append(permuted_score)
        feature_importances[feature_idx] = baseline_score - np.mean(permuted_scores)

    return feature_importances

# Calculate permutation importance
perm_importances = permutation_importance(model, X_test_reshaped, y_test)

# Print feature importances
for i, importance in sorted(perm_importances.items(), key=lambda x: x[1], reverse=True):
    print(f'Feature {i}: {importance}')


# %%
import matplotlib.pyplot as plt

# Make predictions on the test set
y_pred = model.predict(X_test_reshaped)

# Plot actual vs predicted values
plt.figure(figsize=(10, 6))
plt.plot(y_test, label='Actual Opening Price', color='blue')
plt.plot(y_pred, label='Predicted Opening Price', color='red')
plt.title('Actual vs Predicted Opening Price')
plt.xlabel('Time')
plt.ylabel('Opening Price')
plt.legend()
plt.show()


# %%
import matplotlib.pyplot as plt

# Make predictions on the test set
y_pred = model.predict(X_test_reshaped)

# Calculate differences between consecutive predicted opening prices
predicted_growth = [y_pred[i+1] - y_pred[i] for i in range(len(y_pred)-1)]

# Get dates for plotting
dates = WALTONHIL.index[train_size + val_size + 1:]  # Assuming dates are in the index

# Plot predicted growth over time (date)
plt.figure(figsize=(10, 6))
plt.plot(dates, predicted_growth, label='Predicted Growth', color='green')
plt.title('Predicted Growth Over Time')
plt.xlabel('Date')
plt.ylabel('Predicted Growth')
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.legend()
plt.show()


# %% [markdown]
# **Data visualization for SQURPHARMA**

# %%
import matplotlib.pyplot as plt

plt.figure(figsize=(5, 3)) 
plt.plot(SQURPHARMA.index, SQURPHARMA['opening_price'], marker='o',markersize=1, linestyle='-')
plt.title('Opening Price Over Time')
plt.xlabel('Date')
plt.ylabel('Opening Price')
plt.grid(True)
plt.tight_layout() 
plt.show()


# %%
from sklearn.preprocessing import MinMaxScaler

# Define the features and target variable
features = ['opening_price', 'closing_price', 'yesterdays_closing_price', 'trade', 'value_mn', 'volume']
X = SQURPHARMA[features].values
y = SQURPHARMA['opening_price'].values

# Scale the features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Define the sizes for training, validation, and testing sets
train_size = int(0.8 * len(X))  # 80% for training
val_size = int(0.1 * len(X))    # 10% for validation
test_size = len(X) - train_size - val_size  # Remaining 10% for testing

# Split the features and target variable accordingly
X_train = X_scaled[:train_size]
X_val = X_scaled[train_size:train_size+val_size]
X_test = X_scaled[train_size+val_size:]

y_train = y[:train_size]
y_val = y[train_size:train_size+val_size]
y_test = y[train_size+val_size:]

# Reshape the data for LSTM input (assuming a time step of 1)
X_train_reshaped = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
X_val_reshaped = X_val.reshape(X_val.shape[0], 1, X_val.shape[1])
X_test_reshaped = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

# Now you can use X_train_reshaped, y_train, X_val_reshaped, y_val, X_test_reshaped, and y_test for training and evaluating your LSTM model

# %%
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.optimizers import Adam
import numpy as np

# Define the LSTM model
model = Sequential([
    LSTM(units=50, input_shape=(X_train_reshaped.shape[1], X_train_reshaped.shape[2])),
    Dense(units=1)
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

# Train the model
history = model.fit(X_train_reshaped, y_train, epochs=50, batch_size=32, validation_data=(X_val_reshaped, y_val))

# Evaluate the model on the test set
loss = model.evaluate(X_test_reshaped, y_test)
print(f'Test Loss: {loss}')

# Calculate and print RMSE
y_pred = model.predict(X_test_reshaped)
rmse = np.sqrt(np.mean((y_pred - y_test)**2))
print(f'RMSE: {rmse}')

# Plot training history
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# %% [markdown]
# > **3 Day Forecast**

# %%
import numpy as np
import matplotlib.pyplot as plt

# Assuming X_scaled, model, and y are defined

# Get the last available data point
last_data_point = X_scaled[-1].reshape(1, 1, X_scaled.shape[1])

# Predict the next three days
forecast = []
for _ in range(3):
    next_day_prediction = model.predict(last_data_point)
    forecast.append(next_day_prediction)
    # Update last_data_point to include the predicted value for the next day
    last_data_point = np.concatenate([last_data_point, next_day_prediction.reshape(1, 1, 1)], axis=2)

# Extract opening prices from the forecast
forecast_opening_prices = np.array(forecast).reshape(-1)

print("Forecasted opening prices for the next three days:")
print(forecast_opening_prices)

# Number of original days
n_days_original = len(y)

# Plot forecasted opening prices for the next three days without connecting lines
forecast_days = np.arange(n_days_original, n_days_original + 3)
plt.plot(forecast_days, forecast_opening_prices, marker='o', linestyle='None', label='Forecasted Data')

plt.title('Three-Day Forecast')
plt.xlabel('Day')
plt.ylabel('Price')
plt.xticks(np.arange(n_days_original, n_days_original + 3), ['Day 1', 'Day 2', 'Day 3'])
plt.legend()
plt.show()


# %% [markdown]
# > **7 Day Forecast**

# %%
import numpy as np
import matplotlib.pyplot as plt

# Assuming X_scaled, model, and y are defined

# Get the last available data point
last_data_point = X_scaled[-1].reshape(1, 1, X_scaled.shape[1])

# Predict the next seven days
forecast = []
for _ in range(7):
    next_day_prediction = model.predict(last_data_point)
    forecast.append(next_day_prediction)
    # Update last_data_point to include the predicted value for the next day
    last_data_point = np.concatenate([last_data_point, next_day_prediction.reshape(1, 1, 1)], axis=2)

# Extract opening prices from the forecast
forecast_opening_prices = np.array(forecast).reshape(-1)

print("Forecasted opening prices for the next seven days:")
print(forecast_opening_prices)

# Plot forecasted opening prices for the next seven days without connecting lines
forecast_days = np.arange(n_days_original, n_days_original + 7)
plt.plot(forecast_days, forecast_opening_prices, marker='o', linestyle='None', label='Forecasted Data')

plt.title('Seven-Day Forecast')
plt.xlabel('Day')
plt.ylabel('Price')
plt.xticks(np.arange(n_days_original, n_days_original + 7), ['Day 1', 'Day 2', 'Day 3', 'Day 4', 'Day 5', 'Day 6', 'Day 7'])
plt.legend()
plt.show()


# %% [markdown]
# > **30 Day Forecast**

# %%
import numpy as np
import matplotlib.pyplot as plt

# Assuming X_scaled, model, and y are defined

# Get the last available data point
last_data_point = X_scaled[-1].reshape(1, 1, X_scaled.shape[1])

# Predict the next 30 days
forecast = []
for _ in range(30):
    next_day_prediction = model.predict(last_data_point)
    forecast.append(next_day_prediction)
    # Update last_data_point to include the predicted value for the next day
    last_data_point = np.concatenate([last_data_point, next_day_prediction.reshape(1, 1, 1)], axis=2)

# Extract opening prices from the forecast
forecast_opening_prices = np.array(forecast).reshape(-1)

print("Forecasted opening prices for the next 30 days:")
print(forecast_opening_prices)

# Plot forecasted opening prices for the next 30 days without connecting lines
forecast_days = np.arange(n_days_original, n_days_original + 30)
plt.plot(forecast_days, forecast_opening_prices, marker='o', linestyle='None', label='Forecasted Data')

plt.title('Thirty-Day Forecast')
plt.xlabel('Day')

# Labeling x-axis ticks for days 5, 10, 15, 20, 25, and 30
x_ticks = [5, 10, 15, 20, 25, 29]  # Corrected the indexing
plt.xticks(np.arange(n_days_original, n_days_original + 30)[x_ticks], [f'Day {i}' for i in x_ticks])

plt.ylabel('Price')
plt.legend()
plt.show()


# %% [markdown]
# **XAI**

# %%
import matplotlib.pyplot as plt
import numpy as np
import keras.backend as K
import tensorflow as tf

# Define a function to generate Grad-CAM
def generate_grad_cam(model, sequence, class_index):
    # Expand dimensions to make it compatible with model input shape
    sequence = np.expand_dims(sequence, axis=0)
    
    # Get the output of the last LSTM layer
    lstm_output = model.layers[0](sequence)
    
    # Get the output of the Dense layer
    dense_output = model.layers[1](lstm_output)
    
    # Compute the gradient of the class output with respect to the Dense layer output
    with tf.GradientTape() as tape:
        tape.watch(dense_output)
        loss = dense_output[:, class_index]
    grads = tape.gradient(loss, dense_output)
    
    # Get the weights of the Dense layer
    dense_weights = model.layers[1].get_weights()[0]
    
    # Compute the importance of each feature map
    importance = tf.reduce_mean(grads * dense_weights, axis=(0, 1))
    
    # Generate the heatmap
    heatmap = np.dot(importance, lstm_output.numpy().squeeze())
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    
    return heatmap

# Choose a sample from the test set
sample_index = 0  # Choose any sample index from the test set
sample_sequence = X_test_reshaped[sample_index]
sample_label = y_test[sample_index]

# Generate the Grad-CAM heatmap
heatmap = generate_grad_cam(model, sample_sequence, class_index=0)  # Assuming binary classification

# Plot the saliency map (heatmap)
plt.plot(heatmap)
plt.title('Saliency Map (Heatmap)')
plt.xlabel('Time Steps')
plt.ylabel('Importance')
plt.show()


# %%
import numpy as np
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

def permutation_importance(model, X, y, metric=mean_squared_error, n_iterations=100):
    baseline_score = metric(y, model.predict(X))
    feature_importances = {}

    for feature_idx in tqdm(range(X.shape[1]), desc='Calculating Permutation Importances'):
        shuffled_X = X.copy()
        np.random.shuffle(shuffled_X[:, feature_idx])
        permuted_score = 0
        for _ in range(n_iterations):
            permuted_score += metric(y, model.predict(shuffled_X))
        feature_importances[feature_idx] = baseline_score - permuted_score / n_iterations

    return feature_importances

# Calculate permutation importance
perm_importances = permutation_importance(model, X_test_reshaped, y_test)

# Print feature importances
for i, importance in sorted(perm_importances.items(), key=lambda x: x[1], reverse=True):
    print(f'Feature {i}: {importance}')


# %%
import numpy as np
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

def permutation_importance(model, X, y, metric=mean_squared_error, n_iterations=100):
    baseline_score = metric(y, model.predict(X))
    feature_importances = {}

    for feature_idx in tqdm(range(X.shape[2]), desc='Calculating Permutation Importances'):
        permuted_scores = []
        for _ in range(n_iterations):
            shuffled_X = X.copy()
            np.random.shuffle(shuffled_X[:, :, feature_idx])
            permuted_score = metric(y, model.predict(shuffled_X))
            permuted_scores.append(permuted_score)
        feature_importances[feature_idx] = baseline_score - np.mean(permuted_scores)

    return feature_importances

# Calculate permutation importance
perm_importances = permutation_importance(model, X_test_reshaped, y_test)

# Print feature importances
for i, importance in sorted(perm_importances.items(), key=lambda x: x[1], reverse=True):
    print(f'Feature {i}: {importance}')


# %%
import matplotlib.pyplot as plt

# Make predictions on the test set
y_pred = model.predict(X_test_reshaped)

# Plot actual vs predicted values
plt.figure(figsize=(10, 6))
plt.plot(y_test, label='Actual Opening Price', color='blue')
plt.plot(y_pred, label='Predicted Opening Price', color='red')
plt.title('Actual vs Predicted Opening Price')
plt.xlabel('Time')
plt.ylabel('Opening Price')
plt.legend()
plt.show()


# %%
import matplotlib.pyplot as plt

# Make predictions on the test set
y_pred = model.predict(X_test_reshaped)

# Calculate differences between consecutive predicted opening prices
predicted_growth = [y_pred[i+1] - y_pred[i] for i in range(len(y_pred)-1)]

# Get dates for plotting
dates = SQURPHARMA.index[train_size + val_size + 1:]  # Assuming dates are in the index

# Plot predicted growth over time (date)
plt.figure(figsize=(10, 6))
plt.plot(dates, predicted_growth, label='Predicted Growth', color='green')
plt.title('Predicted Growth Over Time')
plt.xlabel('Date')
plt.ylabel('Predicted Growth')
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.legend()
plt.show()


# %% [markdown]
# **Data visualization for ROBI**

# %%
import matplotlib.pyplot as plt

plt.figure(figsize=(5, 3)) 
plt.plot(ROBI.index, ROBI['opening_price'], marker='o',markersize=1, linestyle='-')
plt.title('Opening Price Over Time')
plt.xlabel('Date')
plt.ylabel('Opening Price')
plt.grid(True)
plt.tight_layout() 
plt.show()


# %%
from sklearn.preprocessing import MinMaxScaler

# Define the features and target variable
features = ['opening_price', 'closing_price', 'yesterdays_closing_price', 'trade', 'value_mn', 'volume']
X = ROBI[features].values
y = ROBI['opening_price'].values

# Scale the features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Define the sizes for training, validation, and testing sets
train_size = int(0.8 * len(X))  # 80% for training
val_size = int(0.1 * len(X))    # 10% for validation
test_size = len(X) - train_size - val_size  # Remaining 10% for testing

# Split the features and target variable accordingly
X_train = X_scaled[:train_size]
X_val = X_scaled[train_size:train_size+val_size]
X_test = X_scaled[train_size+val_size:]

y_train = y[:train_size]
y_val = y[train_size:train_size+val_size]
y_test = y[train_size+val_size:]

# Reshape the data for LSTM input (assuming a time step of 1)
X_train_reshaped = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
X_val_reshaped = X_val.reshape(X_val.shape[0], 1, X_val.shape[1])
X_test_reshaped = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

# Now you can use X_train_reshaped, y_train, X_val_reshaped, y_val, X_test_reshaped, and y_test for training and evaluating your LSTM model

# %%
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.optimizers import Adam
import numpy as np

# Define the LSTM model
model = Sequential([
    LSTM(units=50, input_shape=(X_train_reshaped.shape[1], X_train_reshaped.shape[2])),
    Dense(units=1)
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

# Train the model
history = model.fit(X_train_reshaped, y_train, epochs=50, batch_size=32, validation_data=(X_val_reshaped, y_val))

# Evaluate the model on the test set
loss = model.evaluate(X_test_reshaped, y_test)
print(f'Test Loss: {loss}')

# Calculate and print RMSE
y_pred = model.predict(X_test_reshaped)
rmse = np.sqrt(np.mean((y_pred - y_test)**2))
print(f'RMSE: {rmse}')

# Plot training history
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# %% [markdown]
# > **3 Day Forecast**

# %%
import numpy as np
import matplotlib.pyplot as plt

# Assuming X_scaled, model, and y are defined

# Get the last available data point
last_data_point = X_scaled[-1].reshape(1, 1, X_scaled.shape[1])

# Predict the next three days
forecast = []
for _ in range(3):
    next_day_prediction = model.predict(last_data_point)
    forecast.append(next_day_prediction)
    # Update last_data_point to include the predicted value for the next day
    last_data_point = np.concatenate([last_data_point, next_day_prediction.reshape(1, 1, 1)], axis=2)

# Extract opening prices from the forecast
forecast_opening_prices = np.array(forecast).reshape(-1)

print("Forecasted opening prices for the next three days:")
print(forecast_opening_prices)

# Number of original days
n_days_original = len(y)

# Plot forecasted opening prices for the next three days without connecting lines
forecast_days = np.arange(n_days_original, n_days_original + 3)
plt.plot(forecast_days, forecast_opening_prices, marker='o', linestyle='None', label='Forecasted Data')

plt.title('Three-Day Forecast')
plt.xlabel('Day')
plt.ylabel('Price')
plt.xticks(np.arange(n_days_original, n_days_original + 3), ['Day 1', 'Day 2', 'Day 3'])
plt.legend()
plt.show()


# %% [markdown]
# > **7 Day Forecast**

# %%
import numpy as np
import matplotlib.pyplot as plt

# Assuming X_scaled, model, and y are defined

# Get the last available data point
last_data_point = X_scaled[-1].reshape(1, 1, X_scaled.shape[1])

# Predict the next seven days
forecast = []
for _ in range(7):
    next_day_prediction = model.predict(last_data_point)
    forecast.append(next_day_prediction)
    # Update last_data_point to include the predicted value for the next day
    last_data_point = np.concatenate([last_data_point, next_day_prediction.reshape(1, 1, 1)], axis=2)

# Extract opening prices from the forecast
forecast_opening_prices = np.array(forecast).reshape(-1)

print("Forecasted opening prices for the next seven days:")
print(forecast_opening_prices)

# Plot forecasted opening prices for the next seven days without connecting lines
forecast_days = np.arange(n_days_original, n_days_original + 7)
plt.plot(forecast_days, forecast_opening_prices, marker='o', linestyle='None', label='Forecasted Data')

plt.title('Seven-Day Forecast')
plt.xlabel('Day')
plt.ylabel('Price')
plt.xticks(np.arange(n_days_original, n_days_original + 7), ['Day 1', 'Day 2', 'Day 3', 'Day 4', 'Day 5', 'Day 6', 'Day 7'])
plt.legend()
plt.show()


# %% [markdown]
# > **30 Day Forecast**

# %%
import numpy as np
import matplotlib.pyplot as plt

# Assuming X_scaled, model, and y are defined

# Get the last available data point
last_data_point = X_scaled[-1].reshape(1, 1, X_scaled.shape[1])

# Predict the next 30 days
forecast = []
for _ in range(30):
    next_day_prediction = model.predict(last_data_point)
    forecast.append(next_day_prediction)
    # Update last_data_point to include the predicted value for the next day
    last_data_point = np.concatenate([last_data_point, next_day_prediction.reshape(1, 1, 1)], axis=2)

# Extract opening prices from the forecast
forecast_opening_prices = np.array(forecast).reshape(-1)

print("Forecasted opening prices for the next 30 days:")
print(forecast_opening_prices)

# Plot forecasted opening prices for the next 30 days without connecting lines
forecast_days = np.arange(n_days_original, n_days_original + 30)
plt.plot(forecast_days, forecast_opening_prices, marker='o', linestyle='None', label='Forecasted Data')

plt.title('Thirty-Day Forecast')
plt.xlabel('Day')

# Labeling x-axis ticks for days 5, 10, 15, 20, 25, and 30
x_ticks = [5, 10, 15, 20, 25, 29]  # Corrected the indexing
plt.xticks(np.arange(n_days_original, n_days_original + 30)[x_ticks], [f'Day {i}' for i in x_ticks])

plt.ylabel('Price')
plt.legend()
plt.show()


# %% [markdown]
# **XAI**

# %%
import matplotlib.pyplot as plt
import numpy as np
import keras.backend as K
import tensorflow as tf

# Define a function to generate Grad-CAM
def generate_grad_cam(model, sequence, class_index):
    # Expand dimensions to make it compatible with model input shape
    sequence = np.expand_dims(sequence, axis=0)
    
    # Get the output of the last LSTM layer
    lstm_output = model.layers[0](sequence)
    
    # Get the output of the Dense layer
    dense_output = model.layers[1](lstm_output)
    
    # Compute the gradient of the class output with respect to the Dense layer output
    with tf.GradientTape() as tape:
        tape.watch(dense_output)
        loss = dense_output[:, class_index]
    grads = tape.gradient(loss, dense_output)
    
    # Get the weights of the Dense layer
    dense_weights = model.layers[1].get_weights()[0]
    
    # Compute the importance of each feature map
    importance = tf.reduce_mean(grads * dense_weights, axis=(0, 1))
    
    # Generate the heatmap
    heatmap = np.dot(importance, lstm_output.numpy().squeeze())
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    
    return heatmap

# Choose a sample from the test set
sample_index = 0  # Choose any sample index from the test set
sample_sequence = X_test_reshaped[sample_index]
sample_label = y_test[sample_index]

# Generate the Grad-CAM heatmap
heatmap = generate_grad_cam(model, sample_sequence, class_index=0)  # Assuming binary classification

# Plot the saliency map (heatmap)
plt.plot(heatmap)
plt.title('Saliency Map (Heatmap)')
plt.xlabel('Time Steps')
plt.ylabel('Importance')
plt.show()


# %%
import numpy as np
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

def permutation_importance(model, X, y, metric=mean_squared_error, n_iterations=100):
    baseline_score = metric(y, model.predict(X))
    feature_importances = {}

    for feature_idx in tqdm(range(X.shape[1]), desc='Calculating Permutation Importances'):
        shuffled_X = X.copy()
        np.random.shuffle(shuffled_X[:, feature_idx])
        permuted_score = 0
        for _ in range(n_iterations):
            permuted_score += metric(y, model.predict(shuffled_X))
        feature_importances[feature_idx] = baseline_score - permuted_score / n_iterations

    return feature_importances

# Calculate permutation importance
perm_importances = permutation_importance(model, X_test_reshaped, y_test)

# Print feature importances
for i, importance in sorted(perm_importances.items(), key=lambda x: x[1], reverse=True):
    print(f'Feature {i}: {importance}')


# %%
import numpy as np
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

def permutation_importance(model, X, y, metric=mean_squared_error, n_iterations=100):
    baseline_score = metric(y, model.predict(X))
    feature_importances = {}

    for feature_idx in tqdm(range(X.shape[2]), desc='Calculating Permutation Importances'):
        permuted_scores = []
        for _ in range(n_iterations):
            shuffled_X = X.copy()
            np.random.shuffle(shuffled_X[:, :, feature_idx])
            permuted_score = metric(y, model.predict(shuffled_X))
            permuted_scores.append(permuted_score)
        feature_importances[feature_idx] = baseline_score - np.mean(permuted_scores)

    return feature_importances

# Calculate permutation importance
perm_importances = permutation_importance(model, X_test_reshaped, y_test)

# Print feature importances
for i, importance in sorted(perm_importances.items(), key=lambda x: x[1], reverse=True):
    print(f'Feature {i}: {importance}')


# %%
import matplotlib.pyplot as plt

# Make predictions on the test set
y_pred = model.predict(X_test_reshaped)

# Plot actual vs predicted values
plt.figure(figsize=(10, 6))
plt.plot(y_test, label='Actual Opening Price', color='blue')
plt.plot(y_pred, label='Predicted Opening Price', color='red')
plt.title('Actual vs Predicted Opening Price')
plt.xlabel('Time')
plt.ylabel('Opening Price')
plt.legend()
plt.show()


# %%
import matplotlib.pyplot as plt

# Make predictions on the test set
y_pred = model.predict(X_test_reshaped)

# Calculate differences between consecutive predicted opening prices
predicted_growth = [y_pred[i+1] - y_pred[i] for i in range(len(y_pred)-1)]

# Get dates for plotting
dates = ROBI.index[train_size + val_size + 1:]  # Assuming dates are in the index

# Plot predicted growth over time (date)
plt.figure(figsize=(10, 6))
plt.plot(dates, predicted_growth, label='Predicted Growth', color='green')
plt.title('Predicted Growth Over Time')
plt.xlabel('Date')
plt.ylabel('Predicted Growth')
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.legend()
plt.show()


# %% [markdown]
# **Data visualization for RENATA**

# %%
import matplotlib.pyplot as plt

plt.figure(figsize=(5, 3)) 
plt.plot(RENATA.index, RENATA['opening_price'], marker='o',markersize=1, linestyle='-')
plt.title('Opening Price Over Time')
plt.xlabel('Date')
plt.ylabel('Opening Price')
plt.grid(True)
plt.tight_layout() 
plt.show()

# %%
from sklearn.preprocessing import MinMaxScaler

# Define the features and target variable
features = ['opening_price', 'closing_price', 'yesterdays_closing_price', 'trade', 'value_mn', 'volume']
X = RENATA[features].values
y = RENATA['opening_price'].values

# Scale the features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Define the sizes for training, validation, and testing sets
train_size = int(0.8 * len(X))  # 80% for training
val_size = int(0.1 * len(X))    # 10% for validation
test_size = len(X) - train_size - val_size  # Remaining 10% for testing

# Split the features and target variable accordingly
X_train = X_scaled[:train_size]
X_val = X_scaled[train_size:train_size+val_size]
X_test = X_scaled[train_size+val_size:]

y_train = y[:train_size]
y_val = y[train_size:train_size+val_size]
y_test = y[train_size+val_size:]

# Reshape the data for LSTM input (assuming a time step of 1)
X_train_reshaped = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
X_val_reshaped = X_val.reshape(X_val.shape[0], 1, X_val.shape[1])
X_test_reshaped = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

# Now you can use X_train_reshaped, y_train, X_val_reshaped, y_val, X_test_reshaped, and y_test for training and evaluating your LSTM model

# %%
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.optimizers import Adam
import numpy as np

# Define the LSTM model
model = Sequential([
    LSTM(units=50, input_shape=(X_train_reshaped.shape[1], X_train_reshaped.shape[2])),
    Dense(units=1)
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

# Train the model
history = model.fit(X_train_reshaped, y_train, epochs=50, batch_size=32, validation_data=(X_val_reshaped, y_val))

# Evaluate the model on the test set
loss = model.evaluate(X_test_reshaped, y_test)
print(f'Test Loss: {loss}')

# Calculate and print RMSE
y_pred = model.predict(X_test_reshaped)
rmse = np.sqrt(np.mean((y_pred - y_test)**2))
print(f'RMSE: {rmse}')

# Plot training history
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# %% [markdown]
# > **3 Day Forecast**

# %%
import numpy as np
import matplotlib.pyplot as plt

# Assuming X_scaled, model, and y are defined

# Get the last available data point
last_data_point = X_scaled[-1].reshape(1, 1, X_scaled.shape[1])

# Predict the next three days
forecast = []
for _ in range(3):
    next_day_prediction = model.predict(last_data_point)
    forecast.append(next_day_prediction)
    # Update last_data_point to include the predicted value for the next day
    last_data_point = np.concatenate([last_data_point, next_day_prediction.reshape(1, 1, 1)], axis=2)

# Extract opening prices from the forecast
forecast_opening_prices = np.array(forecast).reshape(-1)

print("Forecasted opening prices for the next three days:")
print(forecast_opening_prices)

# Number of original days
n_days_original = len(y)

# Plot forecasted opening prices for the next three days without connecting lines
forecast_days = np.arange(n_days_original, n_days_original + 3)
plt.plot(forecast_days, forecast_opening_prices, marker='o', linestyle='None', label='Forecasted Data')

plt.title('Three-Day Forecast')
plt.xlabel('Day')
plt.ylabel('Price')
plt.xticks(np.arange(n_days_original, n_days_original + 3), ['Day 1', 'Day 2', 'Day 3'])
plt.legend()
plt.show()


# %% [markdown]
# > **7 Day Forecast**

# %%
import numpy as np
import matplotlib.pyplot as plt

# Assuming X_scaled, model, and y are defined

# Get the last available data point
last_data_point = X_scaled[-1].reshape(1, 1, X_scaled.shape[1])

# Predict the next seven days
forecast = []
for _ in range(7):
    next_day_prediction = model.predict(last_data_point)
    forecast.append(next_day_prediction)
    # Update last_data_point to include the predicted value for the next day
    last_data_point = np.concatenate([last_data_point, next_day_prediction.reshape(1, 1, 1)], axis=2)

# Extract opening prices from the forecast
forecast_opening_prices = np.array(forecast).reshape(-1)

print("Forecasted opening prices for the next seven days:")
print(forecast_opening_prices)

# Plot forecasted opening prices for the next seven days without connecting lines
forecast_days = np.arange(n_days_original, n_days_original + 7)
plt.plot(forecast_days, forecast_opening_prices, marker='o', linestyle='None', label='Forecasted Data')

plt.title('Seven-Day Forecast')
plt.xlabel('Day')
plt.ylabel('Price')
plt.xticks(np.arange(n_days_original, n_days_original + 7), ['Day 1', 'Day 2', 'Day 3', 'Day 4', 'Day 5', 'Day 6', 'Day 7'])
plt.legend()
plt.show()


# %% [markdown]
# > **30 Day Forecast**

# %%
import numpy as np
import matplotlib.pyplot as plt

# Assuming X_scaled, model, and y are defined

# Get the last available data point
last_data_point = X_scaled[-1].reshape(1, 1, X_scaled.shape[1])

# Predict the next 30 days
forecast = []
for _ in range(30):
    next_day_prediction = model.predict(last_data_point)
    forecast.append(next_day_prediction)
    # Update last_data_point to include the predicted value for the next day
    last_data_point = np.concatenate([last_data_point, next_day_prediction.reshape(1, 1, 1)], axis=2)

# Extract opening prices from the forecast
forecast_opening_prices = np.array(forecast).reshape(-1)

print("Forecasted opening prices for the next 30 days:")
print(forecast_opening_prices)

# Plot forecasted opening prices for the next 30 days without connecting lines
forecast_days = np.arange(n_days_original, n_days_original + 30)
plt.plot(forecast_days, forecast_opening_prices, marker='o', linestyle='None', label='Forecasted Data')

plt.title('Thirty-Day Forecast')
plt.xlabel('Day')

# Labeling x-axis ticks for days 5, 10, 15, 20, 25, and 30
x_ticks = [5, 10, 15, 20, 25, 29]  # Corrected the indexing
plt.xticks(np.arange(n_days_original, n_days_original + 30)[x_ticks], [f'Day {i}' for i in x_ticks])

plt.ylabel('Price')
plt.legend()
plt.show()


# %% [markdown]
# **XAI**

# %%
import matplotlib.pyplot as plt
import numpy as np
import keras.backend as K
import tensorflow as tf

# Define a function to generate Grad-CAM
def generate_grad_cam(model, sequence, class_index):
    # Expand dimensions to make it compatible with model input shape
    sequence = np.expand_dims(sequence, axis=0)
    
    # Get the output of the last LSTM layer
    lstm_output = model.layers[0](sequence)
    
    # Get the output of the Dense layer
    dense_output = model.layers[1](lstm_output)
    
    # Compute the gradient of the class output with respect to the Dense layer output
    with tf.GradientTape() as tape:
        tape.watch(dense_output)
        loss = dense_output[:, class_index]
    grads = tape.gradient(loss, dense_output)
    
    # Get the weights of the Dense layer
    dense_weights = model.layers[1].get_weights()[0]
    
    # Compute the importance of each feature map
    importance = tf.reduce_mean(grads * dense_weights, axis=(0, 1))
    
    # Generate the heatmap
    heatmap = np.dot(importance, lstm_output.numpy().squeeze())
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    
    return heatmap

# Choose a sample from the test set
sample_index = 0  # Choose any sample index from the test set
sample_sequence = X_test_reshaped[sample_index]
sample_label = y_test[sample_index]

# Generate the Grad-CAM heatmap
heatmap = generate_grad_cam(model, sample_sequence, class_index=0)  # Assuming binary classification

# Plot the saliency map (heatmap)
plt.plot(heatmap)
plt.title('Saliency Map (Heatmap)')
plt.xlabel('Time Steps')
plt.ylabel('Importance')
plt.show()


# %%
import numpy as np
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

def permutation_importance(model, X, y, metric=mean_squared_error, n_iterations=100):
    baseline_score = metric(y, model.predict(X))
    feature_importances = {}

    for feature_idx in tqdm(range(X.shape[1]), desc='Calculating Permutation Importances'):
        shuffled_X = X.copy()
        np.random.shuffle(shuffled_X[:, feature_idx])
        permuted_score = 0
        for _ in range(n_iterations):
            permuted_score += metric(y, model.predict(shuffled_X))
        feature_importances[feature_idx] = baseline_score - permuted_score / n_iterations

    return feature_importances

# Calculate permutation importance
perm_importances = permutation_importance(model, X_test_reshaped, y_test)

# Print feature importances
for i, importance in sorted(perm_importances.items(), key=lambda x: x[1], reverse=True):
    print(f'Feature {i}: {importance}')


# %%
import numpy as np
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

def permutation_importance(model, X, y, metric=mean_squared_error, n_iterations=100):
    baseline_score = metric(y, model.predict(X))
    feature_importances = {}

    for feature_idx in tqdm(range(X.shape[2]), desc='Calculating Permutation Importances'):
        permuted_scores = []
        for _ in range(n_iterations):
            shuffled_X = X.copy()
            np.random.shuffle(shuffled_X[:, :, feature_idx])
            permuted_score = metric(y, model.predict(shuffled_X))
            permuted_scores.append(permuted_score)
        feature_importances[feature_idx] = baseline_score - np.mean(permuted_scores)

    return feature_importances

# Calculate permutation importance
perm_importances = permutation_importance(model, X_test_reshaped, y_test)

# Print feature importances
for i, importance in sorted(perm_importances.items(), key=lambda x: x[1], reverse=True):
    print(f'Feature {i}: {importance}')


# %%
import matplotlib.pyplot as plt

# Make predictions on the test set
y_pred = model.predict(X_test_reshaped)

# Plot actual vs predicted values
plt.figure(figsize=(10, 6))
plt.plot(y_test, label='Actual Opening Price', color='blue')
plt.plot(y_pred, label='Predicted Opening Price', color='red')
plt.title('Actual vs Predicted Opening Price')
plt.xlabel('Time')
plt.ylabel('Opening Price')
plt.legend()
plt.show()


# %%
import matplotlib.pyplot as plt

# Make predictions on the test set
y_pred = model.predict(X_test_reshaped)

# Calculate differences between consecutive predicted opening prices
predicted_growth = [y_pred[i+1] - y_pred[i] for i in range(len(y_pred)-1)]

# Get dates for plotting
dates = RENATA.index[train_size + val_size + 1:]  # Assuming dates are in the index

# Plot predicted growth over time (date)
plt.figure(figsize=(10, 6))
plt.plot(dates, predicted_growth, label='Predicted Growth', color='green')
plt.title('Predicted Growth Over Time')
plt.xlabel('Date')
plt.ylabel('Predicted Growth')
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.legend()
plt.show()


# %% [markdown]
# **Data visualization for BEXIMCO**

# %%
import matplotlib.pyplot as plt

plt.figure(figsize=(5, 3)) 
plt.plot(BEXIMCO.index, BEXIMCO['opening_price'], marker='o',markersize=1, linestyle='-')
plt.title('Opening Price Over Time')
plt.xlabel('Date')
plt.ylabel('Opening Price')
plt.grid(True)
plt.tight_layout() 
plt.show()


# %%
from sklearn.preprocessing import MinMaxScaler

# Define the features and target variable
features = ['opening_price', 'closing_price', 'yesterdays_closing_price', 'trade', 'value_mn', 'volume']
X = BEXIMCO[features].values
y = BEXIMCO['opening_price'].values

# Scale the features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Define the sizes for training, validation, and testing sets
train_size = int(0.8 * len(X))  # 80% for training
val_size = int(0.1 * len(X))    # 10% for validation
test_size = len(X) - train_size - val_size  # Remaining 10% for testing

# Split the features and target variable accordingly
X_train = X_scaled[:train_size]
X_val = X_scaled[train_size:train_size+val_size]
X_test = X_scaled[train_size+val_size:]

y_train = y[:train_size]
y_val = y[train_size:train_size+val_size]
y_test = y[train_size+val_size:]

# Reshape the data for LSTM input (assuming a time step of 1)
X_train_reshaped = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
X_val_reshaped = X_val.reshape(X_val.shape[0], 1, X_val.shape[1])
X_test_reshaped = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

# Now you can use X_train_reshaped, y_train, X_val_reshaped, y_val, X_test_reshaped, and y_test for training and evaluating your LSTM model

# %%
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.optimizers import Adam
import numpy as np

# Define the LSTM model
model = Sequential([
    LSTM(units=50, input_shape=(X_train_reshaped.shape[1], X_train_reshaped.shape[2])),
    Dense(units=1)
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

# Train the model
history = model.fit(X_train_reshaped, y_train, epochs=50, batch_size=32, validation_data=(X_val_reshaped, y_val))

# Evaluate the model on the test set
loss = model.evaluate(X_test_reshaped, y_test)
print(f'Test Loss: {loss}')

# Calculate and print RMSE
y_pred = model.predict(X_test_reshaped)
rmse = np.sqrt(np.mean((y_pred - y_test)**2))
print(f'RMSE: {rmse}')

# Plot training history
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# %% [markdown]
# > **3 Day Forecast**

# %%
import numpy as np
import matplotlib.pyplot as plt

# Assuming X_scaled, model, and y are defined

# Get the last available data point
last_data_point = X_scaled[-1].reshape(1, 1, X_scaled.shape[1])

# Predict the next three days
forecast = []
for _ in range(3):
    next_day_prediction = model.predict(last_data_point)
    forecast.append(next_day_prediction)
    # Update last_data_point to include the predicted value for the next day
    last_data_point = np.concatenate([last_data_point, next_day_prediction.reshape(1, 1, 1)], axis=2)

# Extract opening prices from the forecast
forecast_opening_prices = np.array(forecast).reshape(-1)

print("Forecasted opening prices for the next three days:")
print(forecast_opening_prices)

# Number of original days
n_days_original = len(y)

# Plot forecasted opening prices for the next three days without connecting lines
forecast_days = np.arange(n_days_original, n_days_original + 3)
plt.plot(forecast_days, forecast_opening_prices, marker='o', linestyle='None', label='Forecasted Data')

plt.title('Three-Day Forecast')
plt.xlabel('Day')
plt.ylabel('Price')
plt.xticks(np.arange(n_days_original, n_days_original + 3), ['Day 1', 'Day 2', 'Day 3'])
plt.legend()
plt.show()


# %% [markdown]
# > **7 Day Forecast**

# %%
import numpy as np
import matplotlib.pyplot as plt

# Assuming X_scaled, model, and y are defined

# Get the last available data point
last_data_point = X_scaled[-1].reshape(1, 1, X_scaled.shape[1])

# Predict the next seven days
forecast = []
for _ in range(7):
    next_day_prediction = model.predict(last_data_point)
    forecast.append(next_day_prediction)
    # Update last_data_point to include the predicted value for the next day
    last_data_point = np.concatenate([last_data_point, next_day_prediction.reshape(1, 1, 1)], axis=2)

# Extract opening prices from the forecast
forecast_opening_prices = np.array(forecast).reshape(-1)

print("Forecasted opening prices for the next seven days:")
print(forecast_opening_prices)

# Plot forecasted opening prices for the next seven days without connecting lines
forecast_days = np.arange(n_days_original, n_days_original + 7)
plt.plot(forecast_days, forecast_opening_prices, marker='o', linestyle='None', label='Forecasted Data')

plt.title('Seven-Day Forecast')
plt.xlabel('Day')
plt.ylabel('Price')
plt.xticks(np.arange(n_days_original, n_days_original + 7), ['Day 1', 'Day 2', 'Day 3', 'Day 4', 'Day 5', 'Day 6', 'Day 7'])
plt.legend()
plt.show()


# %% [markdown]
# > **30 Day Forecast**

# %%
import numpy as np
import matplotlib.pyplot as plt

# Assuming X_scaled, model, and y are defined

# Get the last available data point
last_data_point = X_scaled[-1].reshape(1, 1, X_scaled.shape[1])

# Predict the next 30 days
forecast = []
for _ in range(30):
    next_day_prediction = model.predict(last_data_point)
    forecast.append(next_day_prediction)
    # Update last_data_point to include the predicted value for the next day
    last_data_point = np.concatenate([last_data_point, next_day_prediction.reshape(1, 1, 1)], axis=2)

# Extract opening prices from the forecast
forecast_opening_prices = np.array(forecast).reshape(-1)

print("Forecasted opening prices for the next 30 days:")
print(forecast_opening_prices)

# Plot forecasted opening prices for the next 30 days without connecting lines
forecast_days = np.arange(n_days_original, n_days_original + 30)
plt.plot(forecast_days, forecast_opening_prices, marker='o', linestyle='None', label='Forecasted Data')

plt.title('Thirty-Day Forecast')
plt.xlabel('Day')

# Labeling x-axis ticks for days 5, 10, 15, 20, 25, and 30
x_ticks = [5, 10, 15, 20, 25, 29]  # Corrected the indexing
plt.xticks(np.arange(n_days_original, n_days_original + 30)[x_ticks], [f'Day {i}' for i in x_ticks])

plt.ylabel('Price')
plt.legend()
plt.show()


# %% [markdown]
# **XAI**

# %%
import matplotlib.pyplot as plt
import numpy as np
import keras.backend as K
import tensorflow as tf

# Define a function to generate Grad-CAM
def generate_grad_cam(model, sequence, class_index):
    # Expand dimensions to make it compatible with model input shape
    sequence = np.expand_dims(sequence, axis=0)
    
    # Get the output of the last LSTM layer
    lstm_output = model.layers[0](sequence)
    
    # Get the output of the Dense layer
    dense_output = model.layers[1](lstm_output)
    
    # Compute the gradient of the class output with respect to the Dense layer output
    with tf.GradientTape() as tape:
        tape.watch(dense_output)
        loss = dense_output[:, class_index]
    grads = tape.gradient(loss, dense_output)
    
    # Get the weights of the Dense layer
    dense_weights = model.layers[1].get_weights()[0]
    
    # Compute the importance of each feature map
    importance = tf.reduce_mean(grads * dense_weights, axis=(0, 1))
    
    # Generate the heatmap
    heatmap = np.dot(importance, lstm_output.numpy().squeeze())
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    
    return heatmap

# Choose a sample from the test set
sample_index = 0  # Choose any sample index from the test set
sample_sequence = X_test_reshaped[sample_index]
sample_label = y_test[sample_index]

# Generate the Grad-CAM heatmap
heatmap = generate_grad_cam(model, sample_sequence, class_index=0)  # Assuming binary classification

# Plot the saliency map (heatmap)
plt.plot(heatmap)
plt.title('Saliency Map (Heatmap)')
plt.xlabel('Time Steps')
plt.ylabel('Importance')
plt.show()


# %%
import numpy as np
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

def permutation_importance(model, X, y, metric=mean_squared_error, n_iterations=100):
    baseline_score = metric(y, model.predict(X))
    feature_importances = {}

    for feature_idx in tqdm(range(X.shape[1]), desc='Calculating Permutation Importances'):
        shuffled_X = X.copy()
        np.random.shuffle(shuffled_X[:, feature_idx])
        permuted_score = 0
        for _ in range(n_iterations):
            permuted_score += metric(y, model.predict(shuffled_X))
        feature_importances[feature_idx] = baseline_score - permuted_score / n_iterations

    return feature_importances

# Calculate permutation importance
perm_importances = permutation_importance(model, X_test_reshaped, y_test)

# Print feature importances
for i, importance in sorted(perm_importances.items(), key=lambda x: x[1], reverse=True):
    print(f'Feature {i}: {importance}')


# %%
import numpy as np
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

def permutation_importance(model, X, y, metric=mean_squared_error, n_iterations=100):
    baseline_score = metric(y, model.predict(X))
    feature_importances = {}

    for feature_idx in tqdm(range(X.shape[2]), desc='Calculating Permutation Importances'):
        permuted_scores = []
        for _ in range(n_iterations):
            shuffled_X = X.copy()
            np.random.shuffle(shuffled_X[:, :, feature_idx])
            permuted_score = metric(y, model.predict(shuffled_X))
            permuted_scores.append(permuted_score)
        feature_importances[feature_idx] = baseline_score - np.mean(permuted_scores)

    return feature_importances

# Calculate permutation importance
perm_importances = permutation_importance(model, X_test_reshaped, y_test)

# Print feature importances
for i, importance in sorted(perm_importances.items(), key=lambda x: x[1], reverse=True):
    print(f'Feature {i}: {importance}')


# %%
import matplotlib.pyplot as plt

# Make predictions on the test set
y_pred = model.predict(X_test_reshaped)

# Plot actual vs predicted values
plt.figure(figsize=(10, 6))
plt.plot(y_test, label='Actual Opening Price', color='blue')
plt.plot(y_pred, label='Predicted Opening Price', color='red')
plt.title('Actual vs Predicted Opening Price')
plt.xlabel('Time')
plt.ylabel('Opening Price')
plt.legend()
plt.show()


# %%
import matplotlib.pyplot as plt

# Make predictions on the test set
y_pred = model.predict(X_test_reshaped)

# Calculate differences between consecutive predicted opening prices
predicted_growth = [y_pred[i+1] - y_pred[i] for i in range(len(y_pred)-1)]

# Get dates for plotting
dates = BEXIMCO.index[train_size + val_size + 1:]  # Assuming dates are in the index

# Plot predicted growth over time (date)
plt.figure(figsize=(10, 6))
plt.plot(dates, predicted_growth, label='Predicted Growth', color='green')
plt.title('Predicted Growth Over Time')
plt.xlabel('Date')
plt.ylabel('Predicted Growth')
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.legend()
plt.show()


# %% [markdown]
# **Data visualization for UPGDCL**

# %%
import matplotlib.pyplot as plt

plt.figure(figsize=(5, 3))
plt.plot(UPGDCL.index, UPGDCL['opening_price'], marker='o',markersize=1, linestyle='-')
plt.title('Opening Price Over Time')
plt.xlabel('Date')
plt.ylabel('Opening Price')
plt.grid(True)
plt.tight_layout()
plt.show()


# %%
from sklearn.preprocessing import MinMaxScaler

# Define the features and target variable
features = ['opening_price', 'closing_price', 'yesterdays_closing_price', 'trade', 'value_mn', 'volume']
X = UPGDCL[features].values
y = UPGDCL['opening_price'].values

# Scale the features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Define the sizes for training, validation, and testing sets
train_size = int(0.8 * len(X))  # 80% for training
val_size = int(0.1 * len(X))    # 10% for validation
test_size = len(X) - train_size - val_size  # Remaining 10% for testing

# Split the features and target variable accordingly
X_train = X_scaled[:train_size]
X_val = X_scaled[train_size:train_size+val_size]
X_test = X_scaled[train_size+val_size:]

y_train = y[:train_size]
y_val = y[train_size:train_size+val_size]
y_test = y[train_size+val_size:]

# Reshape the data for LSTM input (assuming a time step of 1)
X_train_reshaped = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
X_val_reshaped = X_val.reshape(X_val.shape[0], 1, X_val.shape[1])
X_test_reshaped = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

# Now you can use X_train_reshaped, y_train, X_val_reshaped, y_val, X_test_reshaped, and y_test for training and evaluating your LSTM model

# %%
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.optimizers import Adam
import numpy as np

# Define the LSTM model
model = Sequential([
    LSTM(units=50, input_shape=(X_train_reshaped.shape[1], X_train_reshaped.shape[2])),
    Dense(units=1)
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

# Train the model
history = model.fit(X_train_reshaped, y_train, epochs=50, batch_size=32, validation_data=(X_val_reshaped, y_val))

# Evaluate the model on the test set
loss = model.evaluate(X_test_reshaped, y_test)
print(f'Test Loss: {loss}')

# Calculate and print RMSE
y_pred = model.predict(X_test_reshaped)
rmse = np.sqrt(np.mean((y_pred - y_test)**2))
print(f'RMSE: {rmse}')

# Plot training history
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# %% [markdown]
# > **3 Day Forecast**

# %%
import numpy as np
import matplotlib.pyplot as plt

# Assuming X_scaled, model, and y are defined

# Get the last available data point
last_data_point = X_scaled[-1].reshape(1, 1, X_scaled.shape[1])

# Predict the next three days
forecast = []
for _ in range(3):
    next_day_prediction = model.predict(last_data_point)
    forecast.append(next_day_prediction)
    # Update last_data_point to include the predicted value for the next day
    last_data_point = np.concatenate([last_data_point, next_day_prediction.reshape(1, 1, 1)], axis=2)

# Extract opening prices from the forecast
forecast_opening_prices = np.array(forecast).reshape(-1)

print("Forecasted opening prices for the next three days:")
print(forecast_opening_prices)

# Number of original days
n_days_original = len(y)

# Plot forecasted opening prices for the next three days without connecting lines
forecast_days = np.arange(n_days_original, n_days_original + 3)
plt.plot(forecast_days, forecast_opening_prices, marker='o', linestyle='None', label='Forecasted Data')

plt.title('Three-Day Forecast')
plt.xlabel('Day')
plt.ylabel('Price')
plt.xticks(np.arange(n_days_original, n_days_original + 3), ['Day 1', 'Day 2', 'Day 3'])
plt.legend()
plt.show()


# %% [markdown]
# > **7 Day Forecast**

# %%
import numpy as np
import matplotlib.pyplot as plt

# Assuming X_scaled, model, and y are defined

# Get the last available data point
last_data_point = X_scaled[-1].reshape(1, 1, X_scaled.shape[1])

# Predict the next seven days
forecast = []
for _ in range(7):
    next_day_prediction = model.predict(last_data_point)
    forecast.append(next_day_prediction)
    # Update last_data_point to include the predicted value for the next day
    last_data_point = np.concatenate([last_data_point, next_day_prediction.reshape(1, 1, 1)], axis=2)

# Extract opening prices from the forecast
forecast_opening_prices = np.array(forecast).reshape(-1)

print("Forecasted opening prices for the next seven days:")
print(forecast_opening_prices)

# Plot forecasted opening prices for the next seven days without connecting lines
forecast_days = np.arange(n_days_original, n_days_original + 7)
plt.plot(forecast_days, forecast_opening_prices, marker='o', linestyle='None', label='Forecasted Data')

plt.title('Seven-Day Forecast')
plt.xlabel('Day')
plt.ylabel('Price')
plt.xticks(np.arange(n_days_original, n_days_original + 7), ['Day 1', 'Day 2', 'Day 3', 'Day 4', 'Day 5', 'Day 6', 'Day 7'])
plt.legend()
plt.show()


# %% [markdown]
# > **30 Day Forecast**

# %%
import numpy as np
import matplotlib.pyplot as plt

# Assuming X_scaled, model, and y are defined

# Get the last available data point
last_data_point = X_scaled[-1].reshape(1, 1, X_scaled.shape[1])

# Predict the next 30 days
forecast = []
for _ in range(30):
    next_day_prediction = model.predict(last_data_point)
    forecast.append(next_day_prediction)
    # Update last_data_point to include the predicted value for the next day
    last_data_point = np.concatenate([last_data_point, next_day_prediction.reshape(1, 1, 1)], axis=2)

# Extract opening prices from the forecast
forecast_opening_prices = np.array(forecast).reshape(-1)

print("Forecasted opening prices for the next 30 days:")
print(forecast_opening_prices)

# Plot forecasted opening prices for the next 30 days without connecting lines
forecast_days = np.arange(n_days_original, n_days_original + 30)
plt.plot(forecast_days, forecast_opening_prices, marker='o', linestyle='None', label='Forecasted Data')

plt.title('Thirty-Day Forecast')
plt.xlabel('Day')

# Labeling x-axis ticks for days 5, 10, 15, 20, 25, and 30
x_ticks = [5, 10, 15, 20, 25, 29]  # Corrected the indexing
plt.xticks(np.arange(n_days_original, n_days_original + 30)[x_ticks], [f'Day {i}' for i in x_ticks])

plt.ylabel('Price')
plt.legend()
plt.show()


# %% [markdown]
# **XAI**

# %%
import matplotlib.pyplot as plt
import numpy as np
import keras.backend as K
import tensorflow as tf

# Define a function to generate Grad-CAM
def generate_grad_cam(model, sequence, class_index):
    # Expand dimensions to make it compatible with model input shape
    sequence = np.expand_dims(sequence, axis=0)
    
    # Get the output of the last LSTM layer
    lstm_output = model.layers[0](sequence)
    
    # Get the output of the Dense layer
    dense_output = model.layers[1](lstm_output)
    
    # Compute the gradient of the class output with respect to the Dense layer output
    with tf.GradientTape() as tape:
        tape.watch(dense_output)
        loss = dense_output[:, class_index]
    grads = tape.gradient(loss, dense_output)
    
    # Get the weights of the Dense layer
    dense_weights = model.layers[1].get_weights()[0]
    
    # Compute the importance of each feature map
    importance = tf.reduce_mean(grads * dense_weights, axis=(0, 1))
    
    # Generate the heatmap
    heatmap = np.dot(importance, lstm_output.numpy().squeeze())
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    
    return heatmap

# Choose a sample from the test set
sample_index = 0  # Choose any sample index from the test set
sample_sequence = X_test_reshaped[sample_index]
sample_label = y_test[sample_index]

# Generate the Grad-CAM heatmap
heatmap = generate_grad_cam(model, sample_sequence, class_index=0)  # Assuming binary classification

# Plot the saliency map (heatmap)
plt.plot(heatmap)
plt.title('Saliency Map (Heatmap)')
plt.xlabel('Time Steps')
plt.ylabel('Importance')
plt.show()


# %%
import numpy as np
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

def permutation_importance(model, X, y, metric=mean_squared_error, n_iterations=100):
    baseline_score = metric(y, model.predict(X))
    feature_importances = {}

    for feature_idx in tqdm(range(X.shape[1]), desc='Calculating Permutation Importances'):
        shuffled_X = X.copy()
        np.random.shuffle(shuffled_X[:, feature_idx])
        permuted_score = 0
        for _ in range(n_iterations):
            permuted_score += metric(y, model.predict(shuffled_X))
        feature_importances[feature_idx] = baseline_score - permuted_score / n_iterations

    return feature_importances

# Calculate permutation importance
perm_importances = permutation_importance(model, X_test_reshaped, y_test)

# Print feature importances
for i, importance in sorted(perm_importances.items(), key=lambda x: x[1], reverse=True):
    print(f'Feature {i}: {importance}')


# %%
import numpy as np
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

def permutation_importance(model, X, y, metric=mean_squared_error, n_iterations=100):
    baseline_score = metric(y, model.predict(X))
    feature_importances = {}

    for feature_idx in tqdm(range(X.shape[2]), desc='Calculating Permutation Importances'):
        permuted_scores = []
        for _ in range(n_iterations):
            shuffled_X = X.copy()
            np.random.shuffle(shuffled_X[:, :, feature_idx])
            permuted_score = metric(y, model.predict(shuffled_X))
            permuted_scores.append(permuted_score)
        feature_importances[feature_idx] = baseline_score - np.mean(permuted_scores)

    return feature_importances

# Calculate permutation importance
perm_importances = permutation_importance(model, X_test_reshaped, y_test)

# Print feature importances
for i, importance in sorted(perm_importances.items(), key=lambda x: x[1], reverse=True):
    print(f'Feature {i}: {importance}')


# %%
import matplotlib.pyplot as plt

# Make predictions on the test set
y_pred = model.predict(X_test_reshaped)

# Plot actual vs predicted values
plt.figure(figsize=(10, 6))
plt.plot(y_test, label='Actual Opening Price', color='blue')
plt.plot(y_pred, label='Predicted Opening Price', color='red')
plt.title('Actual vs Predicted Opening Price')
plt.xlabel('Time')
plt.ylabel('Opening Price')
plt.legend()
plt.show()


# %%
import matplotlib.pyplot as plt

# Make predictions on the test set
y_pred = model.predict(X_test_reshaped)

# Calculate differences between consecutive predicted opening prices
predicted_growth = [y_pred[i+1] - y_pred[i] for i in range(len(y_pred)-1)]

# Get dates for plotting
dates = UPGDCL.index[train_size + val_size + 1:]  # Assuming dates are in the index

# Plot predicted growth over time (date)
plt.figure(figsize=(10, 6))
plt.plot(dates, predicted_growth, label='Predicted Growth', color='green')
plt.title('Predicted Growth Over Time')
plt.xlabel('Date')
plt.ylabel('Predicted Growth')
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.legend()
plt.show()


# %% [markdown]
# **Data visualization for BERGERPBL**

# %%
import matplotlib.pyplot as plt

plt.figure(figsize=(5, 3)) 
plt.plot(BERGERPBL.index, BERGERPBL['opening_price'], marker='o',markersize=1, linestyle='-')
plt.title('Opening Price Over Time')
plt.xlabel('Date')
plt.ylabel('Opening Price')
plt.grid(True)
plt.tight_layout() 
plt.show()


# %%
from sklearn.preprocessing import MinMaxScaler

# Define the features and target variable
features = ['opening_price', 'closing_price', 'yesterdays_closing_price', 'trade', 'value_mn', 'volume']
X = BERGERPBL[features].values
y = BERGERPBL['opening_price'].values

# Scale the features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Define the sizes for training, validation, and testing sets
train_size = int(0.8 * len(X))  # 80% for training
val_size = int(0.1 * len(X))    # 10% for validation
test_size = len(X) - train_size - val_size  # Remaining 10% for testing

# Split the features and target variable accordingly
X_train = X_scaled[:train_size]
X_val = X_scaled[train_size:train_size+val_size]
X_test = X_scaled[train_size+val_size:]

y_train = y[:train_size]
y_val = y[train_size:train_size+val_size]
y_test = y[train_size+val_size:]

# Reshape the data for LSTM input (assuming a time step of 1)
X_train_reshaped = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
X_val_reshaped = X_val.reshape(X_val.shape[0], 1, X_val.shape[1])
X_test_reshaped = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

# Now you can use X_train_reshaped, y_train, X_val_reshaped, y_val, X_test_reshaped, and y_test for training and evaluating your LSTM model

# %%
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.optimizers import Adam
import numpy as np

# Define the LSTM model
model = Sequential([
    LSTM(units=50, input_shape=(X_train_reshaped.shape[1], X_train_reshaped.shape[2])),
    Dense(units=1)
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

# Train the model
history = model.fit(X_train_reshaped, y_train, epochs=50, batch_size=32, validation_data=(X_val_reshaped, y_val))

# Evaluate the model on the test set
loss = model.evaluate(X_test_reshaped, y_test)
print(f'Test Loss: {loss}')

# Calculate and print RMSE
y_pred = model.predict(X_test_reshaped)
rmse = np.sqrt(np.mean((y_pred - y_test)**2))
print(f'RMSE: {rmse}')

# Plot training history
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# %% [markdown]
# > **3 Day Forecast**

# %%
import numpy as np
import matplotlib.pyplot as plt

# Assuming X_scaled, model, and y are defined

# Get the last available data point
last_data_point = X_scaled[-1].reshape(1, 1, X_scaled.shape[1])

# Predict the next three days
forecast = []
for _ in range(3):
    next_day_prediction = model.predict(last_data_point)
    forecast.append(next_day_prediction)
    # Update last_data_point to include the predicted value for the next day
    last_data_point = np.concatenate([last_data_point, next_day_prediction.reshape(1, 1, 1)], axis=2)

# Extract opening prices from the forecast
forecast_opening_prices = np.array(forecast).reshape(-1)

print("Forecasted opening prices for the next three days:")
print(forecast_opening_prices)

# Number of original days
n_days_original = len(y)

# Plot forecasted opening prices for the next three days without connecting lines
forecast_days = np.arange(n_days_original, n_days_original + 3)
plt.plot(forecast_days, forecast_opening_prices, marker='o', linestyle='None', label='Forecasted Data')

plt.title('Three-Day Forecast')
plt.xlabel('Day')
plt.ylabel('Price')
plt.xticks(np.arange(n_days_original, n_days_original + 3), ['Day 1', 'Day 2', 'Day 3'])
plt.legend()
plt.show()


# %% [markdown]
# > **7 Day Forecast**

# %%
import numpy as np
import matplotlib.pyplot as plt

# Assuming X_scaled, model, and y are defined

# Get the last available data point
last_data_point = X_scaled[-1].reshape(1, 1, X_scaled.shape[1])

# Predict the next seven days
forecast = []
for _ in range(7):
    next_day_prediction = model.predict(last_data_point)
    forecast.append(next_day_prediction)
    # Update last_data_point to include the predicted value for the next day
    last_data_point = np.concatenate([last_data_point, next_day_prediction.reshape(1, 1, 1)], axis=2)

# Extract opening prices from the forecast
forecast_opening_prices = np.array(forecast).reshape(-1)

print("Forecasted opening prices for the next seven days:")
print(forecast_opening_prices)

# Plot forecasted opening prices for the next seven days without connecting lines
forecast_days = np.arange(n_days_original, n_days_original + 7)
plt.plot(forecast_days, forecast_opening_prices, marker='o', linestyle='None', label='Forecasted Data')

plt.title('Seven-Day Forecast')
plt.xlabel('Day')
plt.ylabel('Price')
plt.xticks(np.arange(n_days_original, n_days_original + 7), ['Day 1', 'Day 2', 'Day 3', 'Day 4', 'Day 5', 'Day 6', 'Day 7'])
plt.legend()
plt.show()


# %% [markdown]
# > **30 Day Forecast**

# %%
import numpy as np
import matplotlib.pyplot as plt

# Assuming X_scaled, model, and y are defined

# Get the last available data point
last_data_point = X_scaled[-1].reshape(1, 1, X_scaled.shape[1])

# Predict the next 30 days
forecast = []
for _ in range(30):
    next_day_prediction = model.predict(last_data_point)
    forecast.append(next_day_prediction)
    # Update last_data_point to include the predicted value for the next day
    last_data_point = np.concatenate([last_data_point, next_day_prediction.reshape(1, 1, 1)], axis=2)

# Extract opening prices from the forecast
forecast_opening_prices = np.array(forecast).reshape(-1)

print("Forecasted opening prices for the next 30 days:")
print(forecast_opening_prices)

# Plot forecasted opening prices for the next 30 days without connecting lines
forecast_days = np.arange(n_days_original, n_days_original + 30)
plt.plot(forecast_days, forecast_opening_prices, marker='o', linestyle='None', label='Forecasted Data')

plt.title('Thirty-Day Forecast')
plt.xlabel('Day')

# Labeling x-axis ticks for days 5, 10, 15, 20, 25, and 30
x_ticks = [5, 10, 15, 20, 25, 29]  # Corrected the indexing
plt.xticks(np.arange(n_days_original, n_days_original + 30)[x_ticks], [f'Day {i}' for i in x_ticks])

plt.ylabel('Price')
plt.legend()
plt.show()


# %% [markdown]
# **XAI**

# %%
import matplotlib.pyplot as plt
import numpy as np
import keras.backend as K
import tensorflow as tf

# Define a function to generate Grad-CAM
def generate_grad_cam(model, sequence, class_index):
    # Expand dimensions to make it compatible with model input shape
    sequence = np.expand_dims(sequence, axis=0)
    
    # Get the output of the last LSTM layer
    lstm_output = model.layers[0](sequence)
    
    # Get the output of the Dense layer
    dense_output = model.layers[1](lstm_output)
    
    # Compute the gradient of the class output with respect to the Dense layer output
    with tf.GradientTape() as tape:
        tape.watch(dense_output)
        loss = dense_output[:, class_index]
    grads = tape.gradient(loss, dense_output)
    
    # Get the weights of the Dense layer
    dense_weights = model.layers[1].get_weights()[0]
    
    # Compute the importance of each feature map
    importance = tf.reduce_mean(grads * dense_weights, axis=(0, 1))
    
    # Generate the heatmap
    heatmap = np.dot(importance, lstm_output.numpy().squeeze())
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    
    return heatmap

# Choose a sample from the test set
sample_index = 0  # Choose any sample index from the test set
sample_sequence = X_test_reshaped[sample_index]
sample_label = y_test[sample_index]

# Generate the Grad-CAM heatmap
heatmap = generate_grad_cam(model, sample_sequence, class_index=0)  # Assuming binary classification

# Plot the saliency map (heatmap)
plt.plot(heatmap)
plt.title('Saliency Map (Heatmap)')
plt.xlabel('Time Steps')
plt.ylabel('Importance')
plt.show()


# %%
import numpy as np
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

def permutation_importance(model, X, y, metric=mean_squared_error, n_iterations=100):
    baseline_score = metric(y, model.predict(X))
    feature_importances = {}

    for feature_idx in tqdm(range(X.shape[1]), desc='Calculating Permutation Importances'):
        shuffled_X = X.copy()
        np.random.shuffle(shuffled_X[:, feature_idx])
        permuted_score = 0
        for _ in range(n_iterations):
            permuted_score += metric(y, model.predict(shuffled_X))
        feature_importances[feature_idx] = baseline_score - permuted_score / n_iterations

    return feature_importances

# Calculate permutation importance
perm_importances = permutation_importance(model, X_test_reshaped, y_test)

# Print feature importances
for i, importance in sorted(perm_importances.items(), key=lambda x: x[1], reverse=True):
    print(f'Feature {i}: {importance}')


# %%
import numpy as np
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

def permutation_importance(model, X, y, metric=mean_squared_error, n_iterations=100):
    baseline_score = metric(y, model.predict(X))
    feature_importances = {}

    for feature_idx in tqdm(range(X.shape[2]), desc='Calculating Permutation Importances'):
        permuted_scores = []
        for _ in range(n_iterations):
            shuffled_X = X.copy()
            np.random.shuffle(shuffled_X[:, :, feature_idx])
            permuted_score = metric(y, model.predict(shuffled_X))
            permuted_scores.append(permuted_score)
        feature_importances[feature_idx] = baseline_score - np.mean(permuted_scores)

    return feature_importances

# Calculate permutation importance
perm_importances = permutation_importance(model, X_test_reshaped, y_test)

# Print feature importances
for i, importance in sorted(perm_importances.items(), key=lambda x: x[1], reverse=True):
    print(f'Feature {i}: {importance}')


# %%
import matplotlib.pyplot as plt

# Make predictions on the test set
y_pred = model.predict(X_test_reshaped)

# Plot actual vs predicted values
plt.figure(figsize=(10, 6))
plt.plot(y_test, label='Actual Opening Price', color='blue')
plt.plot(y_pred, label='Predicted Opening Price', color='red')
plt.title('Actual vs Predicted Opening Price')
plt.xlabel('Time')
plt.ylabel('Opening Price')
plt.legend()
plt.show()


# %%
import matplotlib.pyplot as plt

# Make predictions on the test set
y_pred = model.predict(X_test_reshaped)

# Calculate differences between consecutive predicted opening prices
predicted_growth = [y_pred[i+1] - y_pred[i] for i in range(len(y_pred)-1)]

# Get dates for plotting
dates = BERGERPBL.index[train_size + val_size + 1:]  # Assuming dates are in the index

# Plot predicted growth over time (date)
plt.figure(figsize=(10, 6))
plt.plot(dates, predicted_growth, label='Predicted Growth', color='green')
plt.title('Predicted Growth Over Time')
plt.xlabel('Date')
plt.ylabel('Predicted Growth')
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.legend()
plt.show()


# %% [markdown]
# **Data visualization for LHBL**

# %%
import matplotlib.pyplot as plt

plt.figure(figsize=(5, 3)) 
plt.plot(LHBL.index, LHBL['opening_price'], marker='o',markersize=1, linestyle='-')
plt.title('Opening Price Over Time')
plt.xlabel('Date')
plt.ylabel('Opening Price')
plt.grid(True)
plt.tight_layout() 
plt.show()


# %%
from sklearn.preprocessing import MinMaxScaler

# Define the features and target variable
features = ['opening_price', 'closing_price', 'yesterdays_closing_price', 'trade', 'value_mn', 'volume']
X = LHBL[features].values
y = LHBL['opening_price'].values

# Scale the features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Define the sizes for training, validation, and testing sets
train_size = int(0.8 * len(X))  # 80% for training
val_size = int(0.1 * len(X))    # 10% for validation
test_size = len(X) - train_size - val_size  # Remaining 10% for testing

# Split the features and target variable accordingly
X_train = X_scaled[:train_size]
X_val = X_scaled[train_size:train_size+val_size]
X_test = X_scaled[train_size+val_size:]

y_train = y[:train_size]
y_val = y[train_size:train_size+val_size]
y_test = y[train_size+val_size:]

# Reshape the data for LSTM input (assuming a time step of 1)
X_train_reshaped = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
X_val_reshaped = X_val.reshape(X_val.shape[0], 1, X_val.shape[1])
X_test_reshaped = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

# Now you can use X_train_reshaped, y_train, X_val_reshaped, y_val, X_test_reshaped, and y_test for training and evaluating your LSTM model

# %%
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.optimizers import Adam
import numpy as np

# Define the LSTM model
model = Sequential([
    LSTM(units=50, input_shape=(X_train_reshaped.shape[1], X_train_reshaped.shape[2])),
    Dense(units=1)
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

# Train the model
history = model.fit(X_train_reshaped, y_train, epochs=50, batch_size=32, validation_data=(X_val_reshaped, y_val))

# Evaluate the model on the test set
loss = model.evaluate(X_test_reshaped, y_test)
print(f'Test Loss: {loss}')

# Calculate and print RMSE
y_pred = model.predict(X_test_reshaped)
rmse = np.sqrt(np.mean((y_pred - y_test)**2))
print(f'RMSE: {rmse}')

# Plot training history
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# %% [markdown]
# > **3 Day Forecast**

# %%
import numpy as np
import matplotlib.pyplot as plt

# Assuming X_scaled, model, and y are defined

# Get the last available data point
last_data_point = X_scaled[-1].reshape(1, 1, X_scaled.shape[1])

# Predict the next three days
forecast = []
for _ in range(3):
    next_day_prediction = model.predict(last_data_point)
    forecast.append(next_day_prediction)
    # Update last_data_point to include the predicted value for the next day
    last_data_point = np.concatenate([last_data_point, next_day_prediction.reshape(1, 1, 1)], axis=2)

# Extract opening prices from the forecast
forecast_opening_prices = np.array(forecast).reshape(-1)

print("Forecasted opening prices for the next three days:")
print(forecast_opening_prices)

# Number of original days
n_days_original = len(y)

# Plot forecasted opening prices for the next three days without connecting lines
forecast_days = np.arange(n_days_original, n_days_original + 3)
plt.plot(forecast_days, forecast_opening_prices, marker='o', linestyle='None', label='Forecasted Data')

plt.title('Three-Day Forecast')
plt.xlabel('Day')
plt.ylabel('Price')
plt.xticks(np.arange(n_days_original, n_days_original + 3), ['Day 1', 'Day 2', 'Day 3'])
plt.legend()
plt.show()


# %% [markdown]
# > **7 Day Forecast**

# %%
import numpy as np
import matplotlib.pyplot as plt

# Assuming X_scaled, model, and y are defined

# Get the last available data point
last_data_point = X_scaled[-1].reshape(1, 1, X_scaled.shape[1])

# Predict the next seven days
forecast = []
for _ in range(7):
    next_day_prediction = model.predict(last_data_point)
    forecast.append(next_day_prediction)
    # Update last_data_point to include the predicted value for the next day
    last_data_point = np.concatenate([last_data_point, next_day_prediction.reshape(1, 1, 1)], axis=2)

# Extract opening prices from the forecast
forecast_opening_prices = np.array(forecast).reshape(-1)

print("Forecasted opening prices for the next seven days:")
print(forecast_opening_prices)

# Plot forecasted opening prices for the next seven days without connecting lines
forecast_days = np.arange(n_days_original, n_days_original + 7)
plt.plot(forecast_days, forecast_opening_prices, marker='o', linestyle='None', label='Forecasted Data')

plt.title('Seven-Day Forecast')
plt.xlabel('Day')
plt.ylabel('Price')
plt.xticks(np.arange(n_days_original, n_days_original + 7), ['Day 1', 'Day 2', 'Day 3', 'Day 4', 'Day 5', 'Day 6', 'Day 7'])
plt.legend()
plt.show()


# %% [markdown]
# > **30 Day Forecast**

# %%
import numpy as np
import matplotlib.pyplot as plt

# Assuming X_scaled, model, and y are defined

# Get the last available data point
last_data_point = X_scaled[-1].reshape(1, 1, X_scaled.shape[1])

# Predict the next 30 days
forecast = []
for _ in range(30):
    next_day_prediction = model.predict(last_data_point)
    forecast.append(next_day_prediction)
    # Update last_data_point to include the predicted value for the next day
    last_data_point = np.concatenate([last_data_point, next_day_prediction.reshape(1, 1, 1)], axis=2)

# Extract opening prices from the forecast
forecast_opening_prices = np.array(forecast).reshape(-1)

print("Forecasted opening prices for the next 30 days:")
print(forecast_opening_prices)

# Plot forecasted opening prices for the next 30 days without connecting lines
forecast_days = np.arange(n_days_original, n_days_original + 30)
plt.plot(forecast_days, forecast_opening_prices, marker='o', linestyle='None', label='Forecasted Data')

plt.title('Thirty-Day Forecast')
plt.xlabel('Day')

# Labeling x-axis ticks for days 5, 10, 15, 20, 25, and 30
x_ticks = [5, 10, 15, 20, 25, 29]  # Corrected the indexing
plt.xticks(np.arange(n_days_original, n_days_original + 30)[x_ticks], [f'Day {i}' for i in x_ticks])

plt.ylabel('Price')
plt.legend()
plt.show()


# %% [markdown]
# **XAI**

# %%
import matplotlib.pyplot as plt
import numpy as np
import keras.backend as K
import tensorflow as tf

# Define a function to generate Grad-CAM
def generate_grad_cam(model, sequence, class_index):
    # Expand dimensions to make it compatible with model input shape
    sequence = np.expand_dims(sequence, axis=0)
    
    # Get the output of the last LSTM layer
    lstm_output = model.layers[0](sequence)
    
    # Get the output of the Dense layer
    dense_output = model.layers[1](lstm_output)
    
    # Compute the gradient of the class output with respect to the Dense layer output
    with tf.GradientTape() as tape:
        tape.watch(dense_output)
        loss = dense_output[:, class_index]
    grads = tape.gradient(loss, dense_output)
    
    # Get the weights of the Dense layer
    dense_weights = model.layers[1].get_weights()[0]
    
    # Compute the importance of each feature map
    importance = tf.reduce_mean(grads * dense_weights, axis=(0, 1))
    
    # Generate the heatmap
    heatmap = np.dot(importance, lstm_output.numpy().squeeze())
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    
    return heatmap

# Choose a sample from the test set
sample_index = 0  # Choose any sample index from the test set
sample_sequence = X_test_reshaped[sample_index]
sample_label = y_test[sample_index]

# Generate the Grad-CAM heatmap
heatmap = generate_grad_cam(model, sample_sequence, class_index=0)  # Assuming binary classification

# Plot the saliency map (heatmap)
plt.plot(heatmap)
plt.title('Saliency Map (Heatmap)')
plt.xlabel('Time Steps')
plt.ylabel('Importance')
plt.show()


# %%
import numpy as np
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

def permutation_importance(model, X, y, metric=mean_squared_error, n_iterations=100):
    baseline_score = metric(y, model.predict(X))
    feature_importances = {}

    for feature_idx in tqdm(range(X.shape[1]), desc='Calculating Permutation Importances'):
        shuffled_X = X.copy()
        np.random.shuffle(shuffled_X[:, feature_idx])
        permuted_score = 0
        for _ in range(n_iterations):
            permuted_score += metric(y, model.predict(shuffled_X))
        feature_importances[feature_idx] = baseline_score - permuted_score / n_iterations

    return feature_importances

# Calculate permutation importance
perm_importances = permutation_importance(model, X_test_reshaped, y_test)

# Print feature importances
for i, importance in sorted(perm_importances.items(), key=lambda x: x[1], reverse=True):
    print(f'Feature {i}: {importance}')


# %%
import numpy as np
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

def permutation_importance(model, X, y, metric=mean_squared_error, n_iterations=100):
    baseline_score = metric(y, model.predict(X))
    feature_importances = {}

    for feature_idx in tqdm(range(X.shape[2]), desc='Calculating Permutation Importances'):
        permuted_scores = []
        for _ in range(n_iterations):
            shuffled_X = X.copy()
            np.random.shuffle(shuffled_X[:, :, feature_idx])
            permuted_score = metric(y, model.predict(shuffled_X))
            permuted_scores.append(permuted_score)
        feature_importances[feature_idx] = baseline_score - np.mean(permuted_scores)

    return feature_importances

# Calculate permutation importance
perm_importances = permutation_importance(model, X_test_reshaped, y_test)

# Print feature importances
for i, importance in sorted(perm_importances.items(), key=lambda x: x[1], reverse=True):
    print(f'Feature {i}: {importance}')


# %%
import matplotlib.pyplot as plt

# Make predictions on the test set
y_pred = model.predict(X_test_reshaped)

# Plot actual vs predicted values
plt.figure(figsize=(10, 6))
plt.plot(y_test, label='Actual Opening Price', color='blue')
plt.plot(y_pred, label='Predicted Opening Price', color='red')
plt.title('Actual vs Predicted Opening Price')
plt.xlabel('Time')
plt.ylabel('Opening Price')
plt.legend()
plt.show()


# %%
import matplotlib.pyplot as plt

# Make predictions on the test set
y_pred = model.predict(X_test_reshaped)

# Calculate differences between consecutive predicted opening prices
predicted_growth = [y_pred[i+1] - y_pred[i] for i in range(len(y_pred)-1)]

# Get dates for plotting
dates = LHBL.index[train_size + val_size + 1:]  # Assuming dates are in the index

# Plot predicted growth over time (date)
plt.figure(figsize=(10, 6))
plt.plot(dates, predicted_growth, label='Predicted Growth', color='green')
plt.title('Predicted Growth Over Time')
plt.xlabel('Date')
plt.ylabel('Predicted Growth')
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.legend()
plt.show()



