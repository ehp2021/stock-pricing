#colab: https://colab.research.google.com/drive/1vXZbwxf-OUnffqTZU6l9bbVEOfIVuQA8#scrollTo=Cc29GUSmaRh3


import math
import pandas_datareader as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt 
plt.style.use('fivethirtyeight')
!pip install yfinance



#from pandas_datareader import data as pdr

import yfinance as yf
#yf.pdr_override()

#get stock quote
df = yf.download("SPY", start='2019-01-02', end="2022-10-24")

#show the data
df


df.shape

#visualize hte closing price history
plt.figure(figsize=(16,8))
plt.title("close price history")
plt.plot(df['Close'])
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD', fontsize=18)
plt.show()

#create a new dataframe iwth only the close column
data = df.filter(['Close'])
#convert the dataframe
dataset = data.values
#get the # of rows to train our LSTM model on
# this gives us 80% of the data set.. should be 80% of 960
#math.ceil rounds it up 
training_data_len = math.ceil(len(dataset) * 0.8)

training_data_len 



#scale the data - in practice it's always advantageous to use preprocessing or scsaling or nomralization before presenting to neural network
scaler = MinMaxScaler(feature_range=(0,1))
#transform the data to be between 0 and 1
#scaled data will hodl the dataset taht is scaled, between 0 and 1
scaled_data = scaler.fit_transform(dataset)
scaled_data

#create the training data set 
#create the scaled training dataset 
#contains all the data from 0 to training_data_len
train_data = scaled_data[0:training_data_len, :]
#split hte data into x_train and y_train data sets

x_train = [] #independent variable 
y_train = [] #dependent or target variable

for i in range(60, len(train_data)):
  x_train.append(train_data[i-60:i, 0]) #not including i.. position 0 to 59
  y_train.append(train_data[i, 0]) #includes first 60 values... position 60th value
  if i<=61:
    print(x_train)
    print(y_train)
    print(

#convert x_train and y_train to numpy arrays to train the models
x_train, y_train = np.array(x_train), np.array(y_train)

#reshape teh x_train data set .. LSTM expects input to be 3D in time steps, features
# and right now our data is 2D
#x_train.shape #it's currently 2D i.e. (708, 60)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_train.shape

#build the LSTM model
model = Sequential()
# give it 50 neurons
model.add(LSTM(50, return_sequences=True, input_shape = (x_train.shape[1], 1)))
model.add(LSTM(50, return_sequences=False))
#add a dense layer with 25 neurons
model.add(Dense(25))
model.add(Dense(1))

#compile the model
#optimizer is used to improve upon the loss fxn, and the loss fxn 
#tells us how well the model did in training
model.compile(optimizer = 'adam', loss='mean_squared_error')

#train or fit the model
#batch size is the total # of training exmaples presnt in a batch
# epochs is # of iterations passed forward and backwards thru neural network
model.fit(x_train, y_train, batch_size=1, epochs=1)


#Create the testing data set 
#create a new array containing scaled values from index 708 to 960 (total dataset)
test_data = scaled_data[training_data_len - 60: 960]
#create the data sets x_test and y_test
x_test = []
#all the values taht we want our model to predict, the actual test values, the 61 first values
y_test = dataset[training_data_len:, :] 

for i in range(60, len(test_data)):
  x_test.append(test_data[i-60:i, 0])


  #convert the data to a numpy array so we can use it in the LSTM model
x_test = np.array(x_test)

#reshape hte data from 2D to 3D for LSTM model
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

#get hte models predicted price value
# we want the exact same values after we inverse transform the values
predictions = model.predict(x_test)
#we are kind of unscaling... 
#we want predictions to contain the same values as our y_test data set and we are getting the predcitions based on x_test
predictions = scaler.inverse_transform(predictions)


#evaluate our model
#get hte room mean squared error (RMSE) a good way to see how accurate our model is
#lower values of RMSE means a better fit, a value of 0 for rmse means they were exact
rmse = np.sqrt(np.mean(predictions - y_test)**2)
rmse

#plot the data
train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions
#visaulize hte model
plt.figure(figsize=(16,8))
plt.title('Model')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD', fontsize=18)
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
plt.show()

#show the valid and predcited prices
valid 

#get the quote 
stock_quote = yf.download("SPY", start='2019-01-02', end="2022-12-15")
#create a new data frame
new_df = stock_quote.filter(['Close'])
#get hte last 60 day clsoing price days and convert the df to an array
last_60_days = new_df[-60:].values
#scale the data to be between 0 and 1
last_60_days_scaled = scaler.transform(last_60_days)
#create an empty list 
X_test = []
#append the last 60 days into the X_test list
X_test.append(last_60_days_scaled)
#convert to numpy array
X_test = np.array(X_test)
#reshape the data to 3D
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1],1))
#get the predicted scaled price
pred_price = model.predict(X_test)
#undoing the scaling 
pred_price = scaler.inverse_transform(pred_price)
print(pred_price)

#get the actual quote 
stock_quote_actual = yf.download("SPY", start='2022-08-01', end='2024-02-24')
print(stock_quote_actual['Close'])
