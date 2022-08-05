import os
os.environ['PYTHONHASHSEED']=str(2)

from xml.sax.xmlreader import InputSource
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, LSTM
import numpy as np
import pandas as pd

import random
import math
import datetime

from matplotlib import pyplot

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

import yfinance as yf
from yahoofinancials import YahooFinancials

#Sets all the seeds in order to maintain reproducible results 
def setSeeds():
   os.environ['PYTHONHASHSEED']=str(3)
   tf.random.set_seed(3)
   np.random.seed(3)
   random.seed(3)

#Calculates the Mean Average Percentage Error (MAPE)
def calculateMAPE(actual, predicted): 
    actual, pred = np.array(actual), np.array(predicted)
    return np.mean(np.abs((actual - predicted) / actual)) * 100

#Normalises all necessary data
def normaliseData(inputs_train, inputs_test, outputs_train, outputs_test, forecastInput, inputs_scaler, outputs_scaler, outputDim):
    inputs_train_n = inputs_scaler.fit_transform(inputs_train) #fit_transform sets the settings on the scaler
    inputs_test_n = inputs_scaler.transform(inputs_test) 
    if outputDim == 1: 
        outputs_train_n = outputs_scaler.fit_transform(outputs_train.reshape(-1, 1)) 
        outputs_test_n = outputs_scaler.transform(outputs_test.reshape(-1, 1))   
    else:
        outputs_train_n = outputs_scaler.fit_transform(outputs_train) #Doesn't need to reshape if outputDim > 1
        outputs_test_n = outputs_scaler.transform(outputs_test)  
    forecastInput_n = inputs_scaler.transform(forecastInput.reshape(1, -1))
    return inputs_train_n, inputs_test_n, outputs_train_n, outputs_test_n, forecastInput_n

#Unnormalises data specific data
def unnormaliseData(data_n, scaler):
    data = scaler.inverse_transform(data_n)
    return data


def splitData(stockData, inputDim, outputDim, attributes): #attributes = col index
    inputs = []
    outputs = []
    forecastInput = []

    #These loops are necessary to arrange the data in an acceptable format
    for attribute in attributes:
        for i in range(inputDim):
            forecastInput.append(stockData[-(inputDim+i), attribute])

    inputsTemp = []
    outputsTemp = []

    #These loops are necessary to arrange the data in an acceptable format
    for i in range(len(stockData)-inputDim-outputDim):
        for attribute in attributes:
            for j in range(inputDim):
                inputsTemp.append(stockData[i+j, attribute]) 
        inputs.append(inputsTemp)
        inputsTemp = []
        for j in range(outputDim):
            outputsTemp.append(stockData[i+inputDim+j, 3]) #3 is the close price column
        outputs.append(outputsTemp)
        outputsTemp = []

    inputsArray = np.array(inputs)
    outputsArray = np.array(outputs)
    forecastInputArray = np.array(forecastInput)

    #This method splits the data into test and train sets
    inputsArray_train, inputsArray_test, outputsArray_train, outputsArray_test = train_test_split(inputsArray, outputsArray, test_size=0.2, random_state=12) 

    return inputsArray_train, inputsArray_test, outputsArray_train, outputsArray_test, forecastInputArray

def compileModel(inputDim, outputDim, attributes):
    model = Sequential()
    model.add(Dense((math.ceil((inputDim+outputDim)/2)), input_dim=inputDim*len(attributes), activation='relu')) #Can use a formula (math.ceil((2/3)*inputDim)+outputDim) or (math.ceil((inputDim+outputDim)/2))
    model.add(Dense(outputDim)) #Can use activation functions for these... activation='sigmoid'
    model.compile(loss="mean_squared_error", optimizer="adam", metrics=['mse', 'mae','mape', 'cosine_proximity'])
    model.summary()
    return model

def fitModel(model, inputs_train, outputs_train, epochs):
    history = model.fit(inputs_train, outputs_train, epochs=epochs, batch_size=64)

    return model, history

def testModel(model, inputs_test, outputs_test, outputs_scaler):
    outputs_pred = model.predict(inputs_test)
    outputs_pred = unnormaliseData(outputs_pred, outputs_scaler)
    outputs_test = unnormaliseData(outputs_test, outputs_scaler)

    rmse = math.sqrt(mean_squared_error(outputs_test, outputs_pred))
    mape = calculateMAPE(outputs_test, outputs_pred)

    return outputs_pred, rmse, mape

def forecast(stockData, inputDim, outputDim, attributes, epochs):
    #Set seeds for reproducable results
    setSeeds()
    stockDataArray = stockData.to_numpy()

    inputs_scaler = MinMaxScaler(feature_range=(0, 1))
    outputs_scaler = MinMaxScaler(feature_range=(0, 1))

    inputs_train, inputs_test, outputs_train, outputs_test, forecastInput = splitData(stockDataArray, inputDim, outputDim, attributes)
    inputs_train_n, inputs_test_n, outputs_train_n, outputs_test_n, forecastInput_n = normaliseData(inputs_train, inputs_test, outputs_train, 
        outputs_test, forecastInput, inputs_scaler, outputs_scaler, outputDim)
    
    model, history = fitModel(compileModel(inputDim, outputDim, attributes), inputs_train_n, outputs_train_n, epochs)
    outputs_pred, rmse, mape = testModel(model, inputs_test_n, outputs_test_n, outputs_scaler) 

    forecastOutput_n = model.predict(forecastInput_n)
    forecastOutput = unnormaliseData(forecastOutput_n, outputs_scaler).reshape(-1, 1)
    
    forecastDates = []

    for i in range(outputDim):
        forecastDates.append([stockData.index[-1] + datetime.timedelta(days=i+1)])

    forecastDates = np.array(forecastDates)

    forecastResults = pd.DataFrame({"Date": forecastDates[:,0], "Close": forecastOutput[:,0]})
    forecastResults.set_index('Date', inplace=True)

    return forecastResults, outputs_test, outputs_pred, rmse, mape, history

""" #TEST ONLY

stockData = yf.download("AAPL", start="2021-04-20")
stockDataArray = stockData.to_numpy()

outputDim = 10

forecastOutput, outputs_test, outputs_pred, rmse, mape, history = forecast(stockData, 20, outputDim, [0, 1, 2, 3, 4, 5], 50)

print(forecastOutput)

pyplot.plot(history.history['mse'])
pyplot.plot(history.history['mae'])
pyplot.show()

pyplot.plot(outputs_test, color='blue')
pyplot.plot(outputs_pred, color='orange')
pyplot.show()

print('Test RMSE: %.3f' % rmse)
print('Test MAPE: %.3f' % mape)

forecastOutput = np.transpose(forecastOutput) #This puts the data in a format ready to be plotted

days = []

for i in range(outputDim):
    days.append([datetime.date.today() + datetime.timedelta(days=i+1)])

pyplot.plot(stockData.index, stockDataArray[:, 3], color='blue')
pyplot.plot(days, forecastOutput, color='orange')
pyplot.show()

#TEST ONLY """