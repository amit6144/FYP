#lstm for net13 prediction
import pandas as pd
from numpy import array
from keras.models import Sequential
from keras.layers.core import Activation, Dropout, Dense
from keras.layers import LSTM
from keras.models import Model
from sklearn.model_selection import train_test_split
import numpy as np
from configparser import ConfigParser
import pickle
import os
config=ConfigParser()
config.read('config.ini')
def preprocess(df):
    array=[0]
    for i in df['vinn']:
        while(len(array)<len(df['vinn'])):
            array.append(i) 
    df['prev_vinn']=array
    return df
def train(inputs,outputs,neu1,neu2,neu3,n_epoch):
    file = "train.csv"
    df = pd.read_csv(file)
    df=df.fillna(0)
    try:
        df = pd.get_dummies(df, columns=['process'])
    except:
        pass
    df=preprocess(df)
    try:
        input=inputs.split(',') 
    except:
        input=inputs[0]
    try:
        output=outputs.split(',')
    except:
        output=outputs
    arrays=[]
    print(input)
    for i in input:
        array=np.array(df[i])
        arrays.append(array)
    arrays.append(df['prev_vinn'])
    X = np.column_stack(arrays)
       


    
    #y1 = df['vinn'].values
    #y2 = df['net56'].values
    outputArrays=[]
    for i in output:
        array=np.array(df[i])
        outputArrays.append(array)
    y3=df['net13'].values
    #y = np.column_stack(y3)

    #print (len(X))
    #print(len(y))


    n_samples = len(X)

    X = np.array(X).reshape(n_samples,1,len(arrays))
    y = np.array(y3)

    #print(X[2])
    #print(y[2])
    
    neu1=int(neu1)
    neu2=int(neu2)
    neu3=int(neu3)
    n_epoch=int(n_epoch)
    model = Sequential()
    model.add(LSTM(neu1,activation='relu',return_sequences=True,input_shape = (1,len(arrays))))
    model.add(LSTM(neu2, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam',loss='mse')
    print(model.summary())

    model.fit(X,y,epochs=n_epoch,validation_split=0.1,batch_size=10)

    model.save("algorithm/LSTM/Model/rnn_net13_model")
    print("model saved")
    # Second Model
    arrays.append(model.predict(X))
    X=np.column_stack(arrays)

    #y1 = df['vinn'].values
    #y2 = df['net56'].values
    y3 =np.column_stack(outputArrays)
    #y = np.column_stack(y3)

    #print (len(X))
    #print(len(y))


    n_samples = len(X)

    X = np.array(X).reshape(n_samples,1,len(arrays))
    y = np.array(y3).reshape(len(y3),1,len(outputArrays))

    #print(X[2])
    #print(y[2])

    model = Sequential()
    model.add(LSTM(neu1,activation='relu',return_sequences=True,input_shape = (1,len(arrays))))
    model.add(LSTM(neu2, activation='relu'))
    model.add(Dense(len(outputArrays)))
    model.compile(optimizer='adam',loss='mse')
    print(model.summary())

    model.fit(X,y,epochs=n_epoch,validation_split=0.10,batch_size=10)

    model.save("algorithm/LSTM/Model/rnn_vinn_model")
    print("model saved")

    config.set("inpuFeatures", "LSTM", inputs)
    config.set("outputFeatures", "LSTM", outputs)
    with open("config.ini", 'w') as example:
     config.write(example)



