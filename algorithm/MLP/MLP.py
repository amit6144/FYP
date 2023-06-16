import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import Adam
import numpy as np
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn import metrics
from configparser import ConfigParser
from sklearn.model_selection import train_test_split
import pickle
import os
config=ConfigParser()
config.read('config.ini')

def train(input,output,neu1,neu2,neu3,epoch):
    print(neu1,neu2,neu3)
    file = "train.csv"
    print('Model Building Start')
    df = pd.read_csv(file)
    df=df.replace('True',1)
    print(input)
    #print (df.head())
    try:
       inputs = input.split(',')
    except:
       inputs=input
    try:
        outputs=output.split(',')
    except:
        outputs=output[0]
    config.set("inpuFeatures", "MLP", input)
    config.set("outputFeatures", "MLP", output)
    
    
    #inputs = ["vinp","xpd","temperature",'vinn']
    #df['process']=labelencoder.fit_transform(df['process'])
    #X = df_train.drop('vinn',axis=1)
    #y = df_train["vinn"]


    #df_train = pd.read_csv("train/training_dataset/train_dataset.csv");


    #del df_train['Unnamed: 0']

    #del df_train['Unnamed: 0.1']
    #print (df_train.columns)

    #inputs = ['temperature', 'vdd', 'vinp','vinn']
    #labelencoder = LabelEncoder()

    #df_train['process']=labelencoder.fit_transform(df_train['process'])



    X = df[inputs].values
    y = df[outputs].values
    X = X.astype(float)
    y = y.astype(float)

    print(X.shape)
    print(y.shape)

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

    #X_train = s_scaler.fit_transform(X_train.astype(np.float))
    #X_test = s_scaler.transform(X_test.astype(np.float))
    ### Neural network
    model = Sequential()
    #model.add(Dense(32,activation='relu'))
    #model.add(Dense(8,activation='relu'))
    #model.add(Dense(1))
    model.add(Dense(neu1,activation='relu'))
    #model.add(Dense(16,activation='relu'))
    model.add(Dense(neu2,activation='relu'))
    model.add(Dense(neu3))

    model.compile(optimizer='Adam',loss='mse')

    model.fit(x=X_train,y=y_train,
            validation_data=(X_test,y_test),
            batch_size=128,epochs=int(epoch))
    model.summary()

    model.save('algorithm/MLP/Model/neuron_model')
    print ("model saved")
    with open("config.ini", 'w') as example:
     config.write(example)
def train(choice,datachoice,inputs,outputs,neu1,neu2,neu3,epoch):
    if(choice!='Based on Region'):
        dfs=[]
        if(datachoice=='Pre-Uploaded Data'):
            path='processed_files/'
            for folder in os.listdir(path):
                for file in os.listdir(os.path.join(path+folder)): 
                    newp=os.path.join(path,folder,file)
                    dfs.append(dd.read_csv(newp))
        else:
           path='UseruploadedPreprocess/'
           for file in os.listdir(path):
              newp=os.path.join(path,file)
              dfs.append(dd.read_csv(newp))   
        try:
           data=dd.concat(dfs)
        except:
           return "Data not found"
        data=data.compute()
        data=test_One_hot.Do_one_hot(data,file)
        data=data.replace('True',1)
        df=data
        print(inputs)
        config.set("inpuFeatures", "RF", str(inputs))
        config.set("outputFeatures", "RF", str(outputs))
        X=df[inputs]
        y=df[outputs]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

        model = Sequential()
        #model.add(Dense(32,activation='relu'))
        #model.add(Dense(8,activation='relu'))
        #model.add(Dense(1))
        model.add(Dense(neu1,activation='relu'))
        #model.add(Dense(16,activation='relu'))
        model.add(Dense(neu2,activation='relu'))
        model.add(Dense(neu3))

        model.compile(optimizer='Adam',loss='mse')

        model.fit(x=X_train,y=y_train,
                validation_data=(X_test,y_test),
                batch_size=128,epochs=int(epoch))
        model.summary()

        if not os.path.exists('algorithm\\Random Forest\\model\\'):
          os.mkdir('algorithm\\Random Forest\\model\\')
        pickle.dump(rf, open('algorithm\\Random Forest\\model\\'+model_name,'wb'))
        print(model_name)
        with open("config.ini", 'w') as example:
         config.write(example)
    else:
       path='split5/'
       for folder in os.listdir(path):
        dfs=[]
        for file in os.listdir(os.path.join(path,folder)):
            dfs.append(dd.read_csv(os.path.join(path,folder,file)))
        try:
           data=dd.concat(dfs)
        except:
           return "Data not found"
        data=data.compute()
        data=test_One_hot.Do_one_hot(data,file)
        data=data.replace('True',1)
        df=data
        print(inputs)
        config.set("inpuFeatures", "RF", str(inputs))
        config.set("outputFeatures", "RF", outputs)

        X=df[inputs]
        y=df[outputs]
        rf = RandomForestRegressor(n_estimators=20,max_depth=10)
        # fit model
        rf.fit(X,y)
        model_name="sine_750mv_functional_rf"+".pkl"
        if not os.path.exists('algorithm\\Random Forest\\model\\'):
          os.mkdir('algorithm\\Random Forest\\model\\')
        if not os.path.exists('algorithm\\Random Forest\\model\\'+folder):
           os.mkdir('algorithm\\Random Forest\\model\\'+folder)
        pickle.dump(rf, open('algorithm\\Random Forest\\model\\'+folder+'\\'+model_name,'wb'))
        print(model_name)
        with open("config.ini", 'w') as example:
         config.write(example)

             
       