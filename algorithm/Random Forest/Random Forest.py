from numpy import asarray
import os
import pandas as pd
import numpy as np
import glob
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from configparser import ConfigParser
import test_One_hot
import dask.dataframe as dd
import pickle
import os
config=ConfigParser()
config.read('config.ini')
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
        rf = RandomForestRegressor(n_estimators=20,max_depth=10)
        # fit model
        rf.fit(X,y)
        model_name="sine_750mv_functional_rf"+".pkl"
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

             
       

        