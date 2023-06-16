import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from tensorflow import keras
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from configparser import ConfigParser
import glob
import test_One_hot
import os
import time
import csv

def test(filenames):
    config=ConfigParser()
    config.read('config.ini')
    try:
        inputs=config['inpuFeatures']['mlp'].split(',')
    except:
        inputs=config['inpuFeatures']['mlp']
    try:
        outputs=config['outputFeatures']['mlp'].split(',')
    except:
        outputs=config['outputFeatures']['mlp']
    model_file="algorithm\MLP\Model\\neuron_model"
    file_path="PreproccessedDataset/MLP/"
    try:
        basic_neuron_model = keras.models.load_model(model_file)
    except:
        return "Model Not Found"
    
    for file in filenames:
        df=pd.read_csv(os.path.join(file_path,file))
        df=test_One_hot.Do_one_hot(df,file)
        df=df.replace('True',1)
        print(df.columns)
        X = df[inputs].values
        print("shape of X", X.shape)
        y = df[outputs].values
        print("Shape of y ", y.shape)
        #----------------------------------------
        prediction=basic_neuron_model.predict(X,verbose=0)
        prediction = [item for sublist in prediction for item in sublist]
        #--------------------------
        
        outputs_as_String = ', '.join(map(str, outputs))
        
        df_pred = df[['time',outputs_as_String]]

        df_pred['vinp']=df["vinp"]
        df_pred['Predicted_'+outputs_as_String] = prediction
        for output in outputs:
            df_pred['diff '+output] = df_pred['Predicted_'+output] - df_pred[output]
        if not os.path.exists('algorithm/MLP/Test/'):
            os.mkdir('algorithm/MLP/Test/')
        if not os.path.exists('algorithm/MLP/Test/PredictedFiles'):
            os.mkdir('algorithm/MLP/Test/PredictedFiles')
        export_file = 'algorithm/MLP/Test/PredictedFiles/'+file
        df_pred.to_csv(export_file,index=False)
        print("written to file..")
        print("---------------------------------")
def Matrix(filenames):
    config=ConfigParser()
    config.read('config.ini')
    try:
        inputs=config['inpuFeatures']['mlp'].split(',')
    except:
        inputs=config['inpuFeatures']['mlp']
    try:
        outputs=config['outputFeatures']['mlp'].split(',')
    except:
        outputs=config['outputFeatures']['mlp']
    import pandas as pd
    import numpy as np
    import glob
    import math
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
    file_path="algorithm/MLP/Test/PredictedFiles/"
    result = pd.DataFrame(columns =['FILENAME','MAE','MSE','R2SCORE','SNR'])
    for file in filenames:
        filename1= file
        file=os.path.join(file_path,file)
        df=pd.read_csv(file)
        data=pd.DataFrame(df)
        length_of_rows=len(data)
        n=length_of_rows
        upper=0
        lower=0
        for ind in data.index:
            out=data[outputs[0]][ind]
            pred_out=data['Predicted_'+outputs[0]][ind]
            upper=(out*out)/n
            lower=(out-pred_out)*(out-pred_out)/n
        snr=10*math.log(upper/lower) 
        predvinn=data['Predicted_'+outputs[0]]
        vinn=data[outputs]
        mae = mean_absolute_error(predvinn,vinn)
        mse = mean_squared_error(predvinn,vinn)
        r2=r2_score(predvinn,vinn) 
        result.loc[len(result)]=[filename1,round(mae,2),round(mse,2),round(r2,2),round(snr,2)]
        print(result)
    if not os.path.exists('algorithm/MLP/Test/Matrix'):
            os.mkdir('algorithm/MLP/Test/Matrix')
    result_file="algorithm/MLP/Test/Matrix/metrics_typical.csv"
    result.to_csv(result_file,index=False)

# Graph Section
def Graph(filenames):
    config=ConfigParser()
    config.read('config.ini')
    try:
        inputs=config['inpuFeatures']['mlp'].split(',')
    except:
        inputs=config['inpuFeatures']['mlp']
    try:
        outputs=config['outputFeatures']['mlp'].split(',')
    except:
        outputs=config['outputFeatures']['mlp']
    import pandas as pd
    import numpy as np
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
    import glob
    import warnings
    warnings.filterwarnings('ignore')
    import matplotlib.pyplot as plt
    file_path='algorithm/MLP/Test/PredictedFiles/'
    for file in filenames:
        filename=os.path.join(file_path,file)
        filename1= file.replace(".csv","")
        print(filename1)
        df_result=pd.read_csv(filename)
        X_time=df_result['time']
        acvinn=df_result[outputs]
        prediction=df_result['Predicted_'+outputs[0]]
        fig=plt.figure(figsize=(16,9),facecolor='w', edgecolor='k')
        plt.plot(X_time,acvinn, color="red", linewidth=3, label = acvinn)
        plt.plot(X_time,prediction, color="blue", linewidth=3, label = prediction)
        mae = mean_absolute_error(acvinn, prediction)
        mse = mean_squared_error(acvinn, prediction)
        r2 = r2_score(acvinn, prediction)
        result=("MAE = "+str(mae)+"MSE ="+str(mse)+"r2 score="+str(r2))
        nm=filename1+".jpg"
        plt.xlabel("Time", fontsize=10)
        plt.ylabel(outputs, fontsize=10)
        plt.text(0.225,0.95,filename1, fontsize=18, transform=plt.gcf().transFigure)
        plt.text(0.35, 0, result, fontsize=15, transform=plt.gcf().transFigure)
        plt.grid(True)
        plt.legend()
        plt.legend(['Actual '+outputs[0],'Predicted_'+outputs[0]], loc ="upper right")
        if not os.path.exists('algorithm/MLP/Test/graph'):
            os.mkdir('algorithm/MLP/Test/graph')
        plt.savefig("algorithm/MLP/Test/graph/"+filename1+".jpg")

        

