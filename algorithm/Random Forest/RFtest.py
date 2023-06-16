from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from numpy import asarray
import test_One_hot
import pandas as pd
import numpy as np
import os
import glob
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from configparser import ConfigParser
import pickle

def Test(filenames,choice,region):
    print(choice,region)
    print('*************************************************************8')
    config=ConfigParser()
    config.read('config.ini')
    try:
        inputs=config['inpuFeatures']['rf'].split(',')
    except:
        inputs=config['inpuFeatures']['rf']
    try:
        outputs=config['outputFeatures']['rf'].split(',')
    except:
        outputs=config['outputFeatures']['rf']
    config=ConfigParser()
    config.read('config.ini')
    inputs=config['inpuFeatures']['rf'].split(',')
    print(inputs)
    if(choice=='Without Region'):
        model_file="algorithm\\RandomForest\model\\sine_750mv_functional_rf.pkl"
    elif (region=='Select region'):
        print('Please Select any region')
        return 'Please Select any region' 
    else: 
        model_file="algorithm\\RandomForest\model\\"+region+"\\sine_750mv_functional_rf.pkl"
        
    file_path="PreproccessedDataset/Random Forest/"
    try:
        regression_technique=pickle.load(open(model_file,'rb'))
    except:
        return "Model Not Found"
    print(file_path)
    for filename in filenames:
        filename1= filename
        filename=os.path.join(file_path,filename)
        df_result=pd.DataFrame()
        print(filename1)
        df=pd.read_csv(filename)
        df=test_One_hot.Do_one_hot(df,filename1)
        print(df.columns.values)
        #df["time"] = pd.to_numeric(df["time"])
        #print(df['time'])
        test_time=df['time']
        #inputs=['vinp','vdd','xpd','temperature']
     
        X_test=df[inputs].values
        y_test=df[outputs[0]]
        print(X_test)
        print(y_test)
        model=model_file
        predictions=regression_technique.predict(X_test)
        df_result["Time"] = df["time"]
        df_result["actual "+outputs[0]] = y_test
        df_result["predicted "+outputs[0]] = predictions
        mae = mean_absolute_error(y_test, predictions)
        rms = sqrt(mean_squared_error(y_test, predictions))
        r2 = r2_score(y_test, predictions)
        print("Mean Square error: "+str(rms)+"\nMean Absolute error: "+str(mae)+"\nR2 score: "+str(r2))
        prediction_result_file="algorithm/RandomForest/Test/PredictedFiles/"+filename1
        if not os.path.exists('algorithm/RandomForest/Test/'):
            os.mkdir('algorithm/RandomForest/Test/')
        if not os.path.exists('algorithm/RandomForest/Test/PredictedFiles'):
            os.mkdir('algorithm/RandomForest/Test/PredictedFiles')
        if(choice!='Without Region'):
            if(region!='Select region'):
                try:
                    prediction_result_file="algorithm/RandomForest/Test/PredictedFiles/"+region+'/'+filename1
                    os.mkdir('algorithm/RandomForest/Test/PredictedFiles/'+region)
                except:
                    pass

        df_result.to_csv(prediction_result_file,index=False)
# Matrix Section
def Matrix(filenames,choice,region):
    config=ConfigParser()
    config.read('config.ini')
    try:
        inputs=config['inpuFeatures']['rf'].split(',')
    except:
        inputs=config['inpuFeatures']['rf']
    try:
        outputs=config['outputFeatures']['rf'].split(',')
    except:
        outputs=config['outputFeatures']['rf']
    import pandas as pd
    import numpy as np
    import glob
    import math
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
    if(choice=='Without Region'):
        file_path="algorithm\\RandomForest\\Test\\PredictedFiles\\"
    else:
        file_path="algorithm\\RandomForest\\Test\\PredictedFiles\\"+region
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
            out=data["actual "+outputs[0]][ind]
            pred_out=data["predicted "+outputs[0]][ind]
            upper=(out*out)/n
            lower=(out-pred_out)*(out-pred_out)/n
        snr=10*math.log(upper/lower) 
        predvinn=data["predicted "+outputs[0]]
        vinn=data["actual "+outputs[0]]
        mae = mean_absolute_error(predvinn,vinn)
        mse = mean_squared_error(predvinn,vinn)
        r2=r2_score(predvinn,vinn) 
        result.loc[len(result)]=[filename1,round(mae,2),round(mse,2),round(r2,2),round(snr,2)]
        print(result)
    if not os.path.exists('algorithm/RandomForest/Test/matrix'):
            os.mkdir('algorithm/RandomForest/Test/matrix')
            result_file="algorithm/RandomForest/Test/matrix/metrics_typical.csv"
    # if (choice!='Without Region'):
    #     try:
    #         result_file="algorithm/RandomForest/Test/matrix/"+region+"/metrics_typical.csv"
    #         os.mkdir('algorithm/RandomForest/Test/matrix/'+region)
    #     except:
    #         pass
    result.to_csv(result_file,index=False)
# Graph Section
def Graph(filenames,choice,region):
    config=ConfigParser()
    config.read('config.ini')
    try:
        inputs=config['inpuFeatures']['rf'].split(',')
    except:
        inputs=config['inpuFeatures']['rf']
    try:
        outputs=config['outputFeatures']['rf'].split(',')
    except:
        outputs=config['outputFeatures']['rf']
    import pandas as pd
    import numpy as np
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
    import glob
    import warnings
    warnings.filterwarnings('ignore')
    import matplotlib.pyplot as plt
    if(choice=='Without Region'):
        file_path="algorithm\\RandomForest\\Test\\PredictedFiles\\"
    else:
        file_path="algorithm\\RandomForest\\Test\\PredictedFiles\\"+region
    for file in filenames:
        filename=os.path.join(file_path,file)
        filename1= file.replace(".csv","")
        print(filename1)
        df_result=pd.read_csv(filename)
        X_time=df_result['Time']
        acvinn=df_result["actual "+outputs[0]]
        prediction=df_result["predicted "+outputs[0]]
        fig=plt.figure(figsize=(16,9),facecolor='w', edgecolor='k')
        plt.plot(X_time,acvinn, color="red", linewidth=3, label = acvinn)
        plt.plot(X_time,prediction, color="blue", linewidth=3, label = prediction)
        mae = mean_absolute_error(acvinn, prediction)
        mse = mean_squared_error(acvinn, prediction)
        r2 = r2_score(acvinn, prediction)
        result=("MAE = "+str(mae)+"MSE ="+str(mse)+"r2 score="+str(r2))
        nm=filename1+".jpg"
        plt.xlabel("Time", fontsize=10)
        plt.ylabel("vinn", fontsize=10)
        plt.text(0.225,0.95,filename1, fontsize=18, transform=plt.gcf().transFigure)
        plt.text(0.35, 0, result, fontsize=15, transform=plt.gcf().transFigure)
        plt.grid(True)
        plt.legend()
        plt.legend(["actual "+outputs[0],"predicted "+outputs[0]], loc ="upper right")
        if not os.path.exists('algorithm/RandomForest/Test/graph'):
            os.mkdir('algorithm/RandomForest/Test/graph')
            result_file="algorithm/RandomForest/Test/graph/"+filename+".jpg"
    if (choice!='Without Region'):
        try:
            result_file="algorithm/RandomForest/Test/graph/"+region+'/'+filename+".jpg"
            os.mkdir('algorithm/RandomForest/Test/graph/'+region)
        except:
            pass
        plt.savefig("algorithm/RandomForest/Test/graph/"+filename1+".jpg")

        

