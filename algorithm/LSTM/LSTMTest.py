import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from tensorflow import keras
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import os
import time
import test_One_hot
from configparser  import ConfigParser
import numpy as np
import pandas as pd
from numpy import array
from tensorflow import keras
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import csv
config=ConfigParser()

def Preprocess(df):
    data=[0]
    for vinn in df['vinn']:
        if(len(data)<len(df['vinn'])):
            data.append(vinn)
    df['prev_vinn']=data
    return df

def test(filenmaes):
    config.read('config.ini')
    if not os.path.exists('algorithm/LSTM/Test/'):
            os.mkdir('algorithm/LSTM/Test/')
    if not os.path.exists('algorithm/LSTM/Test/PredictedFiles'):
            os.mkdir('algorithm/LSTM/Test/PredictedFiles')
    predictpath='algorithm/LSTM/Test/PredictedFiles/'
    try:
         net13model = keras.models.load_model('algorithm/LSTM/Model/rnn_net13_model')
         vinnmodel = keras.models.load_model('algorithm/LSTM/Model/rnn_vinn_model')
    except:return "Model Not Found"
    filepath='PreproccessedDataset/LSTM/'
    try:
        inputs=config['inpuFeatures']['lstm'].split(',')
    except:
        inputs=config['inpuFeatures']['lstm']
    try:
        outputs=config['outputFeatures']['lstm'].split(',')
    except:
        outputs=config['outputFeatures']['lstm']
    for filename in filenmaes:    
        df=pd.read_csv(os.path.join(filepath,filename))
        df=test_One_hot.Do_one_hot(df,filename)
        df=df.replace(True,1)
        print(df.columns)
        df=Preprocess(df)
        location = 0
        # vdd,pd,xpd,vinp
        temperature = 45
        vdd = 3.6
        inputFeatures=[]
        inputfield=[]
        for i in inputs:
            inputfield.append(i)
            inputFeatures.append(df[i].iloc[location]) 
        inputFeatures.append(df["prev_vinn"].iloc[location])
        inputfield.append('prev_vinn') 
       
        test_input = array(inputFeatures)
        print ("This is the starting input" , test_input)
        test_input = test_input.reshape(1,1,len(inputFeatures))
        test_output = net13model.predict(test_input,verbose=0)
        pred_net13 = test_output[0][0]
        inputFeatures.append(pred_net13)
        inputfield.append('pred_net13')
        test_input_13 = array(inputFeatures)
        test_input_13 = test_input_13.reshape(1,1,len(inputFeatures))
        test_output_vinn = vinnmodel.predict(test_input_13,verbose=0)
        pred_vinn = test_output_vinn[0][0]
        inputFeatures.append(df[outputs[0]].iloc[location])  
        inputfield.append(outputs[0]) 
        inputfield.append('pred_vinn')
        print("predicted","net13",outputs, pred_net13,pred_vinn)

        #net56 = df["net56"].iloc[location]
        net13 = df["net13"].iloc[location]
        vinn = df[outputs].iloc[location]
        print(inputfield)
        actual_output = [net13,vinn]
        print("actual_output", actual_output)

        num_rows = len(df)
        export = filename
        inputfield.append('pred_net13')
        inputfield.append('pred_vinn')
        field_name = inputfield
        with open (predictpath+export,'a') as f:
            writer = csv.writer(f)
            writer.writerow(inputfield)
        #col = []
        for i in range(location,num_rows):
            print(i)
            inputFeatures=[]
            print(inputfield)
            for j in inputs:
                inputFeatures.append(df[j].iloc[i])
            inputFeatures.append(df["prev_vinn"].iloc[i])    
        
            test_input = array(inputFeatures)
            print ("This is the starting input" , test_input)
            test_input = test_input.reshape(1,1,len(inputFeatures))
            test_output = net13model.predict(test_input,verbose=0)
            pred_net13 = test_output[0][0]
            inputFeatures.append(pred_net13)
            test_input_13 = array(inputFeatures)
            test_input_13 = test_input_13.reshape(1,1,len(inputFeatures))
            test_output_vinn = vinnmodel.predict(test_input_13,verbose=0)
            pred_vinn = test_output_vinn[0][0]
            inputFeatures.append(df[outputs[0]].iloc[i])
            inputFeatures.append(pred_vinn)
            print("predicted","net13",outputs, pred_net13,pred_vinn)
            #net56 = df["net56"].iloc[location]
            net13 = df["net13"].iloc[location]
            vinn = df[outputs].iloc[location]
            actual_output = [net13,vinn]
            print("actual_output", actual_output)
            with open (predictpath+export,'a') as f:
                writer = csv.writer(f)
                writer.writerow(inputFeatures)
                


def Matrix(filenames):
    config=ConfigParser()
    config.read('config.ini')
    try:
        inputs=config['inpuFeatures']['lstm'].split(',')
    except:
        inputs=config['inpuFeatures']['lstm']
    try:
        outputs=config['outputFeatures']['lstm'].split(',')
    except:
        outputs=config['outputFeatures']['lstm']
    import pandas as pd
    import numpy as np
    import glob
    import math
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
    file_path="algorithm/LSTM/Test/PredictedFiles/"
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
            out=float(data[outputs[0]][ind])
            pred_out=float(data['pred_'+outputs[0]][ind])
            upper=(out*out)/n
            lower=(out-pred_out)*(out-pred_out)/n
        snr=10*math.log(upper/lower) 
        predvinn=data['pred_'+outputs[0]]
        vinn=data[outputs]
        mae = mean_absolute_error(predvinn,vinn)
        mse = mean_squared_error(predvinn,vinn)
        r2=r2_score(predvinn,vinn) 
        result.loc[len(result)]=[filename1,round(mae,2),round(mse,2),round(r2,2),round(snr,2)]
        print(result)
        if not os.path.exists('algorithm/LSTM/Test/Matrix'):
            os.mkdir('algorithm/LSTM/Test/Matrix')
    result_file="algorithm/LSTM/Test/Matrix/metrics_typical.csv"
    result.to_csv(result_file,index=False)

# Graph Section
def Graph(filenames):
    config=ConfigParser()
    config.read('config.ini')
    try:
        inputs=config['inpuFeatures']['lstm'].split(',')
    except:
        inputs=config['inpuFeatures']['lstm']
    try:
        outputs=config['outputFeatures']['lstm'].split(',')
    except:
        outputs=config['outputFeatures']['lstm']
    import pandas as pd
    import numpy as np
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
    import glob
    import warnings
    warnings.filterwarnings('ignore')
    import matplotlib.pyplot as plt
    file_path='algorithm/LSTM/Test/PredictedFiles/'
    for file in filenames:
        filename=os.path.join(file_path,file)
        filename1= file.replace(".csv","")
        print(filename1)
        df_result=pd.read_csv(filename)
        X_time=df_result['time']
        acvinn=df_result[outputs]
        prediction=df_result['pred_'+outputs[0]]
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
        if not os.path.exists('algorithm/LSTM/Test/Graph'):
            os.mkdir('algorithm/LSTM/Test/Graph')
        plt.savefig("algorithm/LSTM/Test/Graph/"+filename1+".jpg")

        






