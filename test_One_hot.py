import pandas as pd
import os
import numpy as np
import glob
import re

def Test_one_HOT():
    file_path="test/"
    if not os.path.exists(file_path):
            os.mkdir(file_path)
    for folder in os.listdir(file_path):
        t1=glob.glob(os.path.join(file_path+folder)+"/*.csv")
        for file in t1:
            df_result=pd.DataFrame()
            filename=file.replace(os.path.join(file_path+folder)+"\\","")#read file from path into induvidual files
            print(filename)
            filename1= filename.replace(".csv","")
            df=pd.read_csv(file)
            #del df['Unnamed: 0']
            #print(filename1)
            #filter=re.findall('[A-Z]{3}[A-Z0-9]{10}[0-9]{4}', filename1)
            filter=re.findall("[a-zA-Z]+",filename1)
            print(filter)
            df_test= pd.get_dummies(df, columns = ['process'])
            if filter[2] == 'typical':
                df_test['process_fastnfast']=0
                df_test['process_fastnslow']=0
                df_test['process_slownfast']=0
                df_test['process_slownslow']=0
            elif filter[2] == 'fastnfastp':
                df_test['process_typical']=0
                df_test['process_fastnslow']=0
                df_test['process_slownfast']=0
                df_test['process_slownslow']=0
            elif filter[2] == 'fastnslowp':
                df_test['process_typical']=0
                df_test['process_fastnfast']=0
                df_test['process_slownfast']=0
                df_test['process_slownslow']=0
            elif filter[2] == 'slownfastp':
                df_test['process_typical']=0
                df_test['process_fastnfast']=0
                df_test['process_fastnslow']=0
                df_test['process_slownslow']=0
            elif filter[2] == 'slownslowp':
                df_test['process_typical']=0
                df_test['process_fastnfast']=0
                df_test['process_fastnslow']=0
                df_test['process_slownfast']=0
            else:
                break
            if not os.path.exists('test_one_hot'):
                os.mkdir('test_one_hot')
            df_test.to_csv("test_one_hot/"+filename1+".csv")
    path='split5/'
    folder_list=['rise1','rise2','rise3','functional','fall']
    df=pd.DataFrame()
    for folder in folder_list:
        for file in os.listdir(os.path.join(path,folder)):
            fullpath=os.path.join(path,folder,file)
            df=pd.concat([df,pd.get_dummies(pd.read_csv(fullpath), columns = ['process'])])
            df=df.fillna(0)
    df.to_csv('train.csv')
def Do_one_hot(df,filename1):
            #del df['Unnamed: 0']
            #print(filename1)
            #filter=re.findall('[A-Z]{3}[A-Z0-9]{10}[0-9]{4}', filename1)
            filter=re.findall("[a-zA-Z]+",filename1)
            print(filter)
            df_test= pd.get_dummies(df, columns = ['process'])
            if filter[2] == 'typical':
                df_test['process_fastnfast']=0
                df_test['process_fastnslow']=0
                df_test['process_slownfast']=0
                df_test['process_slownslow']=0
            elif filter[2] == 'fastnfastp':
                df_test['process_typical']=0
                df_test['process_fastnslow']=0
                df_test['process_slownfast']=0
                df_test['process_slownslow']=0
            elif filter[2] == 'fastnslowp':
                df_test['process_typical']=0
                df_test['process_fastnfast']=0
                df_test['process_slownfast']=0
                df_test['process_slownslow']=0
            elif filter[2] == 'slownfastp':
                df_test['process_typical']=0
                df_test['process_fastnfast']=0
                df_test['process_fastnslow']=0
                df_test['process_slownslow']=0
            elif filter[2] == 'slownslowp':
                df_test['process_typical']=0
                df_test['process_fastnfast']=0
                df_test['process_fastnslow']=0
                df_test['process_slownfast']=0
            else:
                print("Error")
            return df_test


