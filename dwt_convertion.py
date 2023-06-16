import pandas as pd
import numpy as np
import pickle
from pywt import wavedec, coeffs_to_array
import os
from tqdm import tqdm

#fetch process, voltage and temperature from file name
def fetch_pvt_from_path(file_path):
    
    pvt = file_path.split("_")
    return pvt[2],float(pvt[1][:-1]),float(pvt[3][:-4])


#one hot encode for process
def one_hot_encode(p):
    
    p_list = ["fastnfastp","fastnslowp","slownfastp","slownslowp","typical"]
    encode = [0,0,0,0,0]
    
    for i in range(len(p_list)):
        if(p_list[i] == p):
            encode[i] = 1
            return encode
        
    raise Exception("encode problem") 


def convert_dwt_and_save1(Mode,Level):
    mode = Mode
    level =int(Level)
    input_folder_path='test_data/'
    if not (os.path.exists('DWTConverted')):
       os.mkdir('DWTConverted')
    input_path='DWTConverted/testInput.csv'
    output_path='DWTConverted/testOutput.csv'

    # #find all the files in given path
    # with open('DWTConverted.pickle', 'rb') as f:
    #         converted=set(pickle.load(f))
    
    # with open('DWTConversion.pickle', 'rb') as f:
    #         totalfiles=set(pickle.load(f))
    # print(totalfiles)
    # toConvert=list(totalfiles.difference(converted))
    # converted=list(converted)
    # files=toConvert
    files=os.listdir('test_data')
    print('file ',files)

    input = []
    output = []

    #find dwt coefficent for all files
    for file in tqdm(files):

      file_path = f"{input_folder_path}{file}"

      #read csv 
      df = pd.read_csv(file_path)

      #pvt
      p,v,t = fetch_pvt_from_path(file_path) 
      
      pvt = [t]
      pvt.extend(one_hot_encode(p))    

      vinn = df["vinn"]
      xpd = df["xpd"]
      vdd = df["vdd"]
      vinp = df["vinp"]

      wave_size = len(vinn)

      #dwt convertion
      coeffs_vinn = wavedec(vinn.tolist(), 'db4', mode=mode,level=level)
      coeffs_xpd = wavedec(xpd.tolist(), 'db4', mode=mode,level=level)
      coeffs_vdd = wavedec(vdd.tolist(), 'db4', mode=mode,level=level)
      coeffs_vinp = wavedec(vinp.tolist(), 'db4', mode=mode,level=level)

      #flatten the coeffs
      flatten_output = coeffs_vinn[0]
      flatten_input = []

      for j in range(len(coeffs_xpd[0])):
        flatten_input.append(float(coeffs_xpd[0][j]))
        flatten_input.append(float(coeffs_vdd[0][j]))
        flatten_input.append(float(coeffs_vinp[0][j]))
        flatten_input.extend(pvt)
            
                
      output.append(flatten_output)
      input.append(flatten_input)
    #   with open('DWTConverted.pickle', 'wb') as f:
    #         converted.append(file)
    #         pickle.dump(set(converted),f)


    #store coefficients in csv file
    np_input = np.array(input)
    df_input = pd.DataFrame(np_input)
    df_input.to_csv(input_path)

    np_output = np.array(output)
    df_output = pd.DataFrame(np_output)
    df_output.to_csv(output_path)

    print("dwt convertion completed...")

    return 
def convert_dwt_and_save(Mode,Level):
    mode = Mode
    level =int(Level)
    input_folder_path='processed_files/'
    if not (os.path.exists('DWTConverted')):
       os.mkdir('DWTConverted')
    input_path='DWTConverted/trainInput.csv'
    output_path='DWTConverted/trainOutput.csv'

    # #find all the files in given path
    # with open('DWTConverted.pickle', 'rb') as f:
    #         converted=set(pickle.load(f))
    
    # with open('DWTConversion.pickle', 'rb') as f:
    #         totalfiles=set(pickle.load(f))
    # print(totalfiles)
    # toConvert=list(totalfiles.difference(converted))
    # converted=list(converted)
    # files=toConvert
    folderlist=os.listdir('processed_files')


    input = []
    output = []

    #find dwt coefficent for all files
    for folder in folderlist:
      for file in os.listdir(os.path.join('processed_files',folder)):
         

        file_path = os.path.join('processed_files',folder,file)

        #read csv 
        df = pd.read_csv(file_path)

        #pvt
        print(file_path)
        p,v,t = fetch_pvt_from_path(file) 
        
        pvt = [t]
        pvt.extend(one_hot_encode(p))    

        vinn = df["vinn"]
        xpd = df["xpd"]
        vdd = df["vdd"]
        vinp = df["vinp"]

        wave_size = len(vinn)

        #dwt convertion
        coeffs_vinn = wavedec(vinn.tolist(), 'db4', mode=mode,level=level)
        coeffs_xpd = wavedec(xpd.tolist(), 'db4', mode=mode,level=level)
        coeffs_vdd = wavedec(vdd.tolist(), 'db4', mode=mode,level=level)
        coeffs_vinp = wavedec(vinp.tolist(), 'db4', mode=mode,level=level)

        #flatten the coeffs
        flatten_output = coeffs_vinn[0]
        flatten_input = []

        for j in range(len(coeffs_xpd[0])):
          flatten_input.append(float(coeffs_xpd[0][j]))
          flatten_input.append(float(coeffs_vdd[0][j]))
          flatten_input.append(float(coeffs_vinp[0][j]))
          flatten_input.extend(pvt)
              
                  
        output.append(flatten_output)
        input.append(flatten_input)
      #   with open('DWTConverted.pickle', 'wb') as f:
      #         converted.append(file)
      #         pickle.dump(set(converted),f)


    #store coefficients in csv file
    np_input = np.array(input)
    df_input = pd.DataFrame(np_input)
    df_input.to_csv(input_path)

    np_output = np.array(output)
    df_output = pd.DataFrame(np_output)
    df_output.to_csv(output_path)

    print("dwt convertion completed...")


    return 
