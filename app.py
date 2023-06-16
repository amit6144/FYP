from flask import Flask,flash,render_template,request,url_for,redirect
import os
from configparser import ConfigParser
import sys
import shutil
import pandas as pd
import dwt_convertion
import importlib


config=ConfigParser()
config.read('config.ini')
features=config['DatasetVerifiaction']['features']
algos=config['Algorithms']['list'].split(',')
features=features.replace('[','')
features=features.replace(']','')
features=features.replace('\'','')
features=features.split(', ')
app=Flask(__name__)
app.secret_key='amit-secret-key'


# DataSpace Massege Part____________________________________________________________________________
massege=''
def message():
    nDataFiles=len(os.listdir('Dataset/'))
    nPreprocessFiles=0
    for folder in os.listdir('processed_files/'):
        for file in os.listdir(os.path.join('processed_files',folder)):
            nPreprocessFiles+=1
    nSplittedFiles=0
    for folder in os.listdir('split5/'):
        for file in os.listdir(os.path.join('split5',folder)):
            nSplittedFiles+=1
    print(nDataFiles)
    print(nPreprocessFiles)
    print(nSplittedFiles)
    if(nDataFiles*10>nPreprocessFiles):
        massege='Some files left for PreProcessing, please click on Preprocessing!' 
    elif(nPreprocessFiles*5>nSplittedFiles):
        massege='Some files left for Data Segmentation, please click on Segmentation!' 



#render the DataSpoce template_________________________________________________________________
@app.route('/')
def home():
    return render_template('index.html',title='Index')
#render the DataSpoce template_________________________________________________________________
@app.route('/dataSpace')
def dataSpace():
    return render_template('dataSpace.html',title='Data Space',massege=massege)
@app.route('/trainingSpace')
def trainSpace():
    return render_template('train.html',title='Train Space',massege=massege,features=features,algos=algos)


@app.route('/uploadData',methods=['POST'])
def upload():
        response_message = 'Hello from before_request!'
        request.environ['message'] = response_message
        if request.method=='POST':
            global filenames
            filenames=[]
            if not (os.path.exists('Dataset')):
                os.mkdir('Dataset')
            files=request.files.getlist('file')
            for file in files:
                file.save(file.filename)
                if(config['DatasetVerifiaction']['opampsine']==str([str(row) for row in pd.read_csv(file.filename).columns.values])):
                    print(pd.read_csv(file.filename).columns.values)
                    try:
                         shutil.move(file.filename,os.path.join('Dataset',file.filename))
                    except:pass
                    filenames.append(file.filename)
                    print('Writing File: ',file.filename)
                    success='Upload Success'
                else:
                    if(error==""):
                        error="These Files are not Matched"
                    
                    error=error+" "+file.filename
                try:
                    os.remove(file.filename)
                except:
                    pass
        return redirect(url_for('dataSpace'))


def trainTestSplit():
    parent_folder = "dwtprocessed_files"

    # Path to the Train_data folder
    train_folder = "Train_data"

    # Path to the test_data folder
    test_folder = "test_data"

    # Get the list of files in the parent folder
    file_list = os.listdir(parent_folder)

    # Calculate the split index based on the 90:10 ratio
    split_index = int(0.9 * len(file_list))

    # Move files to the Train_data folder
    for file_name in file_list[:split_index]:
        source_path = os.path.join(parent_folder, file_name)
        destination_path = os.path.join(train_folder, file_name)
        shutil.move(source_path, destination_path)

    # Move files to the test_data folder
    for file_name in file_list[split_index:]:
        source_path = os.path.join(parent_folder, file_name)
        destination_path = os.path.join(test_folder, file_name)
        shutil.move(source_path, destination_path)

    # Delete the parent folder
    os.rmdir(parent_folder)

# preprocessing______________________________________________________________________________________
def writefile(tmp,name,df_r,wf):
    nam = name + "_"+tmp
    export = 'processed_files/'+wf+nam + ".csv"
    nmp=nam + ".csv"
    nmp=nmp[1:]
    if not os.path.exists('processed_files'):
        os.mkdir('processed_files')
    if not os.path.exists('processed_files/'+wf):
        os.mkdir('processed_files/'+wf)
    #print (export)
    nam = 'temp='+tmp
    df_n = df_r.filter(regex=nam)
    #df_n['time'] = df_r['time']
    #print (df_n.columns)
    col_names = df_n.columns
    for i in col_names:
        dict1 = {'vdd (t': 'vdd','pd (te':'pd', 'xpd (t': 'xpd','vinp (':'vinp','Ivdd/M':'Ivdd', 'Ignd/P':'Ignd','Ivinp/':'Ivinp'}
        dict2 = {'Ipd/MI':'Ipd', 'Ixpd/M':'Ixpd', 'Ivinn/':'Ivinn','net69 ':'net69', 'net56 ':'net56', 'net51 ':'net51'}
        dict3 = {'net31 ':'net31', 'net27 ':'net27', 'net21 ':'net21', 'net13 ':'net13', 'net12 ':'net12', 'vinn (':'vinn'}
        df_n.rename(columns = {i:i[1:7]},inplace = True)
        df_n.rename(columns= dict1, inplace=True)
        df_n.rename(columns = dict2, inplace=True)
        df_n.rename(columns = dict3, inplace=True)
    df_n['time'] = df_r['time']
    df_n['temperature'] = tmp
    if(wf == "ty/"):
        df_n['process'] = 'typical'
    if(wf =="ff/"):
        df_n['process'] = 'fastnfast'
    if(wf == "fs/"):
        df_n['process'] = 'fastnslow'
    if(wf == "sf/"):
        df_n['process'] = 'slownfast'
    if(wf == "ss/"):
        df_n['process'] = 'slownslow'

    df_n['waveform'] = 'sine'
    #print(df_n.columns)
    df_n.to_csv(export,index=False)
    print ("written file ", export)
    

def takeonebyone(file,writefolder,filename):
      df_orig = pd.read_csv(file)
      col_names = df_orig.columns
      #print (len(col_names))
      # check for last X and rename to time
      #print (col_names[0][-1])
      for i in col_names:
          if i[-1] == 'X':
              df_orig.rename(columns={i:'time'}, inplace = True)
      #print ("After renaming X to time ")
      #print ("_____________________________")
      #print (df_orig.columns)

      # Remove duplicate time
      #print ("After removing duplicate time")
      #print ("--------------------------------")
      df_r = df_orig.loc[:,~df_orig.columns.duplicated()]
      #print(df_r.columns)

      #column names, truncate temperature and get temperature used
      col_names = df_r.columns
      get_temp = []
      for i in col_names:
          if "=" and ")" in i:
              position1 = i.index("=")
              position2 = i.index(")")
              temp = i[position1+1:position2]
              get_temp += [temp]

      temperature = list(set(get_temp))
      #print (temperature)
      array=[]
      array=file.split(".")
      if len(array[0]) == 2:
         name = file[3:]
      else :
         name=  file[2:]   
      name = name[:-4]
      name=name[5:]
      for tmp in temperature:
        #   print('************************************')
        #   print('processed_files/'+writefolder+name[1:]+'_'+temp+'.csv')
          if(os.path.exists('processed_files/'+writefolder+name[1:]+'_'+temp+'.csv')):
            #   print('hiiii')
              continue
          writefile(tmp,name,df_r,writefolder)

@app.route('/preprocessing')
def preprocessing():
        print('Start Preprocessing')
        files=os.listdir('Dataset/')
        for file in files: 
                print(file)  
                if(file.find('typical')!=-1):
                    k='ty/'
                if(file.find('slownfastp')!=-1):
                    k='sf/'
                if(file.find('slownslowp')!=-1):
                    k='ss/'
                if(file.find('fastnslowp')!=-1):
                    k='fs/'
                if(file.find('fastnfastp')!=-1):
                    k='ff/'
                fullpath=os.path.join('Dataset',file)
                takeonebyone(fullpath,k,file)
                print('*****************Preprocessed File***************************')
        return 'Success'
# Segmentation Part____________________________________________________________________________________________

def split(file,name,export):
    try:
        df_file = pd.read_csv(file)
        #print(df_file.head())
        #------------------------------------
        #Transient1 separation
        #df_file['vinn']= df_file['vinn'].round(decimals=6)
        vinpmax = df_file['xpd'].max()
        cutoff = 1.65
        l = df_file.vinp[df_file.vinp>cutoff].index.tolist()
        df_transient1 = df_file[0:l[1]]

        #csvname = export+"transient1/"+name+".csv"
        #df_transient1.to_csv(csvname)
        #--------------peak1 within transient1
        r1 = df_transient1.xpd[df_transient1.xpd==0].index.tolist()
        df_rise1 = df_file[0:r1[-1]+1]
        if not (os.path.exists(export+"rise1/")):
            os.mkdir(export+"rise1/")
        csvname = export+"rise1/"+name
        df_rise1.to_csv(csvname)
        #---------------------------------------
        #-------------peak2 within Transient1
        r2 = df_transient1.vinp[df_transient1.vinp>0].index.tolist()
        df_rise2 = df_file[r1[-1]:r2[0]+1]
        if not (os.path.exists(export+"rise2/")):
            os.mkdir(export+"rise2/")
        csvname = export+"rise2/"+name
        df_rise2.to_csv(csvname)
        #--------------------------------------
        #-----------rise3
        df_rise3 = df_file[r2[0]:l[1]]
        if not (os.path.exists(export+"rise3/")):
            os.mkdir(export+"rise3/")
        csvname = export+"rise3/"+name
        df_rise3.to_csv(csvname)
        #-------------------------
        #Functional region split
        df_rest = df_file[l[1]:]
        xpd0list = df_rest.pd[df_rest.pd != 0].index.tolist()
        m = xpd0list[0]-1
        df_functional = df_file[l[1]:m]
        if not (os.path.exists(export+"functional/")):
            os.mkdir(export+"functional/")
        csvname = export+"functional/"+name
        df_functional.to_csv(csvname)
        #---------------------------------------
        #Transient2
        df_t2 = df_file[m+1:]
        if not (os.path.exists(export+"fall/")):
            os.mkdir(export+"fall/")
        csvname = export+"fall/"+name
        df_t2.to_csv(csvname)
    except:
        print('File is not Valid') 
# Segmentation_______________________________________________________________________________________________
@app.route('/segmentaion',methods=['GET'])
def segmentaion():
    allpath = ["processed_files/ty/","processed_files/ff/","processed_files/fs/","processed_files/ss/","processed_files/sf/"]
    lpath = len(allpath[0])-5
    list_files = []
    count =0
    export = "split5/"
    process = ["ty/","ff/","fs/","ss/","sf/"]
    if not os.path.exists(export):
        os.mkdir(export)
    for path,proc in zip(allpath,process):
        if not (os.path.exists(path)):
            continue
        for file in os.listdir(path):
                if(os.path.exists(os.path.join('split5','fall',file)) and os.path.exists(os.path.join('split5','rise1',file)) and 
                   os.path.exists(os.path.join('split5','rise2',file)) and os.path.exists(os.path.join('split5','rise3',file))
                    and os.path.exists(os.path.join('split5','functional',file)) ):continue
                print(file)
                file=os.path.join(path,file)
                count = count+1
                print ("File ", count)
                print ("---------------")
                name = file[lpath:]
                name=name[5:]
                print ("name ", name)
                list_files += [name]
                split(file,name,export)
    return 'success'

# DWT Conversion_________________________________________________________________________________________________________
@app.route('/DWT',methods=['POST'])
def DWT():
    if(request.method=='POST'):
        mode=request.form['mode']
        level=request.form['level']
        dwt_convertion.convert_dwt_and_save(mode,level)
    return redirect(url_for('dataSpace'))

# Training Section_______________________________________________________________________________________________________


@app.route('/uploadPreprocessData',methods=['POST'])
def uploadPreprocessData():
    global massege
    if request.method=='POST':
        files=request.files.getlist('file')
        for file in files:
                file.save(file.filename)
                if(config['DatasetVerifiaction']['preprocessdata']==str([str(row) for row in pd.read_csv(file.filename).columns.values])):
                    print(pd.read_csv(file.filename).columns.values)
                    try:
                         shutil.move(file.filename,os.path.join('UseruploadedPreprocess',file.filename))
                    except:
                        print('error')
                    print('Writing File: ',file.filename)
                    massege='Upload Success'
                else:
                 print('error')
                try:
                    os.remove(file.filename)
                except:
                    pass

    
    massege='Upload Success'

    return redirect(url_for('trainSpace'))
@app.route('/train',methods=['POST'])
def train():
    if request.method=='POST':
        inputs=request.form.getlist('inputs')
        outputs=request.form['outputs']
        choice=request.form['choice']
        datachoice=request.form['datachoice']
        neu1=request.form['neu1']
        neu2=request.form['neu2']
        neu3=request.form['neu3']
        algo=request.form['algo']
        epoch=request.form['epoch']
        sys.path.append(os.path.abspath('algorithm/'+algo))
        module = importlib.import_module(algo)
        res=module.train(choice,datachoice,inputs,outputs,neu1,neu2,neu3,epoch)
        global massege
        if res!=None:
            massege=res


    return redirect(url_for('trainSpace'))


        
        






app.run(debug=false)
