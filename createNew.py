import os
import shutil
#remove dataset
path='dataset/'
for i in os.listdir(path):
    os.remove(os.path.join(path,i))
path='UseruploadedPreprocess/'
for i in os.listdir(path):
    os.remove(os.path.join(path,i))
#remove preprocessFile
path='processed_files/'
for i in os.listdir(path):
    shutil.rmtree(os.path.join(path,i))
#remove segmented data
path='split5/'
for i in os.listdir(path):
    shutil.rmtree(os.path.join(path,i))
#remove DWTConverted
path='DWTConverted/'
for i in os.listdir(path):
    os.remove(os.path.join(path,i))

try:
     shutil.rmtree('algorithm\\Random Forest/model')
except:
    pass

try:
     shutil.rmtree('algorithm\\Random Forest/test')
except:
    pass

try:
     shutil.rmtree('algorithm\\MLP/model')
except:
    pass

try:
     shutil.rmtree('algorithm\\MLP/test')
except:
    pass

try:
     shutil.rmtree('algorithm\\LSTM/model')
except:
    pass
try:
     shutil.rmtree('algorithm\\LSTM/test')
except:
    pass






