import configparser
config=configparser.ConfigParser()
config.read('config.ini')
config.set('outputFeatures','test1',str([1,2,3]))
with open('text.ini','w') as f:
    config.write(f)