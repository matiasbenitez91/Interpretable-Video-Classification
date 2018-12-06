import pickle
import numpy as np
import pandas as pd

data_train=pickle.load(open('data/train_0.p','rb'))
data_test=pickle.load(open('data/test_0.p','rb'))

print ('train_shape  ', data_train[0].shape)
print ('outpput train shape  ', data_train[1].shape)
print ('test shape  ', data_test[0].shape)
print ('output test shape  ', data_test[1].shape)

#print ('class train counter ')
#print (pd.Series(data_train[1]).value_counts())
#print ('class test counter ')
#print (pd.Series(data_test[1]).value_counts())