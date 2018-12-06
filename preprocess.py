
# coding: utf-8

# In[1]:

from keras.applications.resnet50 import ResNet50
from avg_pooling import *
import imageio
import pickle
import numpy as np
import os, subprocess
from matplotlib import pyplot as plt
#import av
import cv2
# imageio.plugins.ffmpeg.download()
import gc 

# In[91]:


input_dir='action_youtube_naudio/'
model = ResNet50(weights='imagenet', include_top=False, input_shape=(240,320,3))

# In[3]:


labels=dict({0:"basketball",1:"biking",2:"diving",3:"golf_swing",
             4:"horse_riding",5:"soccer_juggling",6:"swing",7:"tennis_swing",
            8:"trampoline_jumping",9:"volleyball_spiking",10:"walking"})
labels= {v: k for k, v in labels.items()}


# In[86]:


def crop(length, crop_num):
    if length<=crop_num:
        return None
    else:
        k=int((length-crop_num)/2)
        return (k, crop_num+k)


def get_frames(input_loc, crop_num):
    cap = cv2.VideoCapture(input_loc)
    ret=True
    result=[]
    shape=(240,320,3)
    miss=False
    print ('Converting video ....')
    while(ret):
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Our operations on the frame come here
        #gray = cv2.cvtColor(frame)
        
        # Display the resulting frame
        #cv2.imshow('frame',gray)
        if ret==False:
            break
        
        if frame.shape!=shape:
            miss=True
            break
        result.append(frame)
    # When everything done, release the capture
    cap.release()
    if miss:
        return None, miss
    p=[]
    print (len(result))
    #for i in range (0,50,5):
    #    p.append(result[i])
    k=crop(len(result),crop_num)
    if k==None:
        return None, True
    result=result[k[0]:k[1]]
    res=[]
    for x in range(0,crop_num, int(crop_num/10)):
        res.append(result[x][:224,:224,:])
    collect=gc.collect()
    return res, miss


## Function that creates a sequences of images for each video in a directory
def create_sequence(file):
    mismatch=False
    sequence=[]
    shape=(240,320,3)
    vid = av.open(file)
    i=0
    for frame in vid.decode(video=0):
        img = np.array(frame.to_image())
        if i<100 :
            if img.shape != shape:
                return None,True
            sequence.append(img)
            i+=1
    return np.array(sequence),mismatch


# In[92]:
from sklearn.model_selection import train_test_split
data=[]
output=[]
count=0
for i in os.listdir(input_dir):
    if i in labels:
        print(i)
        for root, dirs, files in os.walk(input_dir+"/"+i):
            for f in [f for f in files if f.endswith(".avi")]:
                check=True
                print (f)
                cur,check=get_frames(root+"/"+f, 40)
                cur=get_average_pooling(np.array(cur), model)#to get a feature vector for ever image, use get_features() and loop over cur
                if check==False:
                    data.append(cur)
                    output.append(labels[i])
                if len(data)>700:
                    print ('saving count: ...'+str(count))
                    X_train, X_test, y_train, y_test = train_test_split(np.array(data), np.array(output), test_size=0.15, random_state=42)
                    pickle.dump([X_train, y_train], open('train_'+str(count)+'.p', 'wb'))
                    pickle.dump([X_test, y_test], open('test_'+str(count)+'.p', 'wb'))
                    count+=1
                    output=[]
                    data=[]
                    collection=gc.collect()
X_train, X_test, y_train, y_test = train_test_split(np.array(data), np.array(output), test_size=0.15, random_state=42)
pickle.dump([X_train, y_train], open('train_'+str(count)+'.p', 'wb'))
pickle.dump([X_test, y_test], open('test_'+str(count)+'.p', 'wb'))         
