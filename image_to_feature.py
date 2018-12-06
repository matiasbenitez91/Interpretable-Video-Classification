from avg_pooling import *
import pickle
import numpy as np
[x, y]=pickle.load(open('data/train_0.p','rb'))
model_extract=ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))
x=get_average_pooling(x, model_extract)
collector=gc.collect()
print('collected ', collector)

[x_train, y_train]=pickle.load(open('data/train_1.p','rb'))
x_train=get_average_pooling(x_train, model_extract)
x=np.append(x, x_train, axis=0)
del x_train
y=np.append(y, y_train, axis=0)
del y_train
collector=gc.collect()
print('collected ', collector)

[x_train, y_train]=pickle.load(open('data/train_2.p','rb'))
x_train=get_average_pooling(x_train, model_extract)
x=np.append(x, x_train, axis=0)
del x_train
y=np.append(y, y_train, axis=0)
del y_train
collector=gc.collect()
print('collected ', collector)


print (' shape x ', x.shape)
print (' shape y ', y.shape)

pickle.dump([x,y], open('train.p', 'wb'), protocol=2)