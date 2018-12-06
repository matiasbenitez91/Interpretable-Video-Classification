from avg_pooling import *
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from keras.layers import Dropout, Activation
import gc





model = Sequential()
model.add(Dense(200, input_dim=2048, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(200, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(11, activation="softmax"))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

 early_stop_= EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=1, mode='auto')
 history= self.model.fit(x, y, batch_size=32, nb_epoch=200, validation_split=0.15, verbose=0, callbacks=[early_stop_])




