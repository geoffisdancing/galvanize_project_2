import pandas as pd
import numpy as np
from keras.datasets import imdb
from keras.models import Sequential, Model
from keras.layers import Dense, Input, Dropout
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.callbacks import ModelCheckpoint
from sklearn.cross_validation import train_test_split

import fitbit_clean as fc

#set var to disease you wish to study from the data
disease_to_study = 'hbp'

# create test_train data and also return the overall data to extract other diseases from
x_train, x_test, y_train, y_test, union_med, X_disease_model_arr, Y = fc.load_clean_custom(disease_to_study) #creating x_train test split for 'disease'

# Build one layer RNN
r_input = Input(shape=(60,1)) #input layer takes in input only (partially for clarity) tuple of shape is (number of time_series observations, features per observations)
lstm = LSTM(64)(r_input) #first lstm layer, 64 is the number of nodes int he LSTM layer
pred = Dense(1, activation='sigmoid')(lstm) #dense output layer, 1 since its a binary output, thus sigmoid activation
rnn_model_1 = Model(input=r_input, output=pred)
rnn_model_1.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# set up to checkpoint model if its performance improves
filepath_one='one_weights.best.hdf5'
checkpoint_one = ModelCheckpoint(filepath_one, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list_one = [checkpoint_one]

print "One layer RNN model"
print rnn_model_1.summary()
rnn_model_1.fit(x_train, y_train, nb_epoch=50, batch_size=25, callbacks=callbacks_list_one) #fit model to x_train and Y_train


# Build Multi-layer RNN
r_input = Input(shape=(60,1))
lstm = LSTM(256, return_sequences=True)(r_input) #return_sequences=True outputs returns an output from this layer as shape (None, 60, 64) which allows for input to the next layer
#another layer, need return sequences
lstm= LSTM(256, return_sequences=True)(lstm) #second layer (would add option return_sequences if intending to add another layer)
lstm=Dropout(0.2)(lstm)
lstm= LSTM(256)(lstm)
pred = Dense(1, activation='sigmoid')(lstm)
rnn_model_multi = Model(input=r_input, output=pred)
rnn_model_multi.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# set up to checkpoint model if its performance improves
filepath_multi='multi_weights.best.hdf5'
checkpoint_multi = ModelCheckpoint(filepath_multi, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list_multi = [checkpoint_multi]

print "Multi-layer RNN model"
print rnn_model_multi.summary()
rnn_model_multi.fit(x_train, y_train, nb_epoch=25, batch_size=25, callbacks=callbacks_list_multi)


X_model_full = X_disease_model_arr.values[:,:,None] #format full data into RNN format, to apply validation split as below.
rnn_model_1.fit(X_model_full, Y, nb_epoch=10, batch_size=25, validation_split=0.15) #fit model to x_train and Y_train

'''
model_dir = G.MOD + model_name + '/'
os.mkdir(model_dir)
temp_path = model_dir + 'temp_model.hdf5'

#update the model if stuff gets better
checkpointer = ModelCheckpoint(filepath=temp_path, verbose=1, save_best_only=True)

# this actually fits the model
output = model.fit_generator(
            train_generator,
            samples_per_epoch=train_generator.N,
            nb_epoch=nb_epoch,
            validation_data=val_generator,
            nb_val_samples=val_generator.N,
                callbacks=[checkpointer])
 '''
