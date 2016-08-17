import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn
from datetime import timedelta
import random

from keras.datasets import imdb
from keras.models import Sequential, Model
from keras.layers import Dense, Input, Dropout
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from sklearn.cross_validation import train_test_split

'''
filepath to workign directory:
cd Data\ Files/2014-/Research/HeH/HeH\ Partners/Fitbit\ Analysis/Fitbit\ Data\ Analysis/160801\ Galvanize\ Data/
'''


def combine_CAD(row):
    #Function to use in df.apply in load_clean functions to combine CAD-related columns
    if row['heart_attack']==1 or row['blockages_in_your_coronary']==1:
        return int(1)
    if row['heart_attack']==2:
        return int(2)
    if row['heart_attack']==3:
        return int(3)
    if row['blockages_in_your_coronary']==2:
        return int(2)
    if row['blockages_in_your_coronary']==3:
        return int(3)



def load_clean_custom(disease_var, days_drop_if_less=180, observation_window=59):
    '''
    INPUT: Requires two files 'meas_fitbit_tracker.txt' and 'surv_medical_conditions.txt' to be in the working dir. Will read and clean
    variables from both txt files, preparing them into shape necessary for the RNN

    OUTPUT: returns test train split vars for the disease_var outcome to use in modeling, after cleaning and subsetting the variables
    Also returns union_med dataframe from which clean_custom function below can then create another disease specific df without having to re-load data (time consuming)
    '''

    #load fitbit data
    fit = pd.read_table('meas_fitbit_tracker.txt', sep='|', parse_dates=[1])
    fit['date']=pd.to_datetime(fit['date'], format='%m/%d/%Y') #convert fitbit date to datetime
    fit = fit.drop(['distance', 'calories', 'floors', 'elevation'], axis=1)
    fit['steps_zero'] = fit.steps
    fit['steps_zero'] = fit['steps_zero'][fit['steps'].isnull()==True]=0
    fit['steps_missing'] = fit.steps
    fit['steps_missing'] = fit['steps_missing'][fit['steps']==0]=float('NaN')
    #now create a days used column for the entire fit df
    fit['days_used']=fit['date'].groupby(fit['user_id']).transform('count')

    #I'm going to pause here to see if I can pivot to building a recurrent neural net with the Fitbit data.

    ##here i will pause to try to run a recurrent neural network on daily data.
    #so what i need to do is to create an X df with the last 60d of people's step data
    #ranging from 0 to x (convert NaN to 0)
    #and a Y df with blockages in coroary vs. not
    # so start by creating datasets according to these parameters

    # First data-clean: drop users with days_used < days_drop_if_less
    fit_clean = fit
    fit_clean.drop(fit_clean[fit_clean.days_used<days_drop_if_less].index, inplace=True)
    #Now to create a dataset with only the last 60d of data for people
    #start by calculating the max date for everyone and dropping data older than observation_window than each person's last date
    fit_clean['max_date']=fit_clean['date'].groupby(fit_clean['user_id']).transform('max')
    fit_clean.drop(fit_clean[fit_clean.date<(fit_clean.max_date-timedelta(days=observation_window))].index, inplace=True)
    fit_clean.steps.fillna(0, inplace=True) #replace NaN steps with 0 for neural network

    #read med_cond into df and create a mini df with only desired columns
    med_cond = pd.read_table('surv_medical_conditions.txt', sep='|')
    mini_med_cond = med_cond[['user_id', 'hbp', 'high_chol', 'diabetes', 'blockages_in_your_coronary', 'heart_attack', 'pvd', 'clots', 'chf', 'stroke', 'enlarged', 'afib', 'arrhythmia', 'murmur', 'valve', 'congenital' ,'pulm_hyper', 'aorta', 'sleep_apnea', 'copd', 'asthma', 'arrhythmia1', 'arrhythmia2']]

    # now create a column that combines "heart attack" or "blockages in your coronary"
    mini_med_cond['CAD']=mini_med_cond.apply(lambda row: combine_CAD(row), axis=1)
    #drop duplicate entries for mini_med_cond by user_id, keeping last entry  (hacky way around the fact that there are duplicate entries per user in union_med)
    mini_med_cond.drop_duplicates(subset='user_id', keep='last', inplace=True)

    #now merge fit and mini_med_cond df's
    union_med = pd.merge(fit_clean, mini_med_cond, how='left', on='user_id', copy=True)
    union_med.sort_values(by=['user_id', 'date'], inplace=True)  #sorting data on user_id and date
    X_disease = union_med[union_med[disease_var]==1]
    X_disease_control = union_med[union_med[disease_var]==2]
    control_user_ids = X_disease_control.user_id.unique()

    case_num = len(X_disease.user_id.unique())
    X_disease_control_user_list = np.random.choice(control_user_ids, size=case_num, replace=False)
    X_disease_control.sort_values(by=['user_id', 'date'], inplace=True) #sorting data on user_id and date
    X_disease_control = X_disease_control[X_disease_control['user_id'].isin(X_disease_control_user_list)]
    X_disease_control.drop_duplicates(subset=['user_id','date'], inplace=True) #drops duplicate fitbit entries per user per date
    #So now merging X_disease and X_disease_control into the X variable
    X_disease_full = pd.concat([X_disease, X_disease_control])
    X_disease_full.sort_values(by=['user_id', 'date'], inplace=True) #need to sort full X df by user_id to align with the Y df (below)
    #X_disease_full.sort(columns=['user_id', 'date'], inplace=True) #sorting data on user_id and date

    #resetting index of X_disease_full and removing unneeded variable
    X_disease_full.reset_index(inplace=True)
    X_disease_full = X_disease_full.drop(['index'], axis=1)
    # now need to restructure data to be one row per user, and 60 columns wide
    sixty=pd.DataFrame([np.tile(np.arange(0, observation_window+1), case_num*2)]).T ##creating a df with repeating 0-59 to denote each person in df.pivot
    sixty.columns=['ind'] #naming new sequence variable so we can reference it in the pivot method
    X_disease_model = X_disease_full.join(sixty)
    X_disease_model2 = X_disease_model[['user_id', 'steps', 'ind']] #selecting only those columns needed
    #pivoting df to create 1 row per user and 60 features wide (1 feature per daily step count)
    X_disease_model_arr = X_disease_model2.pivot(index='user_id', values='steps', columns='ind')
    #resetting index and removing user_id variable
    X_disease_model_arr.reset_index(inplace=True)
    X_disease_model_arr.drop('user_id', axis=1, inplace=True)
    #"X_disease_model_arr" IS THE PREDICTOR VAR TO USE IN THE RNN MODEL

    #create response variable Y
    Y_labels = X_disease_full.groupby('user_id').mean()[disease_var]
    #So it looks like there are 13 NaNs and 2 "3" values in the Y_labels, so I will turn both of these to "2"
    Y_labels.replace(to_replace=3, value=2, inplace=True)
    Y_labels.fillna(value=2, inplace=True)
    Y_labels.replace(to_replace=2, value=0, inplace=True) #changing no disease_var to 0 from 2
    Y = Y_labels.values #FINAL TARGET VAR TO USE IN THE RNN

    x_train, x_test, y_train, y_test = train_test_split(X_disease_model_arr, Y, test_size=0.2)

    #now, last thing, we have to format the X_train/test vars as keras wants them for RNN
    x_train = x_train.values[:,:,None]
    x_test = x_test.values[:,:,None]

    return x_train, x_test, y_train, y_test, union_med



def clean_custom(union_med, disease_var, days_drop_if_less=180, observation_window=59):
    '''
    INPUT: Pre-loaded union_med df as input, and will then create target X, Y test train split variables for RNN

    OUTPUT: returns test train split vars for the disease_var outcome to use in modeling, after cleaning and subsetting the variables
    '''

    X_disease = union_med[union_med[disease_var]==1]
    X_disease_control = union_med[union_med[disease_var]==2]
    control_user_ids = X_disease_control.user_id.unique()

    case_num = len(X_disease.user_id.unique())
    X_disease_control_user_list = np.random.choice(control_user_ids, size=case_num, replace=False)
    X_disease_control.sort(columns=['user_id', 'date'], inplace=True) #sorting data on user_id and date
    X_disease_control = X_disease_control[X_disease_control['user_id'].isin(X_disease_control_user_list)]
    X_disease_control.drop_duplicates(subset=['user_id','date'], inplace=True) #drops duplicate fitbit entries per user per date
    #So now merging X_disease and X_disease_control into the X variable
    X_disease_full = pd.concat([X_disease, X_disease_control])
    X_disease_full.sort(columns='user_id', inplace=True) #need to sort full X df by user_id to align with the Y df (below)
    #X_disease_full.sort(columns=['user_id', 'date'], inplace=True) #sorting data on user_id and date

    #resetting index of X_disease_full and removing unneeded variable
    X_disease_full.reset_index(inplace=True)
    X_disease_full = X_disease_full.drop(['index'], axis=1)
    # now need to restructure data to be one row per user, and 60 columns wide
    sixty=pd.DataFrame([np.tile(np.arange(0, observation_window+1), case_num*2)]).T ##creating a df with repeating 0-59 to denote each person in df.pivot
    sixty.columns=['ind'] #naming new sequence variable so we can reference it in the pivot method
    X_disease_model = X_disease_full.join(sixty)
    X_disease_model2 = X_disease_model[['user_id', 'steps', 'ind']] #selecting only those columns needed
    #pivoting df to create 1 row per user and 60 features wide (1 feature per daily step count)
    X_disease_model_arr = X_disease_model2.pivot(index='user_id', values='steps', columns='ind')
    #resetting index and removing user_id variable
    X_disease_model_arr.reset_index(inplace=True)
    X_disease_model_arr.drop('user_id', axis=1, inplace=True)
    #"X_disease_model_arr" IS THE PREDICTOR VAR TO USE IN THE RNN MODEL

    #create response variable Y
    Y_labels = X_disease_full.groupby('user_id').mean()[disease_var]
    #So it looks like there are 13 NaNs and 2 "3" values in the Y_labels, so I will turn both of these to "2"
    Y_labels.replace(to_replace=3, value=2, inplace=True)
    Y_labels.fillna(value=2, inplace=True)
    Y_labels.replace(to_replace=2, value=0, inplace=True) #changing no disease_var to 0 from 2
    Y = Y_labels.values #FINAL TARGET VAR TO USE IN THE RNN

    x_train, x_test, y_train, y_test = train_test_split(X_disease_model_arr, Y, test_size=0.2)

    #now, last thing, we have to format the X_train/test vars as keras wants them for RNN
    x_train = x_train.values[:,:,None]
    x_test = x_test.values[:,:,None]

    return x_train, x_test, y_train, y_test


def fitbit_onelayer_rnn():
    #creating a LSTN, RNN using the Functional API (rather than Sequential)
    r_input = Input(shape=(60,1)) #input layer takes in input only (partially for clarity) tuple of shape is (number of time_series observations, features per observations)
    lstm = LSTM(64)(r_input) #first lstm layer, 64 is the number of nodes int he LSTM layer
    pred = Dense(1, activation='sigmoid')(lstm) #dense output layer, 1 since its a binary output, thus sigmoid activation
    rnn_model = Model(input=r_input, output=pred)
    rnn_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print "One layer RNN model"
    print rnn_model.summary()
    return rnn_model


def fitbit_multilayer_rnn():
    #scaffold for a multi-layer RNN
    r_input = Input(shape=(60,1))
    lstm = LSTM(256, return_sequences=True)(r_input) #return_sequences=True outputs returns an output from this layer as shape (None, 60, 64) which allows for input to the next layer
    #another layer, need return sequences
    lstm= LSTM(256, return_sequences=True)(lstm) #second layer (would add option return_sequences if intending to add another layer)
    lstm=Dropout(0.2)(lstm)
    lstm= LSTM(256)(lstm)
    pred = Dense(1, activation='sigmoid')(lstm)
    two_model = Model(input=r_input, output=pred)
    two_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print "Multi-layer RNN model"
    print two_model.summary()
    return two_model

'''
    #Sequential code for a single layer LSTM RNN model
    hidden_neurons = 64
    model =  Sequential()
    model.add(LSTM(hidden_neurons, input_shape=(60,1)))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    #model code for LSTM, RNN using the Functional API, with multiple layers
    r_input = Input(shape=(60,1))
    lstm = LSTM(64, return_sequences=True)(r_input) #return_sequences=True outputs returns an output from this layer as shape (None, 60, 64) which allows for input to the next layer
    #another layer, need return sequences
    lstm= LSTM(64)(lstm) #second layer (would add option return_sequences if intending to add another layer)
    pred = Dense(1, activation='sigmoid')(lstm)
    two_model = Model(input=r_input, output=pred)
    two_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
'''


'''General old code
    for user in fit.user_id.unique():
        for i in xrange(len(fit[fit.user_id==user])):
            tot = 0
            tot += fit.steps[i]
            fit['tot_mean_steps'] = tot / len(fit[fit.user_id==user])

    for user in fit.user_id.unique():
        fit['step_tot'] = [step for fit.steps[:,i] for i in xrange(len(fit.user_id==user))]


'''


if __name__ == '__main__':
    #x_train, x_test, y_train, y_test = load_clean_CAD() #creating x_train test split for CAD
    #x_train, x_test, y_train, y_test = load_clean_HF() #creating x_train test split for HF
    x_train, x_test, y_train, y_test, union_med = load_clean_custom('hbp') #creating x_train test split for 'disease'

    rnn_model_1l = fitbit_onelayer_rnn()
    rnn_model_1l.fit(x_train, y_train, nb_epoch=100, batch_size=10) #fit model to x_train and Y_train
    # print "One-layer-RNN model test peformance: "
    # rnn_model_1l.evaluate(x_test, y_test)
    rnn_model_multi = fitbit_multilayer_rnn()
    rnn_model_multi.fit(x_train, y_train, nb_epoch=25, batch_size=10)
    # print "Multi-RNN model test peformance: "
    # rnn_model_multi.evaluate(x_test, y_test)

    X_model_full = X_disease_model_arr.values[:,:,None] #format full data into RNN format, to apply validation split as below.
    rnn_model_1l.fit(X_model_full, Y, nb_epoch=50, batch_size=10, validation_split=0.15) #fit model to x_train and Y_train
