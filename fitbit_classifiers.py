import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn
from datetime import timedelta
import random
from sklearn.cross_validation import train_test_split

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import precision_score, recall_score, roc_auc_score, accuracy_score


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



def load_clean(disease_var, days_drop_if_less=180, observation_window=59):
    fitbit = pd.read_csv('fitbit_features.csv')
    fitbit['date']=pd.to_datetime(fitbit['date'], format='%m/%d/%y') #convert fitbit date to datetime
    fitbit['days_used']=fitbit['date'].groupby(fitbit['user_id']).transform('count')

    fit_clean = fitbit[['user_id', 'weardays', 'datediff', 'propor', 'meansteps', 'validsteps', 'misssteps', 'steptertile', 'maxmiss', 'meanmiss', 'miss21dcount', 'miss7dcount', 'pre21propor', 'steps_worn_bla', 'fail21d', 'meanstep30', 'prop30', 'SD30', 'meanstep60', 'prop60', 'SD60', 'endmean30', 'endprop30', 'endSD30', 'endmean60', 'endprop60', 'SDtotal', 'decq', 'incq']]
    fit_clean.drop_duplicates(subset='user_id', keep='first', inplace=True)
    #hacky approach, but will drop NaNs for now
    #would need to go back through STATA code and see why these are missing for each var
    fit_clean_drop = fit_clean.dropna()

    #read med_cond into df and create a mini df with only desired columns
    med_cond = pd.read_table('surv_medical_conditions.txt', sep='|')
    mini_med_cond = med_cond[['user_id', 'hbp', 'high_chol', 'diabetes', 'blockages_in_your_coronary', 'heart_attack', 'pvd', 'clots', 'chf', 'stroke', 'enlarged', 'afib', 'arrhythmia', 'murmur', 'valve', 'congenital' ,'pulm_hyper', 'aorta', 'sleep_apnea', 'copd', 'asthma', 'arrhythmia1', 'arrhythmia2']]

    # now create a column that combines "heart attack" or "blockages in your coronary"
    mini_med_cond['CAD']=mini_med_cond.apply(lambda row: combine_CAD(row), axis=1)
    #drop duplicate entries for mini_med_cond by user_id, keeping last entry  (hacky way around the fact that there are duplicate entries per user in union_med)
    mini_med_cond.drop_duplicates(subset='user_id', keep='last', inplace=True)
    mini_med_disease = mini_med_cond[['user_id', disease_var]] #select just the disease_var column from mini_med_cond
    #merging cleaned fitbit features dataset (one row per user, with summary statistics over entire usage period) and medical conditions df.
    union_med = pd.merge(fit_clean_drop, mini_med_disease, how='left', on='user_id', copy=True)
    union_med.sort_values(by=['user_id'], inplace=True)  #sorting data on user_id and

    X_disease = union_med[union_med[disease_var]==1]
    X_disease_control = union_med[union_med[disease_var]==2]
    X_disease_full = pd.concat([X_disease, X_disease_control])
    X_disease_full.sort_values(by=['user_id'], inplace=True) #need to sort full X df by user_id to align with the Y df (below)

    #resetting index of X_disease_full and removing unneeded variable
    X_disease_full.reset_index(inplace=True)
    X_disease_full = X_disease_full.drop(['index'], axis=1)
    Y_labels = X_disease_full[[disease_var]]
    Y_labels.replace(to_replace=3, value=2, inplace=True)
    Y_labels.fillna(value=2, inplace=True)
    Y_labels.replace(to_replace=2, value=0, inplace=True) #changing no disease_var to 0 from 2
    X = X_disease_full.drop(disease_var, axis=1)

    x_train, x_test, y_train, y_test = train_test_split(X, Y_labels, test_size=0.2)

    return x_train, x_test, np.ravel(y_train), np.ravel(y_test), union_med



if __name__ == '__main__':

    disease_var = 'high_chol'

    x_train, x_test, y_train, y_test, union_med = load_clean(disease_var, days_drop_if_less=180, observation_window=59)

    print "predictions for " , disease_var

    rf = RandomForestClassifier()
    rf.fit(x_train, y_train)
    cross_val_score(rf, x_train, y_train)

    gb = GradientBoostingClassifier()
    gb.fit(x_train, y_train)
    cross_val_score(gb, x_train, y_train)

    y_hat_rf= rf.predict(x_test)
    print "RF Scores: "
    print "precision: " , precision_score(y_test, y_hat_rf)
    print "recall: " , recall_score(y_test, y_hat_rf)
    print "accuracy: " , accuracy_score(y_test, y_hat_rf)
    print "roc_auc: " , roc_auc_score(y_test, y_hat_rf)

    y_hat_gb = gb.predict(x_test)
    print "GB Scores: "
    print "precision: " , precision_score(y_test, y_hat_gb)
    print "recall: " , recall_score(y_test, y_hat_gb)
    print "accuracy: " , accuracy_score(y_test, y_hat_gb)
    print "roc_auc: " , roc_auc_score(y_test, y_hat_gb)


''' Some prelim results:

In [4]: run fitbit_classifiers
fitbit_classifiers.py:45: SettingWithCopyWarning:
A value is trying to be set on a copy of a slice from a DataFrame.
Try using .loc[row_indexer,col_indexer] = value instead

See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
  mini_med_cond['CAD']=mini_med_cond.apply(lambda row: combine_CAD(row), axis=1)
fitbit_classifiers.py:62: SettingWithCopyWarning:
A value is trying to be set on a copy of a slice from a DataFrame

See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
  Y_labels.replace(to_replace=3, value=2, inplace=True)
fitbit_classifiers.py:64: SettingWithCopyWarning:
A value is trying to be set on a copy of a slice from a DataFrame

See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
  Y_labels.replace(to_replace=2, value=0, inplace=True) #changing no disease_var to 0 from 2
predictions for  CAD
RF Scores:
precision:  0.0
recall:  0.0
accuracy:  0.891525423729
roc_auc:  0.496226415094
GB Scores:
precision:  0.0
recall:  0.0
accuracy:  0.894915254237
roc_auc:  0.498113207547

In [5]: run fitbit_classifiers
fitbit_classifiers.py:45: SettingWithCopyWarning:
A value is trying to be set on a copy of a slice from a DataFrame.
Try using .loc[row_indexer,col_indexer] = value instead

See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
  mini_med_cond['CAD']=mini_med_cond.apply(lambda row: combine_CAD(row), axis=1)
fitbit_classifiers.py:62: SettingWithCopyWarning:
A value is trying to be set on a copy of a slice from a DataFrame

See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
  Y_labels.replace(to_replace=3, value=2, inplace=True)
fitbit_classifiers.py:64: SettingWithCopyWarning:
A value is trying to be set on a copy of a slice from a DataFrame

See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
  Y_labels.replace(to_replace=2, value=0, inplace=True) #changing no disease_var to 0 from 2
predictions for  chf
RF Scores:
precision:  0.0
recall:  0.0
accuracy:  0.972881355932
roc_auc:  0.498263888889
GB Scores:
precision:  0.0
recall:  0.0
accuracy:  0.962711864407
roc_auc:  0.493055555556

In [6]: run fitbit_classifiers
fitbit_classifiers.py:45: SettingWithCopyWarning:
A value is trying to be set on a copy of a slice from a DataFrame.
Try using .loc[row_indexer,col_indexer] = value instead

See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
  mini_med_cond['CAD']=mini_med_cond.apply(lambda row: combine_CAD(row), axis=1)
fitbit_classifiers.py:62: SettingWithCopyWarning:
A value is trying to be set on a copy of a slice from a DataFrame

See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
  Y_labels.replace(to_replace=3, value=2, inplace=True)
fitbit_classifiers.py:64: SettingWithCopyWarning:
A value is trying to be set on a copy of a slice from a DataFrame

See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
  Y_labels.replace(to_replace=2, value=0, inplace=True) #changing no disease_var to 0 from 2
predictions for  diabetes
RF Scores:
precision:  1.0
recall:  0.05
accuracy:  0.935810810811
roc_auc:  0.525
GB Scores:
precision:  0.666666666667
recall:  0.1
accuracy:  0.935810810811
roc_auc:  0.548188405797

In [7]: run fitbit_classifiers
fitbit_classifiers.py:45: SettingWithCopyWarning:
A value is trying to be set on a copy of a slice from a DataFrame.
Try using .loc[row_indexer,col_indexer] = value instead

See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
  mini_med_cond['CAD']=mini_med_cond.apply(lambda row: combine_CAD(row), axis=1)
fitbit_classifiers.py:62: SettingWithCopyWarning:
A value is trying to be set on a copy of a slice from a DataFrame

See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
  Y_labels.replace(to_replace=3, value=2, inplace=True)
fitbit_classifiers.py:64: SettingWithCopyWarning:
A value is trying to be set on a copy of a slice from a DataFrame

See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
  Y_labels.replace(to_replace=2, value=0, inplace=True) #changing no disease_var to 0 from 2
predictions for  hbp
RF Scores:
precision:  0.566666666667
recall:  0.18085106383
accuracy:  0.695945945946
roc_auc:  0.558247314093
GB Scores:
precision:  0.435897435897
recall:  0.18085106383
accuracy:  0.665540540541
roc_auc:  0.53597008637

In [8]: run fitbit_classifiers
fitbit_classifiers.py:45: SettingWithCopyWarning:
A value is trying to be set on a copy of a slice from a DataFrame.
Try using .loc[row_indexer,col_indexer] = value instead

See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
  mini_med_cond['CAD']=mini_med_cond.apply(lambda row: combine_CAD(row), axis=1)
fitbit_classifiers.py:62: SettingWithCopyWarning:
A value is trying to be set on a copy of a slice from a DataFrame

See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
  Y_labels.replace(to_replace=3, value=2, inplace=True)
fitbit_classifiers.py:64: SettingWithCopyWarning:
A value is trying to be set on a copy of a slice from a DataFrame

See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
  Y_labels.replace(to_replace=2, value=0, inplace=True) #changing no disease_var to 0 from 2
predictions for  high_chol
RF Scores:
precision:  0.333333333333
recall:  0.198275862069
accuracy:  0.527210884354
roc_auc:  0.469924447888
GB Scores:
precision:  0.426470588235
recall:  0.25
accuracy:  0.571428571429
roc_auc:  0.515449438202

'''
